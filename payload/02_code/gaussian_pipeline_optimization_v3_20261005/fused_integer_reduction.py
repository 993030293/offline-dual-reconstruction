"""Reduce the unchanged native integer matrix with exact smallest-index ties."""

import numpy as np
import torch
import triton
import triton.language as tl

from integer_sift_worker import LIMIT


@triton.jit
def forward_two(matrix, output, stride, columns, offset, payload_base, BLOCK: tl.constexpr):
    query = tl.program_id(0)
    train = tl.arange(0, BLOCK)
    distances = tl.load(matrix+query*stride+offset+train, mask=train < columns, other=2147483647)
    first, index0 = tl.min(distances, axis=0, return_indices=True, return_indices_tie_break_left=True)
    without_first = tl.where(train == index0, 2147483647, distances)
    second, index1 = tl.min(without_first, axis=0, return_indices=True, return_indices_tie_break_left=True)
    row = payload_base+query*4
    tl.store(output+row, index0.to(tl.int32))
    tl.store(output+row+1, first)
    tl.store(output+row+2, index1.to(tl.int32))
    tl.store(output+row+3, second)


@triton.jit
def reverse_one(matrix, output, stride, rows, columns, offset, payload_base,
                BLOCK_Q: tl.constexpr, BLOCK_T: tl.constexpr):
    train = tl.program_id(0)*BLOCK_T+tl.arange(0, BLOCK_T)
    queries = tl.arange(0, BLOCK_Q)
    best = tl.full((BLOCK_T,), 2147483647, tl.int32)
    index = tl.full((BLOCK_T,), 0, tl.int32)
    for start in range(0, rows, BLOCK_Q):
        query = start+queries
        distance = tl.load(matrix+query[:, None]*stride+offset+train[None, :],
            mask=(query[:, None] < rows) & (train[None, :] < columns), other=2147483647)
        value, local = tl.min(distance, axis=0, return_indices=True, return_indices_tie_break_left=True)
        better = value < best
        best = tl.where(better, value, best)
        index = tl.where(better, start+local.to(tl.int32), index)
    tl.store(output+payload_base+train, index, mask=train < columns)


def nearest_many(query, trains):
    """Keep native matrix math and ABI, and synchronize one complete output."""
    nq = len(query)
    if not 1 <= nq <= LIMIT or not 1 <= len(trains) <= 12 or any(not 2 <= len(t) <= LIMIT for t in trains):
        raise ValueError("The fixed native neighbor batch shape bound failed")
    if torch.__version__ != "2.7.1+cu128" or triton.__version__ != "3.3.1":
        raise RuntimeError("Use the existing pinned Torch and Triton versions")
    pq = max(32, (nq+15)//16*16)
    q = torch.zeros((pq, 128), dtype=torch.int8, device="cuda")
    q[:nq] = torch.from_numpy(query.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
    qi = q.to(torch.int32)
    qnorm = (qi*qi).sum(1, dtype=torch.int32)
    counts = [len(t) for t in trains]
    total = nq*4*len(trains)+sum(counts)
    packed = torch.empty(total, dtype=torch.int32, device="cuda")
    pending, payload_base = list(trains), 0
    while pending:
        group, rows = [], 0
        while pending and (not group or pq*((rows+len(pending[0])+7)//8*8) <= 144_000_000):
            value = pending.pop(0)
            group.append(value)
            rows += len(value)
        pt = max(8, (rows+7)//8*8)
        t = torch.zeros((pt, 128), dtype=torch.int8, device="cuda")
        raw = np.concatenate(group, axis=0).astype(np.int16)-128
        t[:rows] = torch.from_numpy(raw).to(device="cuda", dtype=torch.int8)
        ti = t.to(torch.int32)
        norms = (ti*ti).sum(1, dtype=torch.int32)
        matrix = qnorm[:, None]+norms[None, :]-2*torch._int_mm(q, t.T.contiguous())
        offset = 0
        for train in group:
            count = len(train)
            forward_two[(nq,)](matrix, packed, pt, count, offset, payload_base,
                BLOCK=triton.next_power_of_2(count), num_warps=4 if count <= 4096 else 8)
            reverse_one[(triton.cdiv(count, 32),)](matrix, packed, pt, nq, count, offset,
                payload_base+nq*4, BLOCK_Q=256, BLOCK_T=32, num_warps=4)
            payload_base += nq*4+count
            offset += count
        del matrix, t, ti
    host = packed.cpu().numpy()
    results, offset = [], 0
    for count in counts:
        forward = host[offset:offset+nq*4].reshape(nq, 4)
        offset += nq*4
        reverse = host[offset:offset+count]
        offset += count
        if (forward[:, (1, 3)] < 0).any() or (forward[:, (1, 3)] > 128*255**2).any():
            raise ValueError("An exact batched distance escaped its mathematical bound")
        results.append((forward, reverse))
    if offset != total:
        raise AssertionError("The complete native result ABI changed")
    return results
