"""Serve bounded exact integer SIFT distances over an owned binary stdio pipe."""

import json
import struct
import sys
import time

import numpy as np
import torch

HEADER = struct.Struct("<I")
LIMIT = 12000


def read_exact(stream, count):
    values = bytearray()
    while len(values) < count:
        block = stream.read(count-len(values))
        if not block:
            raise EOFError("A complete owned matching request was truncated")
        values.extend(block)
    return values


def send(record, payload=b""):
    metadata = json.dumps(dict(record, payload_bytes=len(payload)), separators=(",", ":")).encode()
    stream = sys.stdout.buffer
    stream.write(HEADER.pack(len(metadata)))
    stream.write(metadata)
    stream.write(payload)
    stream.flush()


def nearest(query, train):
    nq, nt = len(query), len(train)
    if not 1 <= nq <= LIMIT or not 2 <= nt <= LIMIT or query.shape[1] != 128 or train.shape[1] != 128:
        raise ValueError("The exact native SIFT distance request exceeds its fixed shape bound")
    pq, pt = max(32, (nq+15)//16*16), max(8, (nt+7)//8*8)
    q = torch.zeros((pq, 128), dtype=torch.int8, device="cuda")
    t = torch.zeros((pt, 128), dtype=torch.int8, device="cuda")
    q[:nq] = torch.from_numpy(query.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
    t[:nt] = torch.from_numpy(train.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
    qi, ti = q.to(torch.int32), t.to(torch.int32)
    qnorm = (qi*qi).sum(1, dtype=torch.int32)
    tnorm = (ti*ti).sum(1, dtype=torch.int32)
    distance = qnorm[:, None]+tnorm[None, :]-2*torch._int_mm(q, t.T.contiguous())
    distance = distance[:nq, :nt].contiguous()
    first, index0 = torch.min(distance, dim=1)
    _, reverse = torch.min(distance, dim=0)
    distance.scatter_(1, index0[:, None], torch.iinfo(torch.int32).max)
    second, index1 = torch.min(distance, dim=1)
    integers = torch.stack((index0.to(torch.int32), first, index1.to(torch.int32), second), dim=1).cpu().numpy()
    reversed_indices = reverse.to(torch.int32).cpu().numpy()
    if (integers[:, (1, 3)] < 0).any() or (integers[:, (1, 3)] > 128*255**2).any():
        raise ValueError("An exact integer distance escaped its mathematical bound")
    return integers, reversed_indices


def nearest_many(query, trains):
    nq = len(query)
    if not 1 <= nq <= LIMIT or not 1 <= len(trains) <= 12 or any(not 2 <= len(t) <= LIMIT for t in trains):
        raise ValueError("The fixed native neighbor batch shape bound failed")
    pq = max(32, (nq+15)//16*16)
    q = torch.zeros((pq, 128), dtype=torch.int8, device="cuda")
    q[:nq] = torch.from_numpy(query.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
    qi = q.to(torch.int32)
    qnorm = (qi*qi).sum(1, dtype=torch.int32)
    results, pending = [], list(trains)
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
            distance = matrix[:nq, offset:offset+len(train)]
            first, i0 = torch.min(distance, dim=1)
            _, reverse = torch.min(distance, dim=0)
            distance.scatter_(1, i0[:, None], torch.iinfo(torch.int32).max)
            second, i1 = torch.min(distance, dim=1)
            forward = torch.stack((i0.to(torch.int32), first, i1.to(torch.int32), second), dim=1).cpu().numpy()
            reverse = reverse.to(torch.int32).cpu().numpy()
            if (forward[:, (1, 3)] < 0).any() or (forward[:, (1, 3)] > 128*255**2).any():
                raise ValueError("An exact batched distance escaped its mathematical bound")
            results.append((forward, reverse))
            offset += len(train)
        del matrix, t, ti
    return results


def main():
    if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Use the existing pinned WSL torch2.7.1 CUDA runtime only")
    torch.set_num_threads(4)
    torch.cuda.init()
    stream = sys.stdin.buffer
    while header := stream.read(HEADER.size):
        if len(header) != HEADER.size:
            raise EOFError("An owned matching metadata header was truncated")
        size = HEADER.unpack(header)[0]
        if not 0 < size <= 4096:
            raise ValueError("An owned matching metadata header exceeds its bound")
        request = json.loads(read_exact(stream, size))
        try:
            if request["command"] == "info":
                send({"status": "ready", "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                    "source_version": 1, "maximum_descriptor_rows": LIMIT})
            elif request["command"] == "stop":
                send({"status": "stopped", "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()})
                return
            elif request["command"] == "batch":
                nq, counts = request["query_rows"], request["train_counts"]
                if not isinstance(nq, int) or not 1 <= nq <= LIMIT or not 1 <= len(counts) <= 12 or any(type(n) is not int or not 2 <= n <= LIMIT for n in counts):
                    raise ValueError("A native SIFT batch descriptor bound failed")
                payload = read_exact(stream, (nq+sum(counts))*128)
                query = np.frombuffer(payload, np.uint8, count=nq*128).reshape(nq, 128)
                trains, offset = [], nq*128
                for count in counts:
                    trains.append(np.frombuffer(payload, np.uint8, count=count*128, offset=offset).reshape(count, 128))
                    offset += count*128
                started = time.perf_counter()
                results = nearest_many(query, trains)
                data = b"".join(f.tobytes()+r.tobytes() for f, r in results)
                send({"status": "batched", "query_rows": nq, "train_counts": counts,
                    "kernel_copy_seconds": time.perf_counter()-started}, data)
            elif request["command"] == "nearest":
                nq, nt = request["query_rows"], request["train_rows"]
                if not isinstance(nq, int) or not isinstance(nt, int) or not 1 <= nq <= LIMIT or not 2 <= nt <= LIMIT:
                    raise ValueError("The fixed native descriptor count bound failed")
                payload = read_exact(stream, (nq+nt)*128)
                query = np.frombuffer(payload, np.uint8, count=nq*128).reshape(nq, 128)
                train = np.frombuffer(payload, np.uint8, offset=nq*128).reshape(nt, 128)
                started = time.perf_counter()
                forward, reverse = nearest(query, train)
                send({"status": "matched", "query_rows": nq, "train_rows": nt,
                    "kernel_copy_seconds": time.perf_counter()-started}, forward.tobytes()+reverse.tobytes())
            else:
                raise ValueError("An unsupported owned matching request was received")
        except Exception as error:
            send({"status": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
