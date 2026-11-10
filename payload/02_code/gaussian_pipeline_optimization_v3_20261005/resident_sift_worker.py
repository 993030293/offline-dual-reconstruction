"""Preserve native integer distances using content-verified private CUDA features."""

import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch

from descriptor_residency_plan import DescriptorResidencyPlan, MAXIMUM_CACHE_BYTES, compact_plan
from generic_sift_worker import admitted, INPUT_BYTES, REPLY_BYTES, MATRIX_ELEMENTS
import integer_sift_worker as native


def nearest_resident(query, entries):
    nq = len(query)
    pq = max(32, (nq+15)//16*16)
    q = torch.zeros((pq, 128), dtype=torch.int8, device="cuda")
    q[:nq] = torch.from_numpy(query.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
    qi = q.to(torch.int32)
    qnorm = (qi*qi).sum(1, dtype=torch.int32)
    pending, results = list(entries), []
    while pending:
        group, rows = [], 0
        while pending and (not group or pq*((rows+len(pending[0][0])+7)//8*8) <= MATRIX_ELEMENTS):
            entry = pending.pop(0)
            group.append(entry)
            rows += len(entry[0])
        pt = max(8, (rows+7)//8*8)
        t = torch.zeros((pt, 128), dtype=torch.int8, device="cuda")
        norms = torch.zeros(pt, dtype=torch.int32, device="cuda")
        t[:rows] = torch.cat([value for value, _ in group], dim=0)
        norms[:rows] = torch.cat([norm for _, norm in group], dim=0)
        matrix = qnorm[:, None]+norms[None, :]-2*torch._int_mm(q, t.T.contiguous())
        offset = 0
        for train, _ in group:
            distance = matrix[:nq, offset:offset+len(train)]
            first, i0 = torch.min(distance, dim=1)
            _, reverse = torch.min(distance, dim=0)
            distance.scatter_(1, i0[:, None], torch.iinfo(torch.int32).max)
            second, i1 = torch.min(distance, dim=1)
            forward = torch.stack((i0.to(torch.int32), first, i1.to(torch.int32), second), dim=1).cpu().numpy()
            reverse = reverse.to(torch.int32).cpu().numpy()
            if (forward[:, (1, 3)] < 0).any() or (forward[:, (1, 3)] > 128*255**2).any():
                raise ValueError("An exact resident distance escaped its original mathematical bound")
            results.append((forward, reverse))
            offset += len(train)
        del matrix, t, norms
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-bytes", type=int, default=MAXIMUM_CACHE_BYTES)
    args = parser.parse_args()
    plan = DescriptorResidencyPlan(args.cache_bytes)
    if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("The resident worker requires the existing pinned CUDA runtime")
    torch.set_num_threads(4)
    torch.cuda.init()
    free, _ = torch.cuda.mem_get_info()
    if free < 6*1024**3:
        raise RuntimeError("The original CUDA4GiB budget and2GiB headroom are required")
    resident, peak_resident, total_wire, total_original = {}, 0, 0, 0
    stream = sys.stdin.buffer
    while header := stream.read(native.HEADER.size):
        if len(header) != native.HEADER.size:
            raise EOFError("An owned resident matching header was truncated")
        size = native.HEADER.unpack(header)[0]
        if not 0 < size <= 4096:
            raise ValueError("The original owned metadata budget was exceeded")
        request = json.loads(native.read_exact(stream, size))
        try:
            if request["command"] == "info":
                native.send({"status": "ready", "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                    "source_version": 1, "cache_byte_budget": plan.limit, "input_byte_budget": INPUT_BYTES,
                    "reply_byte_budget": REPLY_BYTES, "matrix_element_budget": MATRIX_ELEMENTS,
                    "content_verified_residency_independent_of_scene_identity": True})
            elif request["command"] == "stop":
                resident.clear()
                plan.clear()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                native.send({"status": "stopped", "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "peak_resident_descriptor_bytes": peak_resident,
                    "resident_entries_after_clear": len(resident), "wire_descriptor_bytes": total_wire,
                    "original_wire_descriptor_bytes": total_original})
                return
            elif request["command"] == "resident_batch":
                nq, items = request["query_rows"], request["train_items"]
                counts = [item[1] for item in items]
                if not admitted(nq, counts):
                    raise ValueError("The unchanged transport, reply or matrix budget excludes this request")
                expected = plan.plan(items)
                wire_plan = compact_plan(expected)
                if request["plan"] != wire_plan:
                    raise ValueError("The independently computed residency plans differ")
                for key in expected["evicted"]:
                    del resident[key]
                payload_size = (nq+sum(count for _, count in expected["missing"]))*128
                payload = native.read_exact(stream, payload_size)
                query = np.frombuffer(payload, np.uint8, count=nq*128).reshape(nq, 128)
                started = time.perf_counter()
                torch.cuda.empty_cache()
                offset = nq*128
                for key, count in expected["missing"]:
                    data = memoryview(payload)[offset:offset+count*128]
                    if hashlib.sha256(data).hexdigest() != key:
                        raise ValueError("The received complete descriptor bytes differ from their content identity")
                    raw = np.frombuffer(data, np.uint8).reshape(count, 128)
                    value = torch.from_numpy(raw.astype(np.int16)-128).to(device="cuda", dtype=torch.int8)
                    integer = value.to(torch.int32)
                    norm = (integer*integer).sum(1, dtype=torch.int32)
                    resident[key] = value, norm
                    del value, norm
                    del integer
                    offset += count*128
                if sum(value.numel()+4*norm.numel() for value, norm in resident.values()) != expected["resident_bytes"]:
                    raise ValueError("Actual CUDA feature residency differs from its byte plan")
                results = nearest_resident(query, [resident[key] for key, _ in items])
                torch.cuda.synchronize()
                if torch.cuda.max_memory_reserved() > 4*1024**3:
                    raise ValueError("The unchanged CUDA4GiB budget was exceeded")
                total_wire += payload_size
                total_original += (nq+sum(counts))*128
                peak_resident = max(peak_resident, expected["resident_bytes"])
                native.send({"status": "resident_batched", "query_rows": nq, "train_counts": counts,
                    "plan": wire_plan, "wire_descriptor_bytes": payload_size,
                    "original_wire_descriptor_bytes": (nq+sum(counts))*128,
                    "kernel_copy_seconds": time.perf_counter()-started}, b"".join(f.tobytes()+r.tobytes() for f, r in results))
            else:
                raise ValueError("An unsupported owned residency command was received")
        except Exception as error:
            native.send({"status": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
