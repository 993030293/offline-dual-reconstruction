"""Admit exact integer SIFT batches by transport and matrix memory budgets."""

import json
import sys
import time

import numpy as np
import torch

import integer_sift_worker as native

INPUT_BYTES = 32*1024**2
REPLY_BYTES = 12*12000*20
MATRIX_ELEMENTS = 144_000_000


def admitted(nq, counts):
    if type(nq) is not int or nq < 1 or not isinstance(counts, list) or not 1 <= len(counts) <= 12:
        return False
    if any(type(nt) is not int or nt < 2 for nt in counts):
        return False
    pq = max(32, (nq+15)//16*16)
    return ((nq+sum(counts))*128 <= INPUT_BYTES
        and nq*16*len(counts)+sum(counts)*4 <= REPLY_BYTES
        and all(pq*max(8, (nt+7)//8*8) <= MATRIX_ELEMENTS for nt in counts))


def main():
    if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("The admitted worker requires the existing pinned CUDA runtime")
    torch.set_num_threads(4)
    torch.cuda.init()
    free, _ = torch.cuda.mem_get_info()
    if free < 6*1024**3:
        raise RuntimeError("The worker requires its original4GiB budget and2GiB free headroom")
    # Row admission comes from bytes and matrix bounds, independent of scene identity.
    native.LIMIT = INPUT_BYTES//128
    stream = sys.stdin.buffer
    while header := stream.read(native.HEADER.size):
        if len(header) != native.HEADER.size:
            raise EOFError("An owned matching header was truncated")
        size = native.HEADER.unpack(header)[0]
        if not 0 < size <= 4096:
            raise ValueError("The owned metadata size bound failed")
        request = json.loads(native.read_exact(stream, size))
        try:
            if request["command"] == "info":
                native.send({"status": "ready", "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                    "source_version": 2, "input_byte_budget": INPUT_BYTES, "reply_byte_budget": REPLY_BYTES,
                    "matrix_element_budget": MATRIX_ELEMENTS, "shape_admission_is_scene_independent": True})
            elif request["command"] == "stop":
                native.send({"status": "stopped", "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()})
                return
            elif request["command"] in ("batch", "nearest"):
                nq = request["query_rows"]
                counts = request["train_counts"] if request["command"] == "batch" else [request["train_rows"]]
                if not admitted(nq, counts):
                    raise ValueError("The original matrix or transport budget excludes this shape")
                payload = native.read_exact(stream, (nq+sum(counts))*128)
                query = np.frombuffer(payload, np.uint8, count=nq*128).reshape(nq, 128)
                trains, offset = [], nq*128
                for count in counts:
                    trains.append(np.frombuffer(payload, np.uint8, count=count*128, offset=offset).reshape(count, 128))
                    offset += count*128
                started = time.perf_counter()
                # Prior requests leave no live CUDA tensors; release cached shapes before allocation.
                torch.cuda.empty_cache()
                results = native.nearest_many(query, trains)
                torch.cuda.synchronize()
                if torch.cuda.max_memory_reserved() > 4*1024**3:
                    raise ValueError("The unchanged CUDA4GiB budget was exceeded")
                record = {"status": "batched" if request["command"] == "batch" else "matched", "query_rows": nq,
                    "kernel_copy_seconds": time.perf_counter()-started}
                record.update(train_counts=counts) if request["command"] == "batch" else record.update(train_rows=counts[0])
                native.send(record, b"".join(f.tobytes()+r.tobytes() for f, r in results))
            else:
                raise ValueError("An unsupported owned matching command was received")
        except Exception as error:
            native.send({"status": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
