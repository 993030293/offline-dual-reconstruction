"""Exercise actual GPU reductions, complete integer outputs, and shape errors."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

STARTED = time.perf_counter()
import numpy as np
import psutil
import torch
import triton

import integer_sift_worker as native
from fused_integer_reduction import nearest_many


def save(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if torch.__version__ != "2.7.1+cu128" or triton.__version__ != "3.3.1" or not torch.cuda.is_available():
        raise RuntimeError("The existing pinned CUDA runtime is required")
    torch.cuda.init()
    process, closed, peak = psutil.Process(), threading.Event(), [0]
    ident = {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time(), "counted": False}
    save(args.output/"PROCESS_IDENTITY.json", ident)
    def sample():
        while not closed.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
    observer = threading.Thread(target=sample, name="owned-GPU-oracle-RSS")
    observer.start()
    paths = [Path(__file__), Path(__file__).with_name("fused_integer_reduction.py"), Path(native.__file__),
        Path(__file__).with_name("fused_integer_match_worker.py")]
    sources = [identity(p) for p in paths]
    rows, checks = [], {}
    try:
        rng = np.random.default_rng(121)
        cases = [(1, [2], "random"), (17, [7, 91], "random"), (513, [1025, 1026], "random"),
            (89, [89]*12, "ties"), (33, [2, 63, 129], "zeros"),
            (65, [2, 31, 127], "extremes"), (257, [8193, 12000], "random"),
            (12000, [12000], "random"), (12000, [12000, 2], "random")]
        for nq, counts, kind in cases:
            query = rng.integers(0, 256, (nq, 128), dtype=np.uint8)
            trains = [rng.integers(0, 256, (n, 128), dtype=np.uint8) for n in counts]
            if kind == "zeros":
                query[:] = 0
                for train in trains:
                    train[:] = 0
            elif kind == "ties":
                query[1::2] = query[::2][:len(query[1::2])]
                for train in trains:
                    train[1::2] = train[::2][:len(train[1::2])]
            elif kind == "extremes":
                query[:] = 0
                for train in trains:
                    train[:] = 255
                    train[::2] = 0
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            old = native.nearest_many(query, trains)
            torch.cuda.synchronize()
            native_peak = torch.cuda.max_memory_reserved()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            new = nearest_many(query, trains)
            torch.cuda.synchronize()
            candidate_peak = torch.cuda.max_memory_reserved()
            exact = len(old) == len(new) == len(trains) and all(
                a.dtype == b.dtype == np.int32 and a.shape == b.shape and a.tobytes() == b.tobytes()
                for left, right in zip(old, new) for a, b in zip(left, right))
            sqrt_exact = all(np.sqrt(a[0][:, (1, 3)].astype(np.float32)/262144).tobytes()
                == np.sqrt(b[0][:, (1, 3)].astype(np.float32)/262144).tobytes() for a, b in zip(old, new))
            rows.append({"nq": nq, "train_counts": counts, "kind": kind, "all_int32_bytes_exact": exact,
                "float32_sqrt_bytes_exact": sqrt_exact, "native_CUDA_reserved_peak": native_peak,
                "candidate_CUDA_reserved_peak": candidate_peak})
            save(args.output/"STATE.json", {"status": "running", "rows": rows, "sources": sources, "counted": False})
            if not exact or not sqrt_exact or max(native_peak, candidate_peak) > 4*1024**3:
                raise AssertionError("An exact GPU or original CUDA4GiB guard failed")
        query = np.zeros((1, 128), np.uint8)
        for tag, q, trains in (("empty_query", query[:0], [np.zeros((2, 128), np.uint8)]),
            ("one_train_row", query, [np.zeros((1, 128), np.uint8)]),
            ("too_many_neighbors", query, [np.zeros((2, 128), np.uint8)]*13),
            ("over_query_limit", np.zeros((12001, 128), np.uint8), [np.zeros((2, 128), np.uint8)])):
            errors = []
            for function in (native.nearest_many, nearest_many):
                try:
                    function(q, trains)
                except Exception as error:
                    errors.append((type(error).__name__, str(error)))
                else:
                    errors.append(None)
            checks[tag+"_exact_original_exception"] = errors[0] == errors[1] and errors[0] is not None
        checks["all9_actual_complete_GPU_shapes_and_sqrt_exact"] = len(rows) == 9 and all(
            r["all_int32_bytes_exact"] and r["float32_sqrt_bytes_exact"] for r in rows)
        checks["all9_both_CUDA4GiB_bounds"] = all(max(r["native_CUDA_reserved_peak"], r["candidate_CUDA_reserved_peak"]) <= 4*1024**3 for r in rows)
        checks["complete_sources_unchanged"] = sources == [identity(p) for p in paths]
    except BaseException as error:
        save(args.output/"FAILURE.json", {"error": repr(error), "rows": rows, "sources": sources,
            "whole_seconds": time.perf_counter()-STARTED, "counted": False})
        raise
    finally:
        torch.cuda.synchronize()
        closed.set()
        observer.join()
        peak[0] = max(peak[0], process.memory_info().rss)
        save(args.output/"WSL_RESOURCE.json", {**ident, "closed": not observer.is_alive(),
            "peak_RSS_bytes": peak[0], "whole_seconds": time.perf_counter()-STARTED})
    checks["Linux32GiB_and_sampler_closed"] = peak[0] <= 32*1024**3 and not observer.is_alive()
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "rows": rows, "sources": sources, "whole_seconds_includes_first_JIT": time.perf_counter()-STARTED,
        "torch": torch.__version__, "triton": triton.__version__, "device": torch.cuda.get_device_name(), "counted": False}
    save(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks", "whole_seconds_includes_first_JIT", "device")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An original complete GPU matching contract failed")


if __name__ == "__main__":
    main()
