"""Freeze and verify matcher routing without semantic or object shape assumptions."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from generic_sift_matching import GenericWorker, admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).with_name(name) for name in
        ("generic_sift_worker.py", "generic_sift_matching.py", "integer_sift_worker.py", "integer_sift_matching.py")]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"PROTOCOL.json", {"sources": identities, "scene_or_GT_inputs": False,
        "maximum_CUDA_GiB": 4, "rows_not_an_object_specific_threshold": True, "new_optimization_count": 0})
    rng = np.random.default_rng(20261005)
    worker, rows = GenericWorker(args.output/"worker.log"), []
    try:
        for nq, nt in ((1, 2), (16, 17), (12001, 3), (4, 20003), (33, 2)):
            q = rng.integers(0, 256, (nq, 128), dtype=np.uint8)
            t = rng.integers(0, 256, (nt, 128), dtype=np.uint8)
            t[1] = t[0]
            q[0] = 0
            if nq > 1:
                q[1] = 255
            q, t = q.astype(np.float32)/512, t.astype(np.float32)/512
            route, reason = admission(q, [t])
            forward, distance, reverse = worker.match(q, t)
            bf = cv2.BFMatcher(cv2.NORM_L2)
            pairs, backward = bf.knnMatch(q, t, k=2), bf.match(t, q)
            exact = all(row[0] == pair[0].trainIdx and row[2] == pair[1].trainIdx
                and float(d[0]) == pair[0].distance and float(d[1]) == pair[1].distance
                for row, d, pair in zip(forward, distance, pairs)) and list(reverse) == [m.trainIdx for m in backward]
            batched = worker.batch(q, [t, t])
            rows.append({"query_rows": nq, "train_rows": nt, "admitted": route == "integer", "reason": reason,
                "all_forward_distances_indices_and_reverse_exact": bool(exact),
                "both_batched_results_exact": all(all(np.array_equal(x, y) for x, y in zip((forward, distance, reverse), r)) for r in batched)})
    finally:
        worker.close()
    good = np.zeros((32, 128), np.float32)
    native_cases = {
        "single_train_descriptor": admission(good, [good[:1]])[0] == "native",
        "empty_train_descriptor": admission(good, [good[:0]])[0] == "native",
        "empty_query": admission(good[:0], [good])[0] == "native",
        "non_quantized_float_descriptor": admission(np.full_like(good, .123456), [good])[0] == "native",
        "matrix_over_budget": admission(np.zeros((12001, 128), np.float32), [np.zeros((12001, 128), np.float32)])[0] == "native",
        "too_many_neighbors": admission(good, [good]*13)[0] == "native"
    }
    checks = {"all_large_small_tie_and_range_oracles": all(r["admitted"] and r["all_forward_distances_indices_and_reverse_exact"]
        and r["both_batched_results_exact"] for r in rows), "all_unsupported_cases_routed_before_execution": all(native_cases.values()),
        "worker_complete_and_original_CUDA_budget": worker.receipt["returncode"] == 0
            and worker.receipt["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3,
        "sources_unchanged": identities == [data_identity(p) for p in sources]}
    result = {"status": "general_shape_and_explicit_route_guards_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "oracle_rows": rows, "native_routes": native_cases, "worker": worker.receipt,
        "sources": identities, "new_optimization_count": 0, "real_object_quality_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks")})
    if not all(checks.values()):
        raise ValueError("A generic native matcher oracle or original budget failed")


if __name__ == "__main__":
    main()
