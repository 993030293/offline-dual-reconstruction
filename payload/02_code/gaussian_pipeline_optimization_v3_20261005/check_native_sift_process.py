"""Diagnose original CPU SIFT and a fixed two-process owned query broker."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from native_sift_process import NativeSiftProcess, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reference = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330/00_contract"
    split = json.loads((reference/"split_manifest.json").read_text(encoding="utf-8"))
    paths = [reference/"extra_images"/r["image_name"] for r in split["extra_frames"] if r["role"] == "H"][:6]
    sources = [Path(__file__), Path(__file__).with_name("native_sift_process.py")]
    source_ids, input_ids = [data_identity(p) for p in sources], [data_identity(p) for p in paths]
    cv2.setNumThreads(8)
    before = time.perf_counter()
    original = pycolmap.Sift(options(), pycolmap.Device.cpu)
    original_values = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        original_values.append(original.extract(gray))
    original_seconds = time.perf_counter()-before
    before = time.perf_counter()
    broker = NativeSiftProcess(paths)
    actual, rows = [], []
    try:
        proxy = broker.bind(pycolmap.Sift(options(), pycolmap.Device.cpu))
        for path, expected in zip(paths, original_values):
            current = proxy.extract(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            exact = all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes() for a, b in zip(current, expected))
            rows.append({"name": path.name, "feature_count": len(current[0]), "all_bytes_shape_dtype_exact": exact})
            actual.append(current)
    finally:
        broker.close()
    current_seconds = time.perf_counter()-before
    checks = {"all6_complete_native_features_exact": len(rows) == 6 and all(r["all_bytes_shape_dtype_exact"] for r in rows),
        "exact_two_workers_and_queue2": broker.trace["workers"] == 2 and broker.trace["peak_pending"] == 2
            and len(broker.trace["saved_workers"]) == 2 and broker.trace["submitted"] == broker.trace["delivered"] == 6,
        "all_actual_workers_closed": broker.closed and broker.trace["closed"],
        "sources_unchanged": source_ids == [data_identity(p) for p in sources],
        "inputs_unchanged": input_ids == [data_identity(p) for p in paths]}
    atomic_json(args.output/"RESULT.json", {"status": "diagnostic_native_bytes_passed" if all(checks.values()) else "required_native_guard_failed",
        "checks": checks, "rows": rows, "trace": broker.trace, "sources": source_ids, "inputs": input_ids,
        "original_complete_batch_seconds": original_seconds, "candidate_complete_batch_seconds": current_seconds,
        "diagnostic_speed_reduction": (original_seconds-current_seconds)/original_seconds,
        "whole_seconds": time.perf_counter()-started, "counted": False,
        "formal_full24_48_cost": "not_run"})
    print(json.dumps({"checks_passed": all(checks.values()), "diagnostic_reduction": (original_seconds-current_seconds)/original_seconds,
        "original_seconds": original_seconds, "candidate_seconds": current_seconds, "counted": False}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
