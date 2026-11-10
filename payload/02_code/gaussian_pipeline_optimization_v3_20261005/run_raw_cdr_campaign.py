"""Measure bulk CDR decoding in six actual full raw-selection entries."""

import argparse
import importlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs
from bulk_odometry_cdr import decode_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_raw_cdr.py"), Path(__file__).with_name("bulk_odometry_cdr.py"), V2/"run_effective_initial.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "trials": 3,
        "incumbent": "current_actual_Q78_Q50_Q53_raw_entry", "only_change": "contiguous_pose7_and_covariance36_CDR_decode",
        "minimum_actual_pose_and_calibration_phase_reduction": .1, "maximum_complete_entry_time_ratio": 1.1,
        "all7338_decisions1002_JPEG11583_pairs_and_raw_odometry_exact": True,
        "quality_cache_common_warm_JPEG_artifact_cache_disabled": True,
        "maximum_seconds_each": 180, "maximum_RSS_GiB": 8, "counted": False})
    sys.path.insert(0, str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import Cdr
    lifecycle = {}
    for count in (7, 36):
        values = [-0., 1., -3.125, 1e-100, 1e100, float("inf"), float("nan")]+[0.]*(count-7)
        for offset in range(4, 20):
            start = offset+(-(offset-4)) % 8
            raw = bytes(start)+struct.pack("<"+str(count)+"d", *values)+b"tail"
            left, right = Cdr(raw), Cdr(raw)
            left.offset = right.offset = offset
            a, b = [left.f64() for _ in range(count)], decode_array(right, count)
            lifecycle[f"count{count}_offset{offset}_bit_exact_and_final_offset"] = struct.pack("<"+str(count)+"d", *a) == struct.pack("<"+str(count)+"d", *b) and left.offset == right.offset
            broken = Cdr(raw[:start+8*count-1])
            broken.offset = offset
            try:
                decode_array(broken, count)
                lifecycle[f"count{count}_offset{offset}_truncated_rejected"] = False
            except struct.error:
                lifecycle[f"count{count}_offset{offset}_truncated_rejected"] = True
    atomic_json(args.output/"LIFECYCLE.json", {"checks": lifecycle, "counted": False})
    if not all(lifecycle.values()):
        raise ValueError("The native CDR value or final offset contract failed")
    pairs = []
    for trial in range(3):
        folder = args.output/f"trial{trial}"
        folder.mkdir()
        durations, phase = {}, {}
        for mode in (["native", "bulk"] if trial % 2 == 0 else ["bulk", "native"]):
            target = folder/mode
            command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                "--log", str(folder/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                sys.executable, str(Path(__file__).with_name("run_raw_cdr.py")), "--decoder", mode, "--output", str(target),
                "--quality-cache", str(args.quality_cache)]
            subprocess.run(command, check=True)
            process = json.loads((folder/(mode+"_process.json")).read_text())
            receipt = json.loads((target/"initial/SELECTION_RECEIPT.json").read_text())
            durations[mode] = process["elapsed_seconds"]
            phase[mode] = receipt["timings"]["raw_pose_and_calibration_seconds"]
        protected = compare_initial_outputs(folder/"native/initial", folder/"bulk/initial")
        row = {"trial": trial, "complete_wall_seconds": durations, "pose_calibration_seconds": phase,
            "all_protected": protected["all_protected"], "checks": protected["checks"],
            "complete_time_protected": durations["bulk"] <= 1.1*durations["native"]}
        pairs.append(row)
        atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs})
        print(json.dumps(row), flush=True)
        if not row["all_protected"]:
            raise ValueError("A native raw decision, odometry value, JPEG or pair changed")
    medians = {kind: {mode: statistics.median(row[kind][mode] for row in pairs) for mode in ("native", "bulk")}
        for kind in ("complete_wall_seconds", "pose_calibration_seconds")}
    reduction = 1-medians["pose_calibration_seconds"]["bulk"]/medians["pose_calibration_seconds"]["native"]
    checks = {"all_original_full_outputs_exact": all(row["all_protected"] for row in pairs),
        "actual_decoder_phase_reduces10percent": reduction >= .1,
        "all_complete_wall_guards": all(row["complete_time_protected"] for row in pairs),
        "all64_lifecycle_checks": len(lifecycle) == 64 and all(lifecycle.values()),
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    atomic_json(args.output/"RESULT.json", {"status": "real_raw_stage_passed_entry_bound" if all(checks.values()) else "failed_verification",
        "checks": checks, "median": medians, "phase_reduction": reduction, "pairs": pairs, "counted": False})
    print(json.dumps({"checks": checks, "median": medians, "phase_reduction": reduction}), flush=True)
    if not all(checks.values()):
        raise ValueError("An actual raw decoding phase or full-entry original guard failed")


if __name__ == "__main__":
    main()
