"""Validate complete saved localization after a private receipt-path setup error."""

import argparse
from pathlib import Path
import time

STARTED = time.perf_counter()
from metric_branch_admission import read, OWNED
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    args.input.resolve().relative_to(OWNED.resolve())
    if (args.input/"RESULT.json").exists():
        raise FileExistsError("The saved private receipt review must be unique")
    contract = read(args.input/"CONTRACT.json")
    hits = list((args.input/"e").rglob("H_LOCALIZATION.json"))
    if len(hits) != 1:
        raise AssertionError("Exactly one actual original localization receipt is required")
    original = read(hits[0])
    integer = read(args.input/"e/entry/entry/INTEGER_SIFT_TRACE.json")
    trace = integer["trace"]
    resource = read(args.input/"e/entry/entry/WSL_RESOURCE.json")
    bindings = read(args.input/"BINDING_RESTORATION.json")
    checks = {"actual_complete24": original["status"] == "all24_localized_and_frozen" and len(original["rows"]) == 24,
        "original_all48_requests": integer["status"] == "actual24_completed" and integer["all48_geometry_calls_unchanged"]
            and len(trace["calls"]) == 48,
        "all24_loc_eval_disjoint": all(row["loc_eval_3D_overlap"] == 0 for row in original["rows"]),
        "sources_unchanged": contract["sources"] == [data_identity(p["path"]) for p in contract["sources"]],
        "inputs_unchanged": contract["inputs"] == [data_identity(p["path"]) for p in contract["inputs"]],
        "original_stop_wait": trace["returncode"] == 0 and "shutdown" in trace,
        "original_GPU_bound": trace["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3,
        "Linux_RSS_closed": resource["closed"] and resource["status"] == "native_worker_returned" and resource["peak_RSS_bytes"] <= 32*1024**3,
        "private_bindings_restored": bindings["worker_restored"] and bindings["argv_restored"]}
    result = {"status": "own_all24_localization_passed_training_quality_pending" if all(checks.values()) else "saved_own_localization_guard_failed",
        "checks": checks, "localization": str(hits[0]), "worker": trace,
        "reviewed_saved_full_native_outputs_only": True, "native_localization_rerun": False,
        "checker_issue": "Private entry directory was shortened to e; final wrapper searched stale entry directory",
        "original_failed_wrapper_log": str(OWNED/"SERIAL_LOCALIZATION_002.log"),
        "review_seconds": time.perf_counter()-STARTED, "map_pose_comparison_is_not_GT_accuracy": True, "counted": False}
    atomic_json(args.input/"RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("A complete saved original localization guard failed")


if __name__ == "__main__":
    main()
