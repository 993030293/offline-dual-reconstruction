"""Accept calibrated reference reuse from complete numerical and real-entry evidence."""

import argparse
import json
from pathlib import Path
import sys

from compare_actual_localization import compare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "cold", "lifecycle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result, lifecycle, cold = [read(p) for p in (args.campaign/"RESULT.json", args.lifecycle/"RESULT.json",
        args.cold/"REFERENCE_CACHE_TRACE.json")]
    left, right = args.campaign/"trial0/baseline/query_entry/entry", args.cold/"query_entry/entry"
    oracle = compare(left, right, left/"NEIGHBOR_TRACE.json", right/"NEIGHBOR_TRACE.json")
    atomic_json(args.output/"COLD_INVARIANTS.json", oracle)
    protocol = read(args.campaign/"PROTOCOL.json")
    identities = {Path(r["path"]).name: r["sha256"] for r in protocol["sources"]}
    checks = {"three_actual24_pairs_passed": result["status"] == "three_actual_pairs_passed_lifecycle_pending"
            and len(result["pairs"]) == 3 and all(p["all24_and48_protected"] and p["actual24_hits"] and p["full_process_protected"] for p in result["pairs"]),
        "cold_all24_and48_exact": oracle["all_protected"] and cold["cache"]["misses"] == 24 and cold["cache"]["hits"] == 0,
        "lifecycle_all21_passed": lifecycle["status"] == "actual_calibrated_cache_lifecycle_passed"
            and len(lifecycle["checks"]) == 21 and all(lifecycle["checks"].values()),
        "producer_source_current_and_bound": identities["calibrated_reference_cache.py"] ==
            {Path(r["path"]).name: r["sha256"] for r in lifecycle["sources"]}["calibrated_reference_cache.py"] ==
            data_identity(Path(__file__).with_name("calibrated_reference_cache.py"))["sha256"],
        "reference_stage_reduces10percent": result["reference_stage_reduction"] >= .1,
        "all_real_processes_memory_and_disk_admitted": result["cache_bytes"] <= 512*1024**2
            and all(v <= 8*1024**3 for p in result["pairs"] for v in p["peak_RSS_bytes"].values()),
        "frozen_sources_unchanged": result["sources_unchanged"]}
    median = result["median"]["complete_process_seconds"]
    acceptance = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed",
        "checks": checks, "actual_median": result["median"], "reference_stage_reduction": result["reference_stage_reduction"],
        "complete_entry_reduction": 1-median["cache"]/median["baseline"],
        "cold_complete_process_seconds": read(args.cold.parent/(args.cold.name+"_process.json"))["elapsed_seconds"],
        "cold_reference_seconds": result["cold_reference_seconds"], "cache_bytes": result["cache_bytes"],
        "deployment": "optional_repeated_calibrated_reference_export; cold_cost_retained",
        "stage_counted": all(checks.values()), "full_route_combination_accepted": False,
        "new_quality_or_accuracy_claim": False, "review": "structured_numerical_and_text_self_review"}
    atomic_json(args.output/"ACCEPTANCE.json", acceptance)
    print(json.dumps(acceptance), flush=True)
    if not all(checks.values()):
        raise AssertionError("Calibrated reference cache acceptance failed")


if __name__ == "__main__":
    main()
