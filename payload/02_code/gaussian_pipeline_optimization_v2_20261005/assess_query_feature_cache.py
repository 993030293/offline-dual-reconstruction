"""Accept exact query caching only with actual cold, paired and lifecycle evidence."""

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
    result = read(args.campaign/"RESULT.json")
    lifecycle = read(args.lifecycle/"RESULT.json")
    cold = read(args.cold/"FEATURE_CACHE_TRACE.json")
    baseline = args.campaign/"trial0/baseline/entry"
    cold_entry = args.cold/"entry"
    oracle = compare(baseline, cold_entry, baseline/"NEIGHBOR_TRACE.json", cold_entry/"NEIGHBOR_TRACE.json")
    atomic_json(args.output/"COLD_INVARIANTS.json", oracle)
    bound = read(args.campaign/"PROTOCOL.json")
    identities = {Path(r["path"]).name: r["sha256"] for r in bound["sources"]}
    lifecycle_sources = {Path(r["path"]).name: r["sha256"] for r in lifecycle["sources"]}
    checks = {"current_source_three_actual_pairs": result["status"] == "three_actual_pairs_passed_lifecycle_pending"
            and len(result["pairs"]) == 3 and all(result["checks"].values()),
        "actual_cold_all24_arrays_poses_and48_matching_exact": oracle["all_protected"],
        "real_SIFT_lifecycle_all16": lifecycle["status"] == "real_SIFT_lifecycle_passed"
            and len(lifecycle["checks"]) == 16 and all(lifecycle["checks"].values()),
        "cache_implementation_current_and_lifecycle_bound": identities["query_feature_cache.py"] ==
            lifecycle_sources["query_feature_cache.py"] == data_identity(Path(__file__).with_name("query_feature_cache.py"))["sha256"],
        "actual_cold24_misses_no_hidden_warm": cold["cache"]["misses"] == 24 and cold["cache"]["hits"] == 0,
        "cache_disk_budget": cold["cache"]["cache_bytes"] <= 512*1024**2,
        "all_process_RSS_under8GiB": all(v <= 8*1024**3 for p in result["pairs"] for v in p["peak_process_RSS_bytes"].values())}
    median = result["median"]["complete_process_seconds"]
    acceptance = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed",
        "checks": checks, "actual_median": result["median"], "query_stage_reduction": result["query_stage_reduction"],
        "complete_entry_reduction": 1-median["cache"]/median["baseline"],
        "cold_complete_process_seconds": read(args.cold.parent/(args.cold.name+"_process.json"))["elapsed_seconds"],
        "cold_query_seconds": result["cold_query_feature_seconds"], "cache_bytes": result["cache_bytes"],
        "evidence": [data_identity(p) for p in (args.campaign/"RESULT.json", args.cold/"FEATURE_CACHE_TRACE.json",
            args.lifecycle/"RESULT.json", args.output/"COLD_INVARIANTS.json")],
        "deployment": "optional_repeated_localization; first extraction cost preserved",
        "stage_counted": all(checks.values()), "full_route_combination_accepted": False,
        "self_review_not_independent": True, "new_accuracy_or_quality_claim": False}
    atomic_json(args.output/"ACCEPTANCE.json", acceptance)
    print(json.dumps(acceptance), flush=True)
    if not all(checks.values()):
        raise AssertionError("Query cache evidence chain acceptance failed")


if __name__ == "__main__":
    main()
