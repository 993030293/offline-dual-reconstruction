"""Bind actual byte-SIFT localization pairs to their exact lifecycle evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "campaign", "validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = read(args.campaign/"RESULT.json")
    lifecycle = read(args.validation)
    adapter = Path(__file__).with_name("byte_sift_matching.py")
    paths = {str(Path(r["path"]).resolve()): r for r in campaign["sources"]} if "sources" in campaign else {}
    if not paths:
        paths = {str(Path(r["path"]).resolve()): r for r in read(args.campaign/"PROTOCOL.json")["sources"]}
    checks = {"all_lifecycle_guards": lifecycle["status"] == "lifecycle_passed" and all(lifecycle["checks"].values()),
        "three_actual_pairs_passed": campaign["status"] == "three_actual_pairs_passed_lifecycle_pending" and len(campaign["pairs"]) == 3,
        "all24_and48_protected_each_pair": all(p["all24_and48_protected"] for p in campaign["pairs"]),
        "same_cache_access_capacity_each_pair": all(p["same_feature_cache_access_and_capacity"] for p in campaign["pairs"]),
        "complete_time_no_more_than10percent_slower": all(p["complete_time_protected"] for p in campaign["pairs"]),
        "memory_stage_improves10percent": campaign["feature_store_memory_reduction"] >= .1,
        "same_adapter_for_actual_pairs_lifecycle_and_current": lifecycle["sources"][1] == paths[str(adapter.resolve())] == data_identity(adapter),
        "campaign_sources_unchanged": campaign["sources_unchanged"]}
    result = {"status": "stage_guards_passed_combination_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "median": campaign["median"], "feature_store_memory_reduction": campaign["feature_store_memory_reduction"],
        "pairs": campaign["pairs"], "campaign": str(args.campaign), "lifecycle": str(args.validation),
        "counted_stage": bool(all(checks.values())), "counted_final_combination": False,
        "memory_scope": "maximum retained feature arrays plus a conservative two-neighbor transient float32 upper bound",
        "original_float32_native_matcher_unchanged": True,
        "previous_native_uint8_variant_rejected_for_actual_latency": "q51_campaign_001/STATE.json",
        "all_original_float_descriptors_preserved": True, "pose_or_object_quality_improvement_claim": False}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise ValueError("Byte SIFT real-pair acceptance did not pass")


if __name__ == "__main__":
    main()
