"""Accept an actual bounded-memory repair separately from its rejected speed hypothesis."""

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
    for name in ("legacy-process", "campaign", "lifecycle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old, campaign, lifecycle = read(args.legacy_process), read(args.campaign/"RESULT.json"), read(args.lifecycle)
    processes = [read(args.campaign/f"trial{i}/candidate_process.json") for i in range(3)]
    original_bound = 8*1024**3
    checks = {"actual_original_whole_mapped_branch_failed_8GiB": old["status"] == "failed_controller" and old["maximum_RSS_bytes"] == original_bound
            and old["peak_tree_RSS_bytes"] > original_bound and old["elapsed_seconds"] < old["maximum_seconds"],
        "all7338_payload_bytes_and_metadata_exact": lifecycle["status"] == "lifecycle_passed" and all(lifecycle["checks"].values()),
        "three_full_raw_runs_completed_with_original_8GiB_limit": all(p["status"] == "completed" and p["returncode"] == 0
            and p["maximum_RSS_bytes"] == original_bound and p["peak_tree_RSS_bytes"] < original_bound for p in processes),
        "all_three7338_decisions1002_JPEG11583_pairs_and_odometry_exact": len(campaign["pairs"]) == 3 and all(p["all_protected"] for p in campaign["pairs"]),
        "source_identities_unchanged_during_actual_runs": campaign["sources_unchanged"] and all(data_identity(Path(row["path"])) == row for row in campaign["sources"]),
        "original_speed_rejection_preserved": campaign["status"] == "rejected_insufficient_wall_benefit"}
    peak = max(p["peak_tree_RSS_bytes"] for p in processes)
    result = {"status": "accepted_bounded_memory_failure_repair" if all(checks.values()) else "failed_verification",
        "mechanism": "Q55_BOUNDED_RAW_MAPPING", "checks": checks, "stage_counted": all(checks.values()),
        "initial_resource_failure": data_identity(args.legacy_process), "three_complete_runs": str(args.campaign),
        "original_peak_RSS_lower_bound_bytes": original_bound, "original_actual_censored_peak_bytes": old["peak_tree_RSS_bytes"],
        "maximum_three_completed_peak_RSS_bytes": peak, "peak_reduction_lower_bound": 1-peak/original_bound,
        "speed_hypothesis": "rejected; no speed improvement versus the accepted indexed reader",
        "complete_entry_seconds_vs_indexed": campaign["median"]["complete_entry_seconds"],
        "default_route": "keep original indexed reader", "deployment": "optional bounded mapping for the formerly resource-failing public mapped reader branch",
        "original_failure_denominator": "censored at resource exhaustion; never claimed complete",
        "correctness_resource_repair_intent_preceded_new_window_trials": True,
        "new_image_or_pose_quality_claim": False, "combination_pending": True,
        "sources": campaign["sources"], "lifecycle": str(args.lifecycle)}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "stage_counted", "maximum_three_completed_peak_RSS_bytes", "peak_reduction_lower_bound", "speed_hypothesis")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A resource repair cannot hide a speed rejection or incomplete source branch")


if __name__ == "__main__":
    main()
