"""Accept actual queued JPEG export only with complete raw-input and byte protections."""

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
    result = read(args.campaign/"RESULT.json")
    lifecycle = read(args.validation)
    adapter = Path(__file__).with_name("parallel_raw_export.py")
    sources = {str(Path(r["path"]).resolve()): r for r in result["sources"]}
    checks = {"three_complete_actual_raw_pairs": len(result["pairs"]) == 3,
        "all_original_decisions_JPEG_pairs_and_odometry_exact": all(r["all_protected"] for r in result["pairs"]),
        "all_queues_bounded_consumed1002_and_closed": all(r["all_queues_consumed1002_and_closed"] for r in result["pairs"]),
        "complete_time_no_more_than10percent_slower_each_pair": all(r["complete_time_protected"] for r in result["pairs"]),
        "complete_export_stage_improves10percent": result["export_stage_reduction"] >= .1,
        "actual_worker_lifecycle_passed": lifecycle["status"] == "lifecycle_passed" and all(lifecycle["checks"].values()),
        "exact_adapter_source_for_lifecycle_pairs_and_current": lifecycle["sources"][1] == sources[str(adapter.resolve())] == data_identity(adapter),
        "all_pair_sources_unchanged": result["sources_unchanged"]}
    acceptance = {"status": "stage_guards_passed_combination_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "median": result["median"], "export_stage_reduction": result["export_stage_reduction"],
        "complete_raw_entry_reduction": 1-result["median"]["complete_entry_seconds"]["queue"]/result["median"]["complete_entry_seconds"]["baseline"],
        "counted_stage": bool(all(checks.values())), "counted_final_combination": False,
        "score_cache_already_warm_on_both_sides": True, "JPEG_artifact_cache_disabled_on_both_sides": True,
        "full_pipeline_cold_cost_or_new_quality_claim": False,
        "campaign": str(args.campaign), "lifecycle": str(args.validation),
        "JPEG_cache_combination_requires_main_thread_encoding_and_is_not_yet_accepted": True}
    atomic_json(args.output/"ACCEPTANCE.json", acceptance)
    print(json.dumps(acceptance), flush=True)
    if not all(checks.values()):
        raise ValueError("The full raw export queue did not satisfy its acceptance guards")


if __name__ == "__main__":
    main()
