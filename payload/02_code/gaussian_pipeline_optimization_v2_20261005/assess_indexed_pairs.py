"""Accept exact indexed candidate graphs while disclosing their small absolute stage benefit."""

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
    adapter = Path(__file__).with_name("indexed_candidate_pairs.py")
    sources = {str(Path(r["path"]).resolve()): r for r in result["sources"]}
    checks = {"three_complete_actual_raw_pairs": len(result["pairs"]) == 3,
        "all7338_decisions1002_images11583_pairs_exact_each_pair": all(r["all_actual_outputs_exact"] and r["denominator_protected"] for r in result["pairs"]),
        "complete_time_no_more_than10percent_slower_each_pair": all(r["complete_time_protected"] for r in result["pairs"]),
        "full_tree_build_and_exact_pair_stage_improves10percent": result["pair_stage_reduction"] >= .1 and result["tree_build_included"],
        "all_lifecycle_guards": lifecycle["status"] == "lifecycle_passed" and all(lifecycle["checks"].values()),
        "current_lifecycle_and_actual_pair_source_exact": lifecycle["sources"][1] == sources[str(adapter.resolve())] == data_identity(adapter),
        "campaign_sources_unchanged": result["sources_unchanged"]}
    acceptance = {"status": "stage_guards_passed_combination_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "median": result["median"], "pair_stage_reduction": result["pair_stage_reduction"],
        "median_absolute_pair_seconds_saved": result["median_absolute_pair_seconds_saved"],
        "counted_stage": bool(all(checks.values())), "counted_final_combination": False,
        "whole_entry_time_improved": result["median"]["complete_entry_seconds"]["indexed"] < result["median"]["complete_entry_seconds"]["baseline"],
        "whole_route_speed_or_quality_improvement_claim": False,
        "deployment_priority": "low; only18ms saved at current1002 size; prioritize recovery and image export first",
        "campaign": str(args.campaign), "lifecycle": str(args.validation)}
    atomic_json(args.output/"ACCEPTANCE.json", acceptance)
    print(json.dumps(acceptance), flush=True)
    if not all(checks.values()):
        raise ValueError("The exact spatial pair-index acceptance did not pass")


if __name__ == "__main__":
    main()
