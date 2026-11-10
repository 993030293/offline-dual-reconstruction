"""Promote one source-bound exact selection component after independent review."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

OUTPUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50/parallel_001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUTPUT)
    args.output.mkdir(parents=True, exist_ok=False)
    current = read(OUTPUT / "parallel_a_001/p2s/RESULT.json")
    external = read(OUTPUT / "parallel_a_001/x2/RESULT.json")
    generic = read(OUTPUT / "parallel_a_001/pa02_oracle_002/RESULT.json")
    recovery = read(OUTPUT / "parallel_a_001/recovery_002/RESULT.json")
    review = read(DOC / "monitor/REVIEW_A.json")
    closure_lease = read(OUTPUT / "parallel_a_001/closure_001_lease.json")
    judged = next(x for x in review["candidates"] if x["id"] == "PA02")
    sources = {}
    for result in (current, external, generic, recovery):
        for source in result["sources"]:
            path = Path(source["path"])
            sources[str(path)] = {"recorded": source, "current": data_identity(path)}
    fields = ("whole_seconds", "complete_selector_seconds", "complete_selector_plus_independent_pair_verification_seconds")
    median = {field: {mode: statistics.median(p[field][mode] for p in current["pairs"])
        for mode in ("native", "hash")} for field in fields}
    reductions = {field: 1 - values["hash"] / values["native"] for field, values in median.items()}
    pair_formula_exact = all(all(
        p["complete_selector_plus_independent_pair_verification_seconds"][mode]
        == p["complete_selector_seconds"][mode] + p["independent_pair_verification_seconds"] / 2
        for mode in ("native", "hash")) for p in current["pairs"])
    checks = {
        "three_actual_current_pairs_all_original_contracts_protected": len(current["pairs"]) == 3
            and all(all(p["checks"].values()) for p in current["pairs"]),
        "complete_stage_including_all_disclosed_pair_verification_reduces10percent": reductions[fields[2]] >= .1,
        "pair_cost_formula_and_medians_recomputed_exact": pair_formula_exact and median == current["median"],
        "all_whole_entry_costs_protected": all(p["whole_seconds"]["hash"] <= 1.1 * p["whole_seconds"]["native"]
            for p in current["pairs"]) and median["whole_seconds"]["hash"] <= median["whole_seconds"]["native"],
        "actual_formal_monitor_accepted_only_PA02": judged["decision"] == "accepted"
            and review["candidate_accepted"] == 1 and review["candidate_rejected"] == 3,
        "actual_parent_monitor_identity_and_independent_scope": review["reviewer_thread_id"] == "01a10c69-5871-7d80-8de1-8180d585d779"
            and review["reviewer_parent_thread_id"] == "01a079ba-7973-7780-bf60-a034f8b50aa3"
            and review["independent_of_A_implementation"] and not review["remaining_required_batch_work"],
        "all_generic_original_value_range_and_error_guards": all(generic["checks"].values())
            and judged["generic_behavior_case_count"] == 26,
        "source_bound_external_body_component_all_three_pairs": len(external["pairs"]) == 3
            and all(external["checks"].values()) and all(all(p["checks"].values()) for p in external["pairs"]),
        "original_binding_cache_error_and_final_lease_closure": all(recovery["checks"].values())
            and closure_lease["returncode"] == 0 and bool(closure_lease["lease_released_utc"]),
        "all_measured_current_source_identity_unchanged": all(x["recorded"] == x["current"] for x in sources.values()),
        "independent_actual_complete_export_verification_scope": judged["actual_protected_artifacts"] == {
            "raw_decisions": 7338, "eligible_frames": 6797, "exported_images": 1002, "matching_pairs": 11583,
            "excluded_evaluation_frames": 48, "complete_runs": 6, "all_JPEG_bytes_exact": True,
            "all_decisions_pairs_K_policy_sensor_identity_and_raw_odometry_arrays_exact": True},
    }
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_acceptance",
        "mechanism": "R107_EXACT_INCREMENTAL_SELECTION_NOVELTY_INDEX", "branch_candidate": "PA02",
        "additional_accepted_count": int(all(checks.values())), "checks": checks, "median": median,
        "reductions": reductions, "source_identity": list(sources.values()),
        "current_result": str(OUTPUT / "parallel_a_001/p2s/RESULT.json"),
        "external_result": str(OUTPUT / "parallel_a_001/x2/RESULT.json"),
        "independent_review": str(DOC / "monitor/REVIEW_A.json"),
        "independent_actual_artifact_verification": str(DOC / "monitor/A_EVIDENCE_001.json"),
        "whole_RAW_entry_speed_improvement_established": False,
        "whole_bounded_poll_resolution_seconds": .5,
        "scope": "exact_existing_warm_quality_selection_component_and_fixed_policy_development_body_pose_component",
        "generic_component_not_object_specific": True, "policy_frame_poses_K_or_JPEG_semantics_changed": False,
        "full_new_object_pipeline_or_geometry_accuracy_claim": False, "strict_blind_evaluation": False,
        "new_training_run_or_object_quality_improvement_claim": False,
        "production_integration_verified": False, "superseded_p2_not_used": True,
        "forced_guard_WSL_cancellation_verified": False, "assessor": data_identity(Path(__file__))}
    atomic_json(args.output / "ACCEPTANCE.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "count": result["additional_accepted_count"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A measured source-bound selection acceptance condition failed")


if __name__ == "__main__":
    main()
