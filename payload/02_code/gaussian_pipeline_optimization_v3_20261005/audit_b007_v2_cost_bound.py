"""Reject an old candidate using a conservative already recorded cost lower bound."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V3OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from gpu_query_node import remaining


def main():
    old = V3OUT / "parallel_b_007"
    paths = [old / "campaign_001/RESULT.json", old / "campaign_002_guard.json", old / "campaign_002_lease.json"]
    identities = [data_identity(path) for path in paths]
    first, second, lease = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    live = remaining(second["owned_process_identities"] + [
        {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
        for prefix in ("guard", "child")])
    a = first["medians"]["native"]["phase_seconds"]
    b = first["medians"]["bin"]["phase_seconds"]
    lower = second["elapsed_seconds"]
    extra = lower / 6
    bound = 1 - (b + extra) / (a + extra)
    checks = {"all_first_three_pairs_exist": len(first["pairs"]) == 3,
        "first_original_complete_science_and_whole_guards": all(value for key, value in first["checks"].items() if key != "phase_reduction_at_least10pct"),
        "later_recorded_cost_not_in_first_result": lower > first["prior_oracle_guard_seconds_charged_total"],
        "actual_interrupted_owned_process_instances_gone": not live,
        "best_case_after_one_extra_lower_bound_below5percent": bound < .05,
        "source_receipts_unchanged": identities == [data_identity(path) for path in paths]}
    if not all(checks.values()):
        raise AssertionError("A conservative old cost-screen premise failed")
    output = V3OUT / "old_b007_v2_cost_bound_001"
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "RESULT.json", {"status": "V2_cost_upper_bound_rejected_without_semantic_or_external_rerun",
        "checks": checks, "sources": [data_identity(Path(__file__))], "receipt_inputs": identities,
        "first_existing_complete_phase_medians_seconds": {"native": a, "bin": b},
        "one_later_interrupted_guard_elapsed_lower_bound_seconds": lower,
        "same_original_six_side_allocation_lower_bound_seconds": extra,
        "maximum_possible_complete_phase_reduction_after_this_lower_bound": bound,
        "additional_unaccounted_fees_can_only_reduce_this_bound": True,
        "guard_running_label_is_stale_not_a_live_process_or_final_total": True,
        "actual_saved_Windows_instances_checked": len(second["owned_process_identities"]) + 2,
        "actual_live_instances": live, "new_optimization_count": 0,
        "current_semantics_generic_external_full_output_recovery_unique_independent_review": "not_rerun_not_claimed",
        "old_source_or_receipt_modified": False})
    print(json.dumps({"status": "V2_cost_upper_bound_rejected", "bound": bound, "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
