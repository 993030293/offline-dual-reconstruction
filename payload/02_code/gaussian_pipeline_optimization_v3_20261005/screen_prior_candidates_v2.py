"""Keep a cost lower-bound rejection separate from unfinished V2 reviews."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    base = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
    output = base / "acceptance_v2_reaudit_001"
    output.mkdir(exist_ok=False)
    result = read(base / "sparse_ply_oracle_001/campaign_002/RESULT.json")
    prefixes = ("sparse_ply_oracle_001_generic", "sparse_ply_oracle_002_generic",
        "sparse_ply_rejection_001", "sparse_ply_profile_001", "sparse_export_compare_001")
    rows = [{"path": str(base / (name + "_process.json")),
        "already_measured_guard_seconds": read(base / (name + "_process.json"))["elapsed_seconds"]}
        for name in prefixes]
    fee = sum(row["already_measured_guard_seconds"] for row in rows)
    native = result["medians"]["native"]["phase_seconds"] + fee / 6
    columns = result["medians"]["columns"]["phase_seconds"] + fee / 6
    upper_bound = 1 - columns / native
    if upper_bound >= .05:
        raise AssertionError("A lower-bound fee is insufficient to reject this candidate")
    receipt = {"status": "V2_cost_lower_bound_rejection_and_pending_review_ledger",
        "acceptance_policy_version": 2, "new_optimization_count": 0,
        "R111": {"status": "rejected_by_required_first_cost_lower_bound", "minimum_complete_stage_reduction": .05,
            "original_phase_reduction": result["phase_reduction"], "previously_unallocated_required_first_fees_lower_bound_seconds": fee,
            "phase_seconds_after_only_this_lower_bound": {"native": native, "columns": columns},
            "speed_reduction_upper_bound": upper_bound, "fee_rows": rows,
            "further_review_or_external_fees_excluded_from_this_bound": True,
            "remaining_formal_functional_and_external_checks_not_run": True,
            "this_is_not_full_semantic_or_current_source_process_revalidation": True},
        "R89_R101_B007": "V2_full_fees_and_remaining_protections_not_reaudited_not_accepted",
        "B007_interrupted_second_campaign": "preserved_not_completed_no_other_chat_restart",
        "R113_R122": "original_whole_median_regression_gate_unchanged_not_accepted",
        "R119_R129_B006": "original_quality_or_precision_failure_protection_unchanged_not_accepted",
        "PB02": "historical4_843806percent_below_new5percent_not_repackaged_as_new_mechanism",
        "parent50_goal_complete": False, "production_modified": False, "review_kind": "root_self_review_no_subagents"}
    atomic_json(output / "OTHER_CANDIDATE_SCREEN.json", receipt)
    print({"R111_first_fee_lower_bound_seconds": fee, "R111_V2_speed_upper_bound": upper_bound,
        "new_optimization_count": 0}, flush=True)


if __name__ == "__main__":
    main()
