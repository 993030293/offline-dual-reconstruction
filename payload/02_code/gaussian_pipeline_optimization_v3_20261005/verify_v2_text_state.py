"""Verify V2 report values, acceptance counts, links and current close receipts."""

import ast
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
CODE = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    state = read(DOC / "CURRENT_STATE.json")
    receipts = [read(ROOT / path) for path in state["acceptance_receipts"]]
    report = (DOC / "V2_REAUDIT_RESULTS_20261006.md").read_text(encoding="utf-8")
    checks = {"state_exact17_unique_mechanisms_and_receipts": state["additional_accepted_count"] == 17
        and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 17
        and len(receipts) == len(set(state["acceptance_receipts"])) == 17,
        "every_acceptance_is_actual_accepted_status": all(row["status"].startswith("accepted") for row in receipts),
        "current_policy_speed5percent_only_with_hard_protection_preserved": state["minimum_complete_stage_speed_reduction_current"] == .05
            and state["historical_speed_gate_v1"] == .1 and state["quality_geometry_output_recovery_and_whole_gates_unchanged"],
        "exact_two_new_V2_items": state["V2_newly_accepted_count"] == 2
            and all(row["new_optimization_count"] == 1 and row["acceptance_policy_version"] == 2 for row in receipts[-2:])}
    for name in ("am001", "dpv001", "parallel_a_004"):
        row = read(OUT / name / "ACCEPTANCE_V2.json")
        checks[name + "_all_required_guards_and_V2_cost_passed"] = all(row["checks"].values())
        for field in ("phase", "whole"):
            modes = tuple(row["complete_medians_seconds"][field])
            a, b = [row["complete_medians_seconds"][field][key] for key in modes]
            reduction = 1 - b / a
            checks[name + "_" + field + "_ratio_and_report_decimal_exact"] = (
                math.isclose(reduction, row["reductions"][field], rel_tol=0, abs_tol=1e-14)
                and f"{a:.8f}→{b:.8f}" in report)
    checks["old_R120_negative_zero_retained"] = read(OUT / "am001/NEGATIVE_CLOSURE_001.json")["new_optimization_count"] == 0
    checks["old_R127_negative_and_FINAL_COST_zero_retained"] = all(read(OUT / "dpv001" / name)["new_optimization_count"] == 0
        for name in ("NEGATIVE_CLOSURE_001.json", "FINAL_COST_001.json"))
    checks["old_A004_full_cost10percent_negative_and_new_combo_zero_retained"] = (
        read(DOC / "parallel_001/a_004/FULL_COST_REVISION_001.json")["gates"]["phase_median_at_least10percent"] is False
        and read(OUT / "parallel_a_004/ACCEPTANCE_V2.json")["new_optimization_count"] == 0)
    checks["R111_measured_cost_upper_bound_below5percent"] = (
        read(OUT / "acceptance_v2_reaudit_001/OTHER_CANDIDATE_SCREEN.json")["R111"]["speed_reduction_upper_bound"] < .05)
    paths = [DOC / name for name in ("V2_REAUDIT_RESULTS_20261006.md", "ENGINEERING_ACCEPTANCE_POLICY_V2_20261006.md",
        "V2_PRIOR_CANDIDATE_REAUDIT_TASK.md", "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")]
    targets = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        targets += re.findall(r"\]\((C:/[^)]+)\)", text)
        if path.name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md"):
            checks[path.name + "_current17_and_V2_present"] = "17/50" in text and "V2" in text
    checks["all_linked_local_paths_in_changed_main_documents_exist"] = all(Path(value).exists() for value in targets)
    leases = [*OUT.glob("am001/V2*LEASE*.json"), *OUT.glob("dpv001/V2*LEASE*.json"),
        *OUT.glob("parallel_a_004/V2*LEASE*.json")]
    checks["all7_new_actual_resource_leases_released"] = len(leases) == 7 and all(read(path).get("lease_released_utc") for path in leases)
    checks["shared_primary_lease_released"] = read(OUT / "coordination_001/LOCK_OWNER.json")["status"] == "released"
    helpers = ("reaudit_array_match_v2.py", "finalize_array_match_v2.py", "check_streamed_ply_recovery_v2.py",
        "check_streamed_ply_recovery_v2_002.py", "reaudit_streamed_ply_v2.py", "reaudit_streamed_ply_v2_002.py",
        "finalize_streamed_ply_v2.py", "reaudit_export_scene_combo_v2.py", "close_export_scene_combo_review_v2.py",
        "finalize_export_scene_combo_v2.py", "screen_prior_candidates_v2.py", "verify_v2_text_state.py")
    for name in helpers:
        ast.parse((CODE / name).read_text(encoding="utf-8"), filename=name)
    checks["new_review_helpers_syntax_parse"] = True
    checks["root_only_and_parent50_still_incomplete_and_R131_not_claimed"] = (
        state["execution_mode"] == "single_chat_no_subagents_or_side_chat_dispatch"
        and state["additional_accepted_count"] < state["additional_target"]
        and state["R131_new_optimization_count"] == 0 and "not_run" in state["next_candidate_status"])
    result = {"status": "text_state_cost_and_link_checks_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "check_count": len(checks), "local_link_count": len(targets),
        "additional_accepted_count": 17, "new_mechanism_count_from_text_check": 0,
        "review_kind": "root_text_and_structured_receipt_self_check_no_images_no_document_hashes",
        "not_a_new_algorithm_test_or_independent_review": True}
    (DOC / "V2_TEXT_STATE_VERIFICATION_001.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result, flush=True)
    if not all(checks.values()):
        raise AssertionError("A current count, claim, cost, source syntax or link differs")


if __name__ == "__main__":
    main()
