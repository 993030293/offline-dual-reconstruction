"""Close real producer births before applying the parent's single scoped acceptance increment."""

import datetime
import os
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, matches, read, save

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    acceptance = read(OWN / "ACCEPTANCE_V6_001.json")
    review = read(OWN / "SOURCE_COST_REVIEW_003.json")
    lease = read(OWN / "publisher_lease.json")
    process_rows = review["ended_historical_process_births"] + acceptance["verified_ended_review_births"] + [
        acceptance["producer_process"], {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    state = read(DOC / "CURRENT_STATE.json")
    text = (DOC / "R164_ORDERING_SCRATCH_REUSE_RESULTS.md").read_text(encoding="utf-8")
    brief = (DOC / "NEXT_SORT32_WASM_BRIDGE_COST_008_BRIEF.md").read_text(encoding="utf-8")
    checks = {
        "actual_publisher_tool035812_exit0_and_all_prior_process_births_closed": lease["returncode"] == 0 and bool(lease.get("lease_released_utc")) and all(closed(row) for row in process_rows),
        "all22_separate_quality_recovery_cost_checks_kept": len(review["checks"]) == 22 and all(review["checks"].values()),
        "registered_net_work10_raw_and_adjusted_whole_kept_speed_not_reclassified": acceptance["final_net_work_reduction_under_registered_six_side_model"] >= .1 and not acceptance["accepted_as_speed_or_memory_occupancy_or_FPS_optimization"],
        "normal_failed_and_published_sources_still_pinned": all(matches(row) for row in [*review["sources"], acceptance["publisher_source_identity"]] if not row["path"].lower().endswith(".txt")),
        "directly_read_report_failure_cost_generic_and_scope_boundaries_present": all(token in text for token in ["11.630630", "7564879300", "a09775", "2ee603", "902ff0", "a90914", "未验证", "本族", "ACTUAL_FINAL_POST_EXIT_001.json"]),
        "new008_cost_brief_not_implementation_or_benefit_and_no_agents": all(token in brief for token in ["没有008实现或执行", "single chat/no subagent", "完整速度净2%"]),
        "parent_original34_unique_mechanisms_receipts_before_actual_close": state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 34,
        "single_independent_family_delta1_not_old_R163_or_R162": acceptance["new_optimization_count"] == 1 and acceptance["maximum_this_family_count"] == 1 and acceptance["candidate"].startswith("R164_") and state["R163_new_optimization_count"] == 0,
        "parent_goal_active_production_training_unknown_GT_not_claimed": state["status"] == "active_first_measured_algorithm_pass" and not acceptance["unknown_object_full_GT_training_cross_hardware_and_production_deployment_verified"],
    }
    if not all(checks.values()):
        raise AssertionError(str(checks))
    save(OWN / "ACTUAL_FINAL_POST_EXIT_001.json", {
        "status": "all_required_review_publisher_and_previous_jobs_actually_closed_ready_for_one_parent_increment_after_post_writer_exit",
        "checks": checks, "actual_publisher_tool_chunk": "035812", "actual_publisher_tool_exit_code": 0,
        "actual_review_tool_chunk": "a90914", "actual_review_tool_exit_code": 0,
        "ended_process_birth_records": process_rows, "parent_count_before_handshake": 34, "required_parent_count_after_actual_post_writer_exit": 35,
        "approved_family_increment": 1, "parent_goal_status": "active", "final_net_request_work_reduction": acceptance["final_net_work_reduction_under_registered_six_side_model"],
        "all_array_work_cost_bytes_with_allowances": acceptance["all_array_validation_work_fees_and_registered_allowance_bytes"],
        "post_writer_process": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "post_writer_requires_external_exit0_verification_before_parent_increment": True,
        "last_text_and_file_metadata_execution_cost_is_covered_by_preregistered100_seconds_allowance": True,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "subagents_used": False,
    })
    print({"all9_final_close_checks": True, "parent_current": 34, "parent_ready_after_actual_post_writer_exit": 35, "goal": "active"})


if __name__ == "__main__":
    main()
