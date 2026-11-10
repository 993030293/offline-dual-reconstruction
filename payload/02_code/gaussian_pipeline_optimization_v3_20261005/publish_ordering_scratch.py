"""Publish the scoped decision only after verifying the separate review's actual process exit."""

import datetime
import os
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, identity, matches, read, save

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    result = read(OWN / "SOURCE_COST_REVIEW_003.json")
    lease = read(OWN / "review3_lease.json")
    review_births = [result["review_process"], {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                    {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    if lease["returncode"] != 0 or not lease.get("lease_released_utc") or not all(closed(row) for row in review_births + result["ended_historical_process_births"]):
        raise AssertionError("The actual successful review or prior process exits are incomplete")
    if len(result["checks"]) != 22 or not all(result["checks"].values()) or not all(matches(row) for row in result["sources"] if not row["path"].lower().endswith(".txt")):
        raise AssertionError("Current original inputs or the separate evidence gates changed")
    for row in result["source_snapshots"]:
        if not matches(row):
            raise AssertionError("A recorded normal or failure source snapshot changed")
    text = (DOC / "R164_ORDERING_SCRATCH_REUSE_RESULTS.md").read_text(encoding="utf-8")
    for token in ["11.630630", "7564879300", "不是OS/RSS", "0.063168", "20组", "1504", "a90914", "root自审", "R163 whole负", "35/50", "未部署", "没有完整候选侧"]:
        if token not in text:
            raise AssertionError("The directly read report omits a required scope or evidence boundary: " + token)
    state = read(DOC / "CURRENT_STATE.json")
    if state["additional_accepted_count"] != 34 or len(state["accepted_mechanisms"]) != 34 or len(state["acceptance_receipts"]) != 34:
        raise AssertionError("The parent publication baseline changed")
    declaration = {"status": "scoped_work_acceptance_conditional_on_actual_final_exit_and_parent_receipt_consistency",
                   "candidate": "R164_FULL_ZERO_ORDERING_SCRATCH_REUSE_ORIGINAL_IMAGE_DATA_PROTECTED",
                   "decision": "accept_optional_complete_ordering_array_request_work_mechanism", "new_optimization_count": 1,
                   "maximum_this_family_count": 1, "parent_expected_additional_count_after_actual_close": 35, "parent_goal_status": "active",
                   "raw_new_ordering_requests_B_C_bytes": result["array_work_cost"]["raw_new_ordering_requests_B_C_bytes"],
                   "raw_work_reduction": result["array_work_cost"]["raw_request_work_reduction"],
                   "final_net_work_reduction_under_registered_six_side_model": result["array_work_cost"]["net_request_work_reduction"],
                   "all_array_validation_work_fees_and_registered_allowance_bytes": result["array_work_cost"]["all_known_and_registered_allowance_bytes"],
                   "scope": result["array_work_cost"]["scope"], "accepted_as_speed_or_memory_occupancy_or_FPS_optimization": False,
                   "speed_and_whole_cost": result["execution_time"], "separate_root_review": str(OWN / "SOURCE_COST_REVIEW_003.json"),
                   "report": str(DOC / "R164_ORDERING_SCRATCH_REUSE_RESULTS.md"), "task": str(DOC / "R164_ORDERING_SCRATCH_REUSE_TASK.md"),
                   "all22_review_checks": result["checks"], "actual_review_tool_chunk": "a90914", "actual_review_tool_exit_code": 0,
                   "verified_ended_review_births": review_births, "required_final_exit_receipt": str(OWN / "ACTUAL_FINAL_POST_EXIT_001.json"),
                   "producer_process": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
                   "publisher_source_identity": identity(Path(__file__)),
                   "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "unknown_object_full_GT_training_cross_hardware_and_production_deployment_verified": False,
                   "unmeasured_total_research_text_time_not_zero_and_no_total_ROI": True}
    save(OWN / "ACCEPTANCE_V6_001.json", declaration)
    print({"decision": declaration["decision"], "net_request_work": declaration["final_net_work_reduction_under_registered_six_side_model"],
           "parent_count_unchanged_until_actual_close": 34, "condition": declaration["required_final_exit_receipt"]})


if __name__ == "__main__":
    main()
