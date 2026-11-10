"""Publish scoped work acceptance only after the separate actual root review."""

import json
from pathlib import Path
import time

import psutil

import review_interior_conv_work as review
import run_interior_conv_campaign as campaign


def main():
    started = time.perf_counter()
    own = campaign.OWN
    record = campaign.read(own / "SELF_REVIEW_001.json")
    if record["status"] != "root_self_review_passed" or not all(record["checks"].values()):
        raise AssertionError("The separate original-output review did not pass")
    if not review.closed(record["reviewer"]):
        raise AssertionError("The separate reviewer has not actually exited")
    state = campaign.read(campaign.ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if state["additional_accepted_count"] != 30 or any("R134" in item for item in state["accepted_mechanisms"]):
        raise AssertionError("The scoped R134 mechanism cannot be counted twice")
    work = review.work_cost()
    costs = review.time_cost(extra_seconds=3.5754024 + time.perf_counter() - started, tail_reserve=10.)
    if work["net_reduction_lower_bound_under_registered_six_side_model"] < .1:
        raise AssertionError("The registered complete native work lower bound failed")
    if (costs["adjusted_whole_median_ratio"] > 1.01 or costs["raw_whole_median_ratio"] > 1.01
            or any(row > 1.1 for row in costs["raw_each_whole_ratios"])):
        raise AssertionError("An original complete whole protection failed")
    owner = psutil.Process()
    receipt = {"status": "accepted_optional_complete_SSIM_native_conv_output_work",
        "new_optimization_count": 1, "mechanism": "R134_COMPLETE_FULL_TARGET_INTERIOR_CONV_WORK_REUSE",
        "expected_parent_before": 30, "expected_parent_after": 31, "parent_target": 50,
        "parent_status": "active", "review_is_independent_agent": False,
        "work": work, "cost": costs, "root_review_actual_instance_exited": True,
        "publisher": {"pid": owner.pid, "creation_time": owner.create_time()},
        "raw_work_reduction": 1 - 305088480 / 518268096,
        "speed3percent_training_quality_GT_new_object_pipeline_production_accepted": False,
        "old_R134_speed_c0_and_first_fault_entry_failures_preserved": True,
        "new_private_full_three_pairs_public360_errors_and_fresh_full_recovery_verified": True,
        "post_exit_native_text_and_parent_counter_verification_pending": True}
    campaign.atomic_json(own / "ACCEPTANCE_WORK_001.json", receipt)
    campaign.atomic_json(own / "FINAL_COST_001.json", {"work": work, "cost": costs,
        "publisher": receipt["publisher"], "root_review_actual_tool_seconds": 3.5754024,
        "publisher_so_far_seconds": time.perf_counter() - started,
        "post_exit_and_text_tail_reserve_seconds": 10., "count": 1,
        "new_optimization_count": 1, "expected_parent_count": 31, "parent_goal_status": "active"})
    print(json.dumps({"status": receipt["status"], "net_work_lower_bound": work[
        "net_reduction_lower_bound_under_registered_six_side_model"], "expected_parent_count": 31}), flush=True)


if __name__ == "__main__":
    main()

