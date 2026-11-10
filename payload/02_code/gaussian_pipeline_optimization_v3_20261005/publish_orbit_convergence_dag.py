"""Publish a scoped decision after the separate root reviewer actually exits."""

import os
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import DOC, OWN, closed, cost, identity, matches, read, save


def main():
    started = time.perf_counter()
    review = read(OWN / "SELF_REVIEW_V6_002.json")
    guard = read(OWN / "review4_GUARD.json")
    lease = read(OWN / "review4_lease.json")
    if review["status"] != "separate_root_self_review_passed_pending_actual_review_exit_and_publication0" or not all(review["checks"].values()):
        raise AssertionError("The actual complete separate self-review did not pass")
    if guard["returncode"] != 0 or lease["returncode"] != 0 or not lease.get("lease_released_utc"):
        raise AssertionError("The successful reviewer did not record its actual exit and lease release")
    rows = guard["owned_process_identities"] + [review["reviewer"],
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    if not all(closed(row) for row in rows) or not matches(review["reviewer"]["source"]):
        raise AssertionError("A reviewer instance or its source changed")
    state = read(DOC / "CURRENT_STATE.json")
    if state["additional_accepted_count"] != 33 or len(state["accepted_mechanisms"]) != 33 or any(row.startswith("R162") for row in state["accepted_mechanisms"]):
        raise AssertionError("The parent decision was already published or changed")
    publisher = {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "source": identity(Path(__file__))}
    fees = cost(lease["child_elapsed_seconds"] + time.perf_counter() - started)
    if fees["net_complete_stage_reduction_under_registered_six_side_model"] < .02 or fees["adjusted_whole_median_ratio"] > 1.01 or fees["raw_whole_median_ratio"] > 1.01 or max(fees["each_original_whole_ratios"] + fees["each_adjusted_whole_ratios"]) > 1.1:
        raise AssertionError("The original complete speed or whole protection failed after final costs")
    save(OWN / "FINAL_COST_V6_001.json", dict(fees, status="actual_publisher_exit_and_final_tail_pending", publisher=publisher))
    save(OWN / "ACCEPTANCE_V6_001.json", {
        "status": "accepted_optional_R162_actual_publisher_exit_and_counter_pending",
        "mechanism": "R162_EXACT_DISCRETE_ORBIT_CPU_CONVERGENCE_DAG", "new_optimization_count": 1,
        "same_family_maximum_one": True, "parent_before": 33, "parent_after": 34, "target": 50,
        "review": "SELF_REVIEW_V6_002.json", "cost": "FINAL_COST_V6_001.json", "publisher": publisher,
        "uniform_speed_policy": "V6_NET_SPEED_POLICY.md", "quality_or_input_or_recovery_threshold_reduced": False,
        "scope": "original_static_preview_discrete_camera_CPU_fixed_point_before_view_dependent_GPU_generation",
        "three_fresh_complete_AB_BA_AB_pairs": True, "all_fixed_initial_plus21_pixels_and_full_camera_fields_exact": True,
        "all_original1016_public360_generic128_component_populations_preserved": True,
        "converged1318_cycle41_and_bound145_original_routes_explicit": True,
        "all1504_inputs_unchanged_before_marker_and_first_actual_update": True,
        "actual_original_errors_real_page_close_cancel_and_fresh_full21_recovery": True,
        "root_review_is_independent_agent": False, "self_review_first3_failures_and_all_fees_preserved": True,
        "production_default_or_current_service_deployed": False, "new_training_steps": 0,
        "full_reconstruction_speed_FPS_GT_unknown_object_quality_cross_hardware_verified": False,
    })
    print("Scoped R162 decision saved; actual publisher exit, final costs and parent counter publication remain", flush=True)


if __name__ == "__main__":
    main()
