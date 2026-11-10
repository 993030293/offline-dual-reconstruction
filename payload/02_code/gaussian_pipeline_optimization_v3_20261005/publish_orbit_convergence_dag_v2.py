"""Publish one scoped V2 mechanism after the complete separate reviewer actually exits."""

import os
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import DOC, OWN, closed, identity, matches, read, save
from review_orbit_convergence_dag_v2 import cost, fee_passes


def main():
    started = time.perf_counter()
    review = read(OWN / "SELF_REVIEW_V2_V6_001.json")
    guard, lease = read(OWN / "review_v2_GUARD.json"), read(OWN / "review_v2_lease.json")
    if review["status"] != "V2_separate_root_self_review_passed_pending_actual_exit_and_publication0" or not all(review["checks"].values()):
        raise AssertionError("The separate V2 reviewer did not pass all registered gates")
    rows = guard["owned_process_identities"] + [review["reviewer"],
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    if guard["returncode"] != 0 or lease["returncode"] != 0 or not lease.get("lease_released_utc") or not all(closed(row) for row in rows) or not matches(review["reviewer"]["source"]):
        raise AssertionError("The actual reviewer or its original source has not closed unchanged")
    state = read(DOC / "CURRENT_STATE.json")
    if state["additional_accepted_count"] != len(state["accepted_mechanisms"]) or state["additional_accepted_count"] != 33 or any(row.startswith("R162") for row in state["accepted_mechanisms"]):
        raise AssertionError("The parent33 changed or the same family was already counted")
    publisher = {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "source": identity(Path(__file__))}
    fees = cost(lease["child_elapsed_seconds"] + time.perf_counter() - started)
    if not fee_passes(fees):
        raise AssertionError("The complete net2percent or whole gate failed with actual additional fees")
    save(OWN / "FINAL_COST_V2_V6_001.json", dict(fees, status="V2_actual_publisher_exit_and_final_tail_pending", publisher=publisher))
    save(OWN / "ACCEPTANCE_V2_V6_001.json", {
        "status": "accepted_optional_R162_V2_actual_publisher_exit_and_counter_pending",
        "mechanism": "R162_V2_EXACT_DISCRETE_ORBIT_CPU_CONVERGENCE_DAG", "new_optimization_count": 1,
        "same_family_maximum_one": True, "parent_before": 33, "parent_after": 34, "target": 50,
        "review": "SELF_REVIEW_V2_V6_001.json", "cost": "FINAL_COST_V2_V6_001.json", "publisher": publisher,
        "actual_reviewer_exit_verified": True, "uniform_speed_policy": "V6_NET_SPEED_POLICY.md",
        "quality_input_error_cancel_recovery_threshold_reduced": False,
        "scope": "static_preview_discrete_idle_controller_CPU_fixed_point_before_view_dependent_GPU_work",
        "three_NEW_fresh_V2_AB_BA_AB_complete_pairs": True,
        "initial_plus21_fixed_RGBA_and_full_camera_populations_exact": True,
        "all1504_original_public_generic_and_all11_unsupported_kept": True,
        "five_component_and_four_real_DOM_navigation_intent_cases_passed": True,
        "actual_original_errors_page_close_cancel_fresh_complete21_recovery": True,
        "V1_provisional_publication_retracted_and_ALL_V1_V2_failures_fees_retained": True,
        "root_review_is_independent_agent": False, "production_default_or_current_service_deployed": False,
        "new_training_steps": 0, "full_reconstruction_FPS_GT_unknown_object_quality_cross_hardware_verified": False,
    })
    print("V2 scoped decision saved; actual publisher exit, final tail and parent publication remain", flush=True)


if __name__ == "__main__":
    main()
