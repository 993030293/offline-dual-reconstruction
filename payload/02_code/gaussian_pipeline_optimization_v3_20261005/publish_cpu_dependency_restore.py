"""Publish a scoped V6 decision after the real root review has exited."""

import time

STARTED = time.perf_counter()

import argparse
from pathlib import Path
import shutil

import psutil

from review_cpu_dependency_restore import OWN, DOC, calculate_cost, closed, read, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-tool-seconds", type=float, required=True)
    parser.add_argument("--root-tool-chunk", required=True)
    args = parser.parse_args()
    for name in ("ACCEPTANCE_V6_001.json", "FINAL_COST_V6_001.json"):
        if (OWN / name).exists():
            raise FileExistsError("A unique publication receipt is required")
    review = read(OWN / "SELF_REVIEW_V6_001.json")
    parent = read(DOC / "CURRENT_STATE.json")
    if not all(review["checks"].values()) or not closed(review["reviewer"]):
        raise AssertionError("The successful actual root reviewer must be closed")
    if data_identity(Path(review["actual_executed_review_source"]["path"])) != review["actual_executed_review_source"]:
        raise AssertionError("The reviewed implementation source changed")
    if parent["additional_accepted_count"] != 31 or any(value.startswith("R142") for value in parent["accepted_mechanisms"]):
        raise AssertionError("The prepublication unique family count changed")
    source = data_identity(Path(__file__).resolve())
    shutil.copyfile(__file__, OWN / "EXECUTED_V6_PUBLISHER_SOURCE.py")
    cost = calculate_cost(args.root_tool_seconds + time.perf_counter() - STARTED)
    if cost["net_complete_entry_reduction_under_registered_six_side_model"] < .02:
        raise AssertionError("The declared uniform V6 net2percent criterion failed")
    if cost["adjusted_whole_median_ratio"] > 1.01 or any(value > 1.1 for value in cost["each_original_whole_ratios"]):
        raise AssertionError("An unchanged whole entry protection failed")
    publisher = {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}
    atomic_json(OWN / "FINAL_COST_V6_001.json", dict(cost, status="decision_cost_actual_publisher_exit_pending",
        actual_root_tool={"chunk_id": args.root_tool_chunk, "exit_code": 0, "wall_time_seconds": args.root_tool_seconds},
        publisher=publisher, actual_publisher_source=source))
    atomic_json(OWN / "ACCEPTANCE_V6_001.json", {
        "status": "optional_scoped_V6_native_CPU_dependency_restore_DAG_accepted_actual_publication_exit_pending",
        "mechanism": "R142_NATIVE_CPU_DEPENDENCY_RESTORE_DAG", "new_optimization_count": 1,
        "same_family_maximum_one": True, "parent_before": 31, "parent_after": 32, "additional_target": 50,
        "actual_counter_publication_pending": True, "uniform_speed_policy": "V6_NET_SPEED_POLICY.md",
        "original_V5_net3_failure_preserved": True, "old_R142_GPU_startup_failures_preserved": True,
        "review": "SELF_REVIEW_V6_001.json", "root_review_is_independent_agent": False,
        "cost": "FINAL_COST_V6_001.json", "publisher": publisher,
        "scope": "fresh_pinned_Torch_CPU_dependency_import_overlapping_immutable_complete_input_identity_before_CUDA",
        "development_evidence_reused_unchanged_not_fresh_or_blind_V6_formal_replay": True,
        "actual_three_pairs_complete1016_original_checkpoint1000_restore_no_training": True,
        "actual_original_nine_state_checks_next32_and_all_pixels_matrices_calibration_order_exact": True,
        "actual_generic_checksum_SIGINT_and_new_directory_complete_recovery_passed": True,
        "training_steps": 0, "production_deployed": False, "whole_reconstruction_speed_or_quality_claim": False,
        "unknown_object_GT_cross_hardware_generalization_verified": False,
    })
    print("V6 scoped decision written; actual publisher exit and ledger verification remain", flush=True)


if __name__ == "__main__":
    main()
