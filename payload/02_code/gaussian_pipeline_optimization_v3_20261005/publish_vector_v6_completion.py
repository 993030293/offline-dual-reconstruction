"""Publish the scoped SIMD decision after the real separate reviewer exited."""

import time

STARTED = time.perf_counter()

import argparse
from pathlib import Path
import shutil

import psutil

from review_vector_v6_completion import OWN, DOC, read, atomic_json, closed, cost, data_identity


def final_cost(root_envelope, publisher_seconds):
    value = cost(root_envelope + publisher_seconds)
    field = "actual_root_review_and_publication_so_far_seconds"
    value["components"]["conservative_observed_complete_root_envelope_and_publisher_seconds"] = value["components"].pop(field)
    value["root_envelope_is_upper_bound_through_observed_terminal_ack_not_CPU_time"] = True
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-tool-wall-seconds", type=float, required=True)
    parser.add_argument("--root-observed-envelope-upper-seconds", type=float, required=True)
    args = parser.parse_args()
    for name in ("ACCEPTANCE_V6_001.json", "FINAL_COST_V6_001.json"):
        if (OWN / name).exists():
            raise FileExistsError("An original publication receipt cannot be overwritten")
    review = read(OWN / "SELF_REVIEW_V6_001.json")
    parent = read(DOC / "CURRENT_STATE.json")
    if len(review["checks"]) != 15 or not all(review["checks"].values()) or not closed(review["reviewer"]):
        raise AssertionError("The successful separate root review must actually exit")
    if data_identity(Path(review["actual_executed_review_source"]["path"])) != review["actual_executed_review_source"]:
        raise AssertionError("The actual reviewed source changed")
    if parent["additional_accepted_count"] != 32 or any(name.startswith("R123") for name in parent["accepted_mechanisms"]):
        raise AssertionError("The unique family parent changed before publication")
    if args.root_observed_envelope_upper_seconds < args.root_tool_wall_seconds:
        raise AssertionError("The observed complete envelope cannot be shorter than actual tool waits")
    calculated = final_cost(args.root_observed_envelope_upper_seconds, time.perf_counter() - STARTED)
    if calculated["net_complete_stage_reduction_under_registered_six_side_model"] < .02:
        raise AssertionError("The uniform V6 net2percent criterion failed")
    source = data_identity(Path(__file__).resolve())
    shutil.copyfile(__file__, OWN / "EXECUTED_PUBLISHER_V6.py")
    publisher = {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}
    atomic_json(OWN / "FINAL_COST_V6_001.json", dict(calculated, status="decision_cost_actual_publisher_exit_pending",
        root_actual_tools={"initial_chunk": "666ace", "initial_session": 72768, "terminal_chunk": "d35fac", "exit_code": 0,
            "total_tool_wait_seconds": args.root_tool_wall_seconds}, root_observed_terminal_snapshot_rounded_up_UTC="2026-10-07 13:40:44 UTC",
        root_observed_envelope_upper_seconds=args.root_observed_envelope_upper_seconds, publisher=publisher, actual_publisher_source=source))
    atomic_json(OWN / "ACCEPTANCE_V6_001.json", {"status": "optional_strict_SIMD_complete_sampling_V6_accepted_actual_publication_exit_pending",
        "mechanism": "R123_STRICT_SIMD_COMPLETE_RGB_SAMPLING", "same_family_maximum_one": True, "new_optimization_count": 1,
        "parent_before": 32, "parent_after": 33, "target": 50, "actual_counter_publication_pending": True,
        "scope": "pinned_native_versions_and_admitted_AVX2_complete_RGB_sampling_preserving_original_interpolation_rounding",
        "original_V5_and_g0_failures_preserved": True, "original_observed_formal_and_component_evidence_reused_not_new_fresh_or_blind": True,
        "actual_both_partial_interrupts_real_JPEG_write_failure_and_new_directory_full1050_recovery_passed": True,
        "review": "SELF_REVIEW_V6_001.json", "root_review_is_independent_agent": False, "cost": "FINAL_COST_V6_001.json",
        "publisher": publisher, "training_steps": 0, "production_deployed": False,
        "unknown_object_GT_cross_hardware_or_training_quality_verified": False,
        "original_memory_increase_preserved_not_a_memory_optimization": True,
        "arbitrary_multiple_CPP_filesystem_fault_order_proven": False})
    print("V6 SIMD scoped decision written; actual publisher exit and parent33 verification remain", flush=True)


if __name__ == "__main__":
    main()
