"""Accept one scoped work mechanism after the actual publisher has exited."""

from datetime import datetime, timezone
from pathlib import Path
import time

import psutil

from run_open_image_handoff_campaign import OUT
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    prepared = read(OUT / "ACCEPTANCE_PREPARED_002.json")
    failed = read(OUT / "ACCEPTANCE_PREPARED_001.json")
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if not all(prepared["checks"].values()) or not windows_closed(prepared["own_process"]) or not windows_closed(failed["own_process"]):
        raise ValueError("The actual passing and failed publisher instances must have exited")
    if prepared["actual_source"] != data_identity(prepared["actual_source"]["path"]) or prepared["actual_source"]["sha256"] != prepared["source_copy"]["sha256"]:
        raise ValueError("The executed passing publisher source or snapshot changed")
    if parent["additional_accepted_count"] != 29 or parent["R156_new_optimization_count"] != 0:
        raise ValueError("One previously unpromoted family must be counted exactly once")
    acceptance = dict(prepared)
    acceptance.update(status="accepted_optional_scoped_source_file_work_actual_publisher_exited_speed_gate_not_passed",
        new_optimization_count=1, parent_count_before=29, parent_count_after=30,
        accepted_mechanism="R156_LOADED_NATIVE_IMAGE_HANDOFF_SOURCE_READ_WORK",
        actual_passing_publisher_tool_exit={"chunk_id": "85a4af", "returncode": 0, "seconds": .2917399},
        failed_first_publisher_preserved={"path": str(OUT / "ACCEPTANCE_PREPARED_001.json"), "actual_tool_returncode": 1, "seconds": .268376},
        accepted_at_utc=datetime.now(timezone.utc).isoformat())
    acceptance["known_fee_plus_registered10_reserve_through_publisher_exit_seconds"] = prepared["known_fee_plus_registered10_reserve_before_publisher_exit_seconds"] + .2917399
    acceptance["acceptance_finalizer_own_process"] = {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}
    acceptance["finalizer_source"] = data_identity(Path(__file__))
    acceptance["before_result_seconds"] = time.perf_counter() - started
    path = OUT / "ACCEPTANCE_WORK_001.json"
    if path.exists():
        raise FileExistsError("The scoped family acceptance must not be overwritten")
    atomic_json(path, acceptance)
    atomic_json(OUT / "FINAL_COST_001.json", {
        "status": "all_prior_actual_tool_exits_accounted_finalizer_and_counter_post_exit_pending",
        "known_fee_plus_registered10_reserve_seconds": acceptance["known_fee_plus_registered10_reserve_through_publisher_exit_seconds"],
        "raw_whole_medians": acceptance["raw_whole_medians"],
        "source_native_read_work_control": 4064, "source_native_read_work_handoff": 3048,
        "raw_source_read_work_reduction": .25,
        "measured_additional_work": read(OUT / "r1/RESULT.json")["measured_extra_source_read_work"],
        "inherited_source_work_budget_not_historical_direct_counter": 16256,
        "net_source_work_reduction_under_six_side_model": acceptance["net_source_read_work_under_registered_six_side_model"],
        "V5_speed_gate_passed": False, "finalizer_process": acceptance["acceptance_finalizer_own_process"],
        "goal_remains_active": True, "new_optimization_count": 1,
        "unknown_total_development_time_and_work_not_assumed_zero": True})
    print({"accepted_scoped_work": 1, "parent_before": 29, "parent_after_pending_state_publication": 30,
           "publisher_actual_exit_verified": True, "speed_gate": False})


if __name__ == "__main__":
    main()
