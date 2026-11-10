"""Record actual composition publication exit with the original fee reserve once."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher-command-seconds", type=float, required=True)
    parser.add_argument("--publisher-command-chunk", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = OUT / "FINAL_COST_001.json"
    if output.exists():
        raise FileExistsError("Keep the original terminal composition cost receipt")
    publication = read(OUT / "PUBLICATION_001.json")
    accepted = read(OUT / "ACCEPTANCE_V5_001.json")
    formal = read(OUT / "c0/RESULT.json")
    state_path = DOC / "CURRENT_STATE.json"
    state = read(state_path)
    checks = {
        "actual_publisher_instance_exited_and_window_covers_internal": windows_closed(publication["own_process"]) and args.publisher_command_seconds >= publication["internal_before_result_seconds"],
        "unique29_parent_unchanged_and_combination_count0": state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 29 and accepted["new_optimization_count"] == state["Q68_R153_new_optimization_count"] == 0,
        "actual_executed_publisher_source_and_copy_unchanged": publication["source"] == data_identity(publication["source"]["path"]) and data_identity(OUT / "FINALIZER_SOURCES" / Path(publication["source"]["path"]).name)["sha256"] == publication["source"]["sha256"],
        "original_registered10_reserve_once_and_no_new_library_build": accepted["registered_reserve_seconds"] == 10 and formal["no_new_library_build_existing_parents_not_recounted"],
    }
    if not all(checks.values()):
        raise AssertionError("The actual publication did not finish with protected counts")
    native, combo = (formal["raw_complete_stage_medians"][key] for key in ("control", "combo"))
    before_writer = publication["known_fee_plus_registered10_reserve_before_publisher_window_seconds"] + args.publisher_command_seconds
    internal = time.perf_counter() - started
    fee = before_writer + internal
    bound = equal_fee_cost_bound(native, combo, fee)
    if bound["signed_reduction_at_known_lower_fee"] < .03:
        raise AssertionError("The final unchanged engineering net-cost gate failed")
    shutil.copyfile(Path(__file__), OUT / "FINALIZER_SOURCES" / Path(__file__).name)
    process = psutil.Process()
    atomic_json(output, {"status": "accepted_existing_Q68_R153_composition_actual_review_and_publisher_closed_parent29_active_new0",
        "checks": checks, "actual_closed_publisher_instance": publication["own_process"],
        "actual_publisher_command": {"returncode": 0, "seconds": args.publisher_command_seconds, "tool_chunk": args.publisher_command_chunk},
        "actual_review_command": publication["actual_review_command"], "known_fee_plus_registered10_reserve_seconds": fee,
        "final_writer_internal_charged_seconds": internal, "registered10_reserve_included_once": True,
        "recorded_incremental_execution_fee_model": bound,
        "complete_stage_net_reduction_under_registered_incremental_execution_cost_model": bound["signed_reduction_at_known_lower_fee"],
        "final_writer_tail_and_last_text_bookkeeping_protected_by_original_reserve": True,
        "unknown_text_research_not_assumed_zero_or_total_ROI": True,
        "source": data_identity(Path(__file__)), "own_process_for_actual_tool_exit_observation": {"pid": process.pid, "creation_time": process.create_time()},
        "new_optimization_count": 0, "parent_count": 29, "remaining": 21, "goal_remains_active": True,
        "production_default_changed": False, "generalization_pending": True})
    state.update(Q68_R153_status="accepted_optional_existing_composition_actual_review_and_publisher_exits_verified_new0",
        Q68_R153_final_cost=output.relative_to(ROOT).as_posix(),
        Q68_R153_complete_stage_net_reduction_under_registered_incremental_execution_cost_model=bound["signed_reduction_at_known_lower_fee"],
        Q68_R153_final_known_fee_plus_registered10_reserve_seconds=fee,
        Q68_R153_actual_review_and_publisher_instances_exited=True,
        current_state_updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    print({"final_incremental_net": bound["signed_reduction_at_known_lower_fee"], "fee_plus_reserve": fee, "new": 0, "parent": 29, "remaining": 21})


if __name__ == "__main__":
    main()
