"""Record actual publisher exits and keep the original final tail reserve once."""

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

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-command-seconds", type=float, required=True)
    parser.add_argument("--post-command-chunk", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    final_path = BASE / "FINAL_COST_001.json"
    if final_path.exists():
        raise FileExistsError("Never overwrite the final execution cost receipt")
    post = read(BASE / "FINAL_POST_EXIT_001.json")
    publication = read(BASE / "PUBLICATION_001.json")
    accepted = read(BASE / "ACCEPTANCE_V5_001.json")
    formal = read(BASE / "c0/RESULT.json")
    state_path = DOC / "CURRENT_STATE.json"
    state = read(state_path)
    checks = {
        "actual_preparation_and_post_instances_exited": windows_closed(publication["own_process"]) and windows_closed(post["own_process"]),
        "actual_post_exit_window_covers_prior_internal_window": args.post_command_seconds >= post["internal_charged_seconds_replaced_by_actual_outer_exit_later"],
        "post_all8_and_acceptance_checks_passed": len(post["checks"]) == 8 and all(post["checks"].values()) and all(accepted["checks"].values()),
        "unique29_parent_receipts_and_only_one_R153": state["additional_accepted_count"] == 29
            and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 29
            and sum(name.startswith("R153") for name in state["accepted_mechanisms"]) == 1
            and state["R153_new_optimization_count"] == accepted["new_optimization_count"] == 1,
        "executed_publication_sources_and_saved_copies_still_exact": all(
            row["source"] == data_identity(row["source"]["path"])
            and data_identity(BASE / "FINALIZER_SOURCES" / Path(row["source"]["path"]).name)["sha256"] == row["source"]["sha256"]
            for row in (post, publication)),
        "registered_10second_reserve_included_once": accepted["registered_reserve_seconds"] == 10 and accepted["reserve_already_included_not_added_twice"],
    }
    if not all(checks.values()):
        atomic_json(BASE / "FAILED_FINAL_EXIT_001.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("The actual final publisher exit evidence failed")
    native, fused = (formal["raw_complete_stage_medians"][key] for key in ("original", "fused"))
    known_before_writer = post["known_fee_plus_registered10_reserve_before_post_window_seconds"] + args.post_command_seconds
    writer_internal = time.perf_counter() - started
    fee = known_before_writer + writer_internal
    bound = equal_fee_cost_bound(native, fused, fee)
    if bound["signed_reduction_at_known_lower_fee"] < .03:
        raise AssertionError("The final cost failed the unchanged uniform 3 percent gate")
    shutil.copyfile(Path(__file__), BASE / "FINALIZER_SOURCES" / Path(__file__).name)
    process = psutil.Process()
    atomic_json(final_path, {
        "status": "accepted_one_scoped_R153_after_actual_publisher_post_exits_parent29_active",
        "checks": checks, "new_optimization_count": 1, "parent_count": 29, "remaining": 21,
        "actual_closed_preparation_instance": publication["own_process"],
        "actual_closed_post_instance": post["own_process"],
        "actual_root_review_command": publication["actual_root_review_command"],
        "actual_preparation_command": post["actual_publisher_command"],
        "actual_post_command": {"returncode": 0, "seconds": args.post_command_seconds, "tool_chunk": args.post_command_chunk},
        "known_fee_plus_registered10_reserve_before_final_writer_seconds": known_before_writer,
        "final_writer_internal_charged_seconds": writer_internal,
        "known_fee_plus_registered10_reserve_seconds": fee,
        "recorded_execution_fee_model": bound,
        "complete_stage_net_reduction_under_registered_execution_cost_model": bound["signed_reduction_at_known_lower_fee"],
        "raw_complete_stage_reduction": formal["raw_stage_reduction"],
        "registered_reserve_seconds": 10, "reserve_already_included_not_added_twice": True,
        "final_writer_tail_and_last_text_bookkeeping_protected_by_registered_reserve": True,
        "unknown_historical_text_and_research_not_assumed_zero_or_total_ROI": True,
        "complete_entry_scope_not_entire_SfM_training_pipeline": True,
        "post_internal_window_replaced_not_double_charged": True,
        "final_writer_instance_for_actual_tool_exit_observation": {"pid": process.pid, "creation_time": process.create_time()},
        "source": data_identity(Path(__file__)), "goal_remains_active": True,
        "production_default_changed": False, "generalization_pending": True})
    state.update(R153_status="accepted_optional_exact_complete_camera_verification_V5_actual_publisher_post_exits_closed1",
        R153_final_cost=final_path.relative_to(ROOT).as_posix(),
        R153_final_post_exit="04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/FINAL_POST_EXIT_001.json",
        R153_complete_stage_net_reduction_under_registered_execution_cost_model=bound["signed_reduction_at_known_lower_fee"],
        R153_final_known_fee_plus_registered10_reserve_seconds=fee,
        R153_actual_publisher_and_post_instances_exited=True,
        current_state_updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    print({"final_net_under_registered_execution_cost_model": bound["signed_reduction_at_known_lower_fee"],
        "known_fee_plus_reserve": fee, "new": 1, "parent": 29, "remaining": 21})


if __name__ == "__main__":
    main()
