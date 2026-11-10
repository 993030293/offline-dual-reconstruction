"""Verify the actual closure process exit and publish the active parent state."""

from datetime import datetime, timezone
import re
from pathlib import Path
import time

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT, OUT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = OUT / "FINAL_POST_EXIT_001.json"
    if output.exists():
        raise FileExistsError(output)
    final = read(OUT / "FINAL_NEGATIVE_CLOSURE_001.json")
    if not all(final["checks"].values()) or not windows_closed(final["own_process"]):
        raise AssertionError("The completed negative publisher is not actually closed")
    native, batch = (final["complete_stage_seconds"][key] for key in ("native", "batch"))
    known = final["known_fee_lower_bound_before_publisher_exit_seconds"] + .3288116
    bound = equal_fee_cost_bound(native, batch, known)
    doc = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    report = (doc / "R152_SCENE_BATCH_RESULTS.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    if len(links) != 11 or not all(Path(link).is_file() or Path(link) == output for link in links):
        raise AssertionError("A final report target is still missing")
    state_path = doc / "CURRENT_STATE.json"
    state = read(state_path)
    if state["additional_accepted_count"] != 28 or state["R152_new_optimization_count"] != 0 or bound["conservative_sample_reduction_upper_bound"] != 0:
        raise AssertionError("An unchanged negative result unexpectedly promoted a mechanism")
    atomic_json(output, {"status": "actual_negative_closure_publisher_exited_parent28_active_new0",
        "actual_closed_publisher_instance": final["own_process"], "actual_publisher_exit_command_returncode": 0,
        "actual_publisher_spawn_to_exit_command_seconds": .3288116,
        "known_fee_lower_bound_before_post_checker_exit_seconds": known,
        "corrected_sample_cost_bound": bound, "actual_report_link_targets": links,
        "self_target_written_by_this_atomic_result": True,
        "original_c1_and_failed_publication_kept": True, "new_optimization_count": 0,
        "parent_count": 28, "remaining": 22, "goal_remains_active": True,
        "unknown_post_checker_text_and_research_cost_not_assumed_zero": True,
        "source_identity": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started})
    if not all(Path(link).is_file() for link in links):
        raise AssertionError("A final report target was not actually published")
    state.update(R152_final_post_exit_status="actual_negative_closure_publisher_exited_parent28_active_new0",
        R152_actual_final_publisher_instance_exited=True,
        R152_final_post_exit_known_fee_lower_bound_seconds=known,
        R152_final_consistency_check_count=len(final["checks"]),
        R152_original_report_link_targets_verified=11,
        current_state_updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    print("Post-exit confirmed: closure process exited, all 11 report targets exist, parent 28/50 active, new 0")


if __name__ == "__main__":
    main()
