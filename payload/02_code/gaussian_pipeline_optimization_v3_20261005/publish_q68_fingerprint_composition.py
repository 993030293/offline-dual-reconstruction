"""Publish the reviewed existing-mechanism composition without increasing count."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
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
    parser.add_argument("--review-command-seconds", type=float, required=True)
    parser.add_argument("--review-command-chunk", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    accepted_path = OUT / "ACCEPTANCE_V5_001.json"
    if accepted_path.exists() or (OUT / "PUBLICATION_001.json").exists():
        raise FileExistsError("Never overwrite an existing composition acceptance")
    review = read(OUT / "r0/RESULT.json")
    formal = read(OUT / "c0/RESULT.json")
    state_path = DOC / "CURRENT_STATE.json"
    state = read(state_path)
    if (len(review["checks"]) != 16 or not all(review["checks"].values()) or
            not windows_closed(review["own_process"]) or args.review_command_seconds < review["internal_before_result_seconds"]):
        raise AssertionError("The separately completed review did not pass or actually exit")
    if (state["additional_accepted_count"] != 29 or len(state["accepted_mechanisms"]) != 29 or
            len(set(state["accepted_mechanisms"])) != 29 or len(state["acceptance_receipts"]) != 29 or
            state["Q68_R153_new_optimization_count"] != 0):
        raise AssertionError("The protected unique parent count changed")
    if (formal["sources"] != [data_identity(row["path"]) for row in formal["sources"]] or
            review["source"] != data_identity(review["source"]["path"])):
        raise AssertionError("An executed full composition or review source changed")
    links = []
    for name in ("Q68_R153_COMPOSITION_RESULTS.md", "NEXT_COMPOSED_SCENE_COST_BRIEF.md"):
        links.extend(re.findall(r"\]\((C:/[^)]+)\)", (DOC / name).read_text(encoding="utf-8")))
    if not links or not all(Path(p).is_file() for p in links):
        raise AssertionError("An actual direct-text report target is missing")
    native, combo = (formal["raw_complete_stage_medians"][key] for key in ("control", "combo"))
    fee_before_publisher = review["known_integration_fee_plus_registered10_reserve_before_review_exit_seconds"] + args.review_command_seconds
    fee = fee_before_publisher + time.perf_counter() - started
    bound = equal_fee_cost_bound(native, combo, fee)
    if bound["signed_reduction_at_known_lower_fee"] < .03 or combo / native > 1.01 or max(formal["whole_ratios"]) > 1.1:
        raise AssertionError("The unchanged final whole or registered net-cost gate failed")
    copies = OUT / "FINALIZER_SOURCES"
    copies.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), copies / Path(__file__).name)
    receipt = {
        "status": "accepted_optional_existing_Q68_R153_complete_verification_composition",
        "composition": ["Q68_COMPLETE_MULTIFILE_IDENTITY_BOUNDED_SCHEDULING", "R153_EXACT_NATIVE_LAYOUT_CAMERA_PIXEL_DIGEST_FUSION"],
        "new_optimization_count": 0, "parent_accepted_count": 29, "remaining": 21,
        "registered_frames": 1016, "complete_input_files": 1036, "fresh_complete_fixed_order_pairs": 3,
        "all15_scientific_fields_and2_actual_files_full_bytes_exact": True,
        "all_original_four_file_identity_phases_before_after_and_native_order_preserved": True,
        "normal_interfaces10_public_development360_and_actual_error_full_recovery_passed": True,
        "raw_complete_stage_medians": formal["raw_complete_stage_medians"], "raw_stage_reduction": formal["raw_stage_reduction"],
        "phase_and_resource_medians": review["phase_and_resource_medians"], "cost_before_publisher_actual_exit": bound,
        "final_cost": str(OUT / "FINAL_COST_001.json"), "review": str(OUT / "r0/RESULT.json"),
        "actual_complete_error_recovery": str(OUT / "v1/RESULT.json"), "interface_public_file_checks": str(OUT / "v0_interfaces/RESULT.json"),
        "stage_scope": review["stage_scope"], "fee_scope_incremental_composition_of_existing_accepted_parents": True,
        "existing_parent_cost_receipts_not_rewritten_or_assumed_zero": True,
        "registered_reserve_seconds": 10, "unknown_text_research_not_assumed_zero_or_total_ROI": True,
        "review_type": "separate_root_self_review_not_independent_agent", "local_effect_confirmed": True,
        "generalization_pending": True, "strict_blind_evaluation": False, "production_default_changed": False,
        "not_verified": ["training_speed", "render_quality_improvement", "entire_LIO_SfM_training_pipeline_speed", "new_object_whole_pipeline", "independent_GT", "other_machines"],
        "source": data_identity(Path(__file__))}
    atomic_json(accepted_path, receipt)
    process = psutil.Process()
    atomic_json(OUT / "PUBLICATION_001.json", {
        "status": "review_actually_closed_optional_composition_published_publisher_exit_fee_pending0",
        "own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "actual_review_command": {"returncode": 0, "seconds": args.review_command_seconds, "tool_chunk": args.review_command_chunk},
        "known_fee_plus_registered10_reserve_before_publisher_window_seconds": fee_before_publisher,
        "internal_before_result_seconds": time.perf_counter() - started,
        "source": data_identity(Path(__file__)), "new_optimization_count": 0, "parent_count": 29,
        "actual_direct_text_link_targets": links, "goal_remains_active": True})
    state.update(Q68_R153_status="accepted_optional_existing_composition_actual_publisher_exit_cost_pending0",
        Q68_R153_acceptance=accepted_path.relative_to(ROOT).as_posix(),
        Q68_R153_report="research/2026-10-05-gaussian-optimization-round3-50/Q68_R153_COMPOSITION_RESULTS.md",
        Q68_R153_review="04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001/r0/RESULT.json",
        Q68_R153_actual_complete_recovery="04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001/v1/RESULT.json",
        Q68_R153_complete_stage_raw_reduction=formal["raw_stage_reduction"],
        previous_turn_classification="actual_progress_existing_Q68_R153_composition_implemented_three_complete1016_pairs_interfaces_actual_errors_full_recovery_reviewed_new0_parent29",
        current_optimization_action="explore_current_composed_Scene_resize_decode_normalization_and_constructor_cost_next_preregistered_diagnostic_only",
        next_composed_scene_cost_brief="research/2026-10-05-gaussian-optimization-round3-50/NEXT_COMPOSED_SCENE_COST_BRIEF.md",
        current_state_updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    print({"accepted_existing_composition": True, "raw_gain": formal["raw_stage_reduction"], "net_before_publisher_exit": bound["signed_reduction_at_known_lower_fee"], "new": 0, "parent": 29})


if __name__ == "__main__":
    main()
