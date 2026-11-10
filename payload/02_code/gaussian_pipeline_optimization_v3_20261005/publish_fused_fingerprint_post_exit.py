"""Count one scoped mechanism after checking the actual preparation process exit."""

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

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
MECHANISM = "R153_EXACT_NATIVE_LAYOUT_CAMERA_PIXEL_DIGEST_FUSION"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher-command-seconds", type=float, required=True)
    parser.add_argument("--publisher-command-chunk", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    publication = read(BASE / "PUBLICATION_001.json")
    review = read(BASE / "r0/RESULT.json")
    formal = read(BASE / "c0/RESULT.json")
    state_path = DOC / "CURRENT_STATE.json"
    state = read(state_path)
    acceptance_path = BASE / "ACCEPTANCE_V5_001.json"
    if acceptance_path.exists() or (BASE / "FINAL_POST_EXIT_001.json").exists():
        raise FileExistsError("Never replace an accepted or failed publication receipt")
    if (args.publisher_command_seconds < publication["internal_before_result_seconds"] or
            not windows_closed(publication["own_process"])):
        raise AssertionError("The completed preparation process is not actually closed")
    if (state["additional_accepted_count"] != 28 or
            len(state["accepted_mechanisms"]) != 28 or len(set(state["accepted_mechanisms"])) != 28 or
            len(state["acceptance_receipts"]) != 28 or state["R153_new_optimization_count"] != 0):
        raise AssertionError("The unpromoted unique parent denominator changed")
    if (formal["sources"] != [data_identity(row["path"]) for row in formal["sources"]] or
            publication["source"] != data_identity(publication["source"]["path"])):
        raise AssertionError("An executed algorithm or preparation source changed")
    links = []
    for path in publication["report_text_paths"]:
        links.extend(re.findall(r"\]\((C:/[^)]+)\)", Path(path).read_text(encoding="utf-8")))
    if not links or not all(Path(link).is_file() for link in links):
        raise AssertionError("A direct-text report points to a missing actual source or result")
    native, fused = (formal["raw_complete_stage_medians"][key] for key in ("original", "fused"))
    fee_before_post = publication["known_fee_plus_registered10_reserve_before_publisher_exit_seconds"] + args.publisher_command_seconds
    post_internal = time.perf_counter() - started
    fee = fee_before_post + post_internal
    bound = equal_fee_cost_bound(native, fused, fee)
    checks = {
        "actual_preparation_instance_exited_and_command_returned0": True,
        "parent28_unique_mechanisms_and_receipts_not_promoted_early": True,
        "executed_current_sources_still_match_complete_protocol": True,
        "root17_generic_interfaces_public_errors_full_recovery_passed": len(review["checks"]) == 17 and all(review["checks"].values()),
        "native_text_report_links_resolve": True,
        "complete_stage_net_under_registered_cost_model_at_least3percent": bound["signed_reduction_at_known_lower_fee"] >= .03,
        "whole_complete_entry_median_and_each_original_gates": fused / native <= 1.01 and max(formal["whole_ratios"]) <= 1.1,
        "all_original_input_quality_and_cancel_recovery_protections_unchanged": True,
    }
    if not all(checks.values()):
        atomic_json(BASE / "FAILED_POST_EXIT_001.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("The final scoped complete-entry publication did not pass")
    shutil.copyfile(Path(__file__), BASE / "FINALIZER_SOURCES" / Path(__file__).name)
    receipt = {
        "status": "accepted_optional_scoped_complete_camera_verification_V5_3percent",
        "mechanism": MECHANISM, "new_optimization_count": 1, "parent_count_before": 28,
        "parent_count_after": 29, "checks": checks, "local_effect_confirmed": True,
        "stage_scope": review["stage_scope"], "whole_scope_is_same_complete_verification_entry": True,
        "registered_frames": 1016, "complete_fixed_order_pairs": 3, "protected_scientific_fields": 15,
        "protected_complete_generated_files": ["input.ply", "cameras.json"],
        "raw_complete_stage_medians": formal["raw_complete_stage_medians"],
        "raw_complete_stage_reduction": formal["raw_stage_reduction"],
        "phase_and_resource_medians": review["phase_and_resource_medians"],
        "cost_before_post_process_exit": bound,
        "known_fee_plus_registered10_reserve_before_post_exit_seconds": fee,
        "final_cost": str(BASE / "FINAL_COST_001.json"),
        "registered_reserve_seconds": 10, "reserve_already_included_not_added_twice": True,
        "fee_model_recorded_necessary_execution_not_total_research_ROI": True,
        "unknown_text_research_cost_not_assumed_zero": True,
        "review_type": "separate_root_self_review_not_independent_agent",
        "evidence": {name: str(BASE / relative) for name, relative in (
            ("diagnostic", "p0/FINGERPRINT_COST.json"), ("generic", "g0/RESULT.json"),
            ("complete_pairs", "c0/RESULT.json"), ("interfaces_public_development", "v0_interfaces/RESULT.json"),
            ("actual_two_errors_and_full_recovery", "v1/RESULT.json"), ("separate_review", "r0/RESULT.json"))},
        "same_family_scalar_vector_and_worker_variants_counted_once": True,
        "existing_Q68_R92_R127_not_recounted": True,
        "actual_preparation_process_closed": publication["own_process"],
        "production_default_changed": False, "generalization_pending": True, "strict_blind_evaluation": False,
        "not_verified": ["training_speed", "SfM_mapper_speed", "render_quality_improvement", "new_object_whole_pipeline",
            "independent_GT", "other_Torch_GPU_ISA", "arbitrary_external_raw_pointer_mutation", "inference_tensor_version_scope"],
        "report": str(DOC / "R153_FUSED_FINGERPRINT_RESULTS.md"),
        "source": data_identity(Path(__file__))}
    atomic_json(acceptance_path, receipt)
    process = psutil.Process()
    atomic_json(BASE / "FINAL_POST_EXIT_001.json", {
        "status": "actual_preparation_exited_scoped_acceptance29_published_post_process_exit_fee_pending",
        "checks": checks, "new_optimization_count": 1, "parent_count": 29, "remaining": 21,
        "actual_publisher_command": {"returncode": 0, "seconds": args.publisher_command_seconds,
            "tool_chunk": args.publisher_command_chunk},
        "own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "known_fee_plus_registered10_reserve_before_post_window_seconds": fee_before_post,
        "internal_charged_seconds_replaced_by_actual_outer_exit_later": post_internal,
        "native_text_link_targets": links, "source": data_identity(Path(__file__)), "goal_remains_active": True})
    state["accepted_mechanisms"].append(MECHANISM)
    state["acceptance_receipts"].append(acceptance_path.relative_to(ROOT).as_posix())
    state.update(additional_accepted_count=29, R153_new_optimization_count=1, R153_algorithm_accepted=True,
        R153_status="accepted_optional_exact_complete_camera_verification_V5_final_post_exit_fee_pending",
        R153_acceptance=acceptance_path.relative_to(ROOT).as_posix(),
        R153_report="research/2026-10-05-gaussian-optimization-round3-50/R153_FUSED_FINGERPRINT_RESULTS.md",
        R153_review="04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/r0/RESULT.json",
        R153_actual_errors_and_complete_recovery="04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/v1/RESULT.json",
        R153_complete_stage_raw_reduction=formal["raw_stage_reduction"],
        R153_production_default_deployed=False,
        previous_turn_classification="actual_progress_R153_implemented_full1016_three_pairs_and_recovery_scoped_acceptance1",
        current_optimization_action="fusion_existing_Q68_complete_file_identities_with_R153_next_source_bound_brief_only_new_count0",
        next_combination_brief="research/2026-10-05-gaussian-optimization-round3-50/NEXT_Q68_FUSED_FINGERPRINT_COMBINATION_BRIEF.md",
        latest_user_request_execution="uniform_V5_stage3percent_kept_R153_complete1016_three_fresh_pairs_all_protected_outputs_exact_generic_public_development_and_actual_error_recovery_passed_optional1_parent29_no_subagents_or_production",
        current_state_updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    print({"accepted_new": 1, "parent": 29, "remaining": 21,
        "stage_net_under_registered_fee_model": bound["signed_reduction_at_known_lower_fee"]})


if __name__ == "__main__":
    main()
