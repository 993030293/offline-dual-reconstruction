"""Publish the scoped work acceptance only after the separate reviewer exits."""

import json
from pathlib import Path
import shutil
import sys

import psutil

from run_cpu_dependency_restore import ROOT, read
from run_deferred_restore_equal_pair_001 import BASE, save, gpu


def main():
    result = read(BASE / "review_win0/RESULT.json")
    writer = result["writer_identity"]
    try:
        process = psutil.Process(writer["pid"])
        reviewer_ended = process.create_time() != writer["creation_time"] or not process.is_running()
    except psutil.NoSuchProcess:
        reviewer_ended = True
    if not reviewer_ended or not all(result["checks"].values()):
        raise AssertionError("The separate review did not pass and close")
    master = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if master["additional_accepted_count"] != 36 or len(master["accepted_mechanisms"]) != 36:
        raise ValueError("The frozen acceptance parent changed")
    fees = list(result["fees"])
    fees.append({"reason": "actual_external_Windows_review_287d90_exit0", "seconds": 1.4144142,
        "receipt": str(BASE / "review_win0/RESULT.json"), "saved_writer_ended": reviewer_ended})
    fee = sum(row["seconds"] for row in fees)
    b, c = result["B_whole_median_seconds"], result["C_whole_median_seconds"]
    net_work = 20 / (22 + fee * 22 / (6 * b))
    if net_work < .1 or (c + fee / 6) / (b + fee / 6) > 1.01:
        raise AssertionError("The charged work or complete whole protection failed")
    family_bytes = sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file())
    if family_bytes + 1024**2 >= 2 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise MemoryError("Final publication cannot keep the original storage reserve")
    artifact = {"schema_version": 1, "status": "scoped_work_acceptance_passed_pending_external_publication_close",
        "mechanism": "R168_V2_COMPLETE_CHECKPOINT_DEFERRED_BOOLEAN_MATERIALIZATION",
        "new_optimization_count": 1, "parent_count": 36, "accepted_count_after_registration": 37,
        "uniform_policy": "V6_net_speed2percent_or_preregistered_net_work10percent_quality_whole_recovery_kept",
        "scope": "complete_existing1016_native_checkpoint_restore_CUDA_boolean_host_materializations",
        "full_speed_accepted": False, "whole_speed_claim": "not_supported",
        "B_whole_seconds": result["B_whole_seconds"], "C_whole_seconds": result["C_whole_seconds"],
        "B_whole_median_seconds": b, "C_whole_median_seconds": c,
        "raw_work_population": {"baseline": 22, "candidate": 2, "all_normal_CUDA_leaf_comparisons": 22},
        "raw_work_reduction": 20 / 22, "net_work_reduction_index": net_work,
        "known_fee_seconds": fee, "fees": fees, "symmetric_fee_per_formal_side_seconds": fee / 6,
        "charged_complete_whole_ratio": (c + fee / 6) / (b + fee / 6),
        "unknown_research_effort_not_zero_not_total_research_ROI": True,
        "input_contract": {"images": 1016, "input_files": 1038, "gaussians": 134539,
            "checkpoint_iteration": 1000, "new_training_steps": 0, "all_nine_fields_and_next32_exact": True,
            "all_pixels_four_matrices_scene_files_and_device_alias_layout_exact": True},
        "checks": result["checks"], "actual_review_tool_exit": {"chunk_id": "287d90", "exit_code": 0,
            "saved_reviewer": writer, "saved_instance_ended": reviewer_ended},
        "native_review": str(BASE / "review_native0/RESULT.json"),
        "windows_review": str(BASE / "review_win0/RESULT.json"),
        "V1_early_SIGINT_missing_drain_rejected_preserved": True,
        "controller_injector_and_generated_syntax_failures_preserved_and_charged": True,
        "no_production_deployment_or_new_training_mapper_LIO": True,
        "generalization_boundary": "object_category_agnostic_numeric_component_not_unknown_object_full_pipeline_or_GT",
        "family_bytes_before_publication": family_bytes,
        "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "actual_publisher_exit_final_text_state_and_birth_closure_still_required": True}
    save(BASE / "ACCEPTANCE_WORK_V6_001.json", artifact)
    print(json.dumps({"new_count": 1, "parent": 36, "count_after_registration": 37,
        "known_fee_seconds": fee, "net_work_reduction_index": net_work,
        "charged_complete_whole_ratio": artifact["charged_complete_whole_ratio"],
        "full_speed_accepted": False, "reviewer_ended": reviewer_ended}), flush=True)


if __name__ == "__main__":
    main()
