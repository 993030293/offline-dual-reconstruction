"""Consolidate completed receipts without changing frozen measurements or failures."""

import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    pair = read(root / "post_checks_002/paired_acceptance/RESULT.json")
    if not pair["all_primary_gates_passed"] or not all(row["passed"] for row in pair["gates"].values()):
        raise ValueError("A primary gate has not passed")
    repeats = read(root / "post_checks_002/matching_repeats/RESULT.json")
    if repeats["status"] != "all_sparse_native_repeats_exact" or len(repeats["results"]) != 3:
        raise ValueError("The three native component checks are incomplete")
    for row in repeats["results"]:
        if not row["all_requested_native_pair_rows_and_blobs_exact"] or not all(
                item["all_values_and_blobs_equal"] for item in row["complete_native_tables"]):
            raise ValueError("A native repeat changed database values")
    box2 = read(root / "post_checks_002/box2_sparse_probe/RESULT.json")
    if not box2["original_registration_guard_passed"] or not box2["pose_gap_registration_guard_passed"]:
        raise ValueError("The separate Box2 registration check failed")
    preview = {}
    for name, folder in [("reference", "reference_settled_preview_002"), ("candidate", "candidate_settled_preview_001")]:
        controls = read(root / "post_checks_002" / (name + "_preview") / "RESULT.json")
        pixels = read(root / "post_checks_002" / folder / "RESULT.json")
        if controls["pageErrors"] or controls["final"]["errors"] or len(pixels["rows"]) != 4:
            raise ValueError("A model preview check is incomplete")
        if not all(row["final"]["nonUniformImage"] for row in pixels["rows"]):
            raise ValueError("A checked WebGL view is blank")
        preview[name] = {"controls": str(root / "post_checks_002" / (name + "_preview") / "RESULT.json"),
            "pixels": str(root / "post_checks_002" / folder / "RESULT.json"),
            "url": pixels["url"], "default_nonblank_observed_by_ms": pixels["rows"][0]["final"]["elapsedSinceNavigationMs"],
            "first_ready_signal_precedes_visible_splats": not pixels["rows"][0]["first"]["nonUniformImage"]}
    rows = []
    by_side = []
    for campaign in pair["stage_costs"]:
        values = {item["stage"]: item["elapsed_seconds"] for item in campaign["rows"]}
        by_side.append(values)
    for stage in by_side[0]:
        reference, candidate = by_side[0][stage], by_side[1][stage]
        rows.append({"category": "guarded_pipeline_stage", "stage": stage,
            "reference_seconds": reference, "candidate_seconds": candidate,
            "saved_seconds": reference - candidate, "reduction_fraction": 1 - candidate / reference})
    for stage, values in pair["timing"].items():
        rows.append({"category": "primary_clock" if stage != "raw_to_preview_plus_full_QA_seconds" else "cumulative_QA_execution",
            "stage": stage, "reference_seconds": values["reference"], "candidate_seconds": values["candidate"],
            "saved_seconds": values["saved_seconds"], "reduction_fraction": values["reduction_fraction"]})
    native, alignments = {}, {}
    for name, folder in [("reference", "formal_reference_001"), ("candidate", "formal_sparse_001")]:
        side = root / folder / "box1/optimized"
        sfm, augmentation = read(side / "sfm/RESULT.json"), read(side / "pose_gap/RESULT.json")
        native[name] = {"first_SfM_mapping_seconds": sfm["mapping_seconds"],
            "first_SfM_matching_seconds": sfm["matching_seconds"], "augmentation": augmentation["timings"]}
        alignment = read(side / "metric/ALIGNMENT.json")
        alignments[name] = {"Sim3_scale": alignment["new_Sim3"]["scale"], "coarse_position_residuals_m": alignment["residuals_m"]}
    recovery = read(root / "candidate_completion_recovery_001/RESULT.json")
    if not recovery["all_actual_localization_failures_and_retries_included"]:
        raise ValueError("Recovery costs have not been accounted for")
    result = {"status": "completed_numerical_checks_and_visible_preview_text_self_review_pending",
        "scope": "one_machine_Box1_engineering_comparison_with_explicit_current_campaign_QA_recovery",
        "timing": pair["timing"], "stages": rows, "native_costs": native,
        "alignments": alignments, "complete_primary_gates": pair["gates"],
        "matching_repeat_seconds": [row["native_matching_seconds"] for row in repeats["results"]],
        "matching_repeats_all_native_tables_equal": True,
        "box2_local_probe": {"original_count": box2["original_count"], "added_count": box2["added_count"],
            "registered_count": box2["models"][0]["registered_count"], "timings": box2["timings"],
            "historical_inputs_diagnostic_only": True, "fresh_full_Box2_speed_claim": False},
        "preview": preview, "QA_cost_accounting": {key: recovery[key] for key in [
            "outer_original_seconds", "recovery_costs", "diagnostic_cost", "full_QA_timing_kind",
            "full_QA_is_not_contiguous_operator_wall_time", "operator_investigation_editing_and_idle_not_counted_as_algorithm_time"]},
        "boundaries": {"no_independent_GT": True, "not_strict_blind_evaluation": True,
            "no_statistically_stable_full_pipeline_speed_claim": True,
            "no_all_object_generalization_claim": True, "GPU_BA_candidate_not_executed": True,
            "recorded_odometry_decoded_no_full_LIO_replay": True,
            "original_pair_failure_preserved": True, "frozen_full_entry_requires_explicit_QA_recovery_when_optional_statistic_is_NaN": True,
            "first_ready_signal_not_first_visible_pixel_clock": True,
            "novel_drone_view_floaters_remaining": True}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "FINAL_EVIDENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "TIMING.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": result["status"], "output": str(args.output), "timing": result["timing"]}))


if __name__ == "__main__":
    main()
