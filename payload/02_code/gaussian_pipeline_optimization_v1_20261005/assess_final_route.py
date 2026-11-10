"""Reconcile stage evidence with the actual new-map training and display receipts."""

import argparse
import json
from pathlib import Path

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, mechanisms = {}, []

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise ValueError("Final route acceptance refused: " + name)

    def stage(identifier, evidence, value):
        check(identifier + "_stage_evidence", value)
        mechanisms.append({"id": identifier, "evidence": str(args.root / evidence)})

    quality = read(args.root / "p1_quality_profile_001/RESULT.json")
    raw = read(args.root / "p1_raw_combined_001/EQUIVALENCE.json")
    warm = read(args.root / "p1_raw_combined_warm_001/EQUIVALENCE.json")
    check("all7338_raw_decisions_and1002_images_and_pairs_equivalent", all(
        item["status"] == "passed_full_sequence_equivalence" and item["raw_frame_count"] == 7338
        and item["image_count"] == 1002 and item["all_decisions_and_selection_reasons_equal"]
        and item["all_exported_JPEG_payloads_equal"] and item["all_candidate_pairs_equal"] for item in (raw, warm)))
    p0 = read(args.root / "p0_pair_s0_001/evaluation/EVALUATION.json")
    stage("P0_FULL_K", "p0_pair_s0_001/evaluation/EVALUATION.json", p0["development_metric_guard_passed"]
          and p0["full_minus_legacy"]["target"]["mean_PSNR_dB"] > 0)
    stage("P1_QUALITY_THROUGHPUT", "p1_quality_profile_001/RESULT.json",
          quality["comparisons"]["parallel_scoring"]["stage_speed_gate_passed"])
    stage("P1_COMPACT_DERIVATIVE", "p1_quality_profile_001/RESULT.json",
          quality["comparisons"]["compact_derivative"]["stage_speed_gate_passed"])
    cache = read(args.root / "p1_cache_contract_001/RESULT.json")
    stage("P1_PROVENANCE_CACHE", "p1_cache_contract_001/RESULT.json", cache["status"] == "passed_cache_contract"
          and cache["wall_reduction_fraction"] >= .1 and all(cache[key] for key in
          ("source_and_contract_changes_miss", "partial_commit_ignored", "corrupt_value_rejected", "all_128_scores_equal")))
    coverage = read(args.root / "p1_selection_profile_001/RESULT.json")
    stage("P1_BATCHED_COVERAGE", "p1_selection_profile_001/RESULT.json",
          coverage["speed_gate_passed"] and coverage["all_7338_decisions_equal"])
    e2 = read(args.root / "e2_temporal_augmentation_001/ACCEPTANCE.json")
    stage("E2_QUALITY_SUPPORTED_TEMPORAL_AUGMENTATION", "e2_temporal_augmentation_001/ACCEPTANCE.json",
          e2["stage_accepted"] and e2["original_registered_after"] == 256 and e2["total_registered_after"] == 273
          and e2["original_frames_removed"] == 0 and not e2["quality_gate_relaxed"])
    recovery = read(args.root / "final_combination_recovery12000_001/COMBINATION_ACCEPTANCE.json")
    stage("P3_COMPLETE_TRAINING_RECOVERY", "final_combination_recovery12000_001/COMBINATION_ACCEPTANCE.json",
          recovery["full_budget_combination_accepted"] and recovery["all17_state_sampling_native_PLY_and_save_schedule_checks_passed"]
          and recovery["all10_invalid_checkpoint_cases_rejected"] and recovery["both_actual_six_view_reviews_passed"])
    p8 = read(args.root / "p8_demand_viewer_001/ACCEPTANCE.json")
    stage("P8_DEMAND_VIEWER", "p8_demand_viewer_001/ACCEPTANCE.json", p8["counted"] and p8["all_six_protected_framebuffers_identical"])
    p10 = read(args.root / "p10_packed_viewer_002/ACCEPTANCE.json")
    stage("P10_PACKED_VIEWER_ASSET", "p10_packed_viewer_002/ACCEPTANCE.json", p10["median_stage_reduction"] >= .1
          and p10["all_four_native_buffers_roundtrip_exact"] and p10["six_raw_framebuffers_identical"])
    p6 = read(args.root / "p6b_gsplat12k_acceptance_001/ACCEPTANCE.json")
    stage("P6_GSPLAT_BACKEND", "p6b_gsplat12k_acceptance_001/ACCEPTANCE.json",
          p6["status"] == "accepted_stage_optimization" and all(p6["checks"].values()))
    p14 = read(args.root / "p14_lazy_viewer_002/ACCEPTANCE.json")
    stage("P14_LAZY_MODEL_LOADING", "p14_lazy_viewer_002/ACCEPTANCE.json", p14["median_initialization_reduction"] >= .1
          and p14["all_original_assets_and_SH_preserved"] and p14["six_raw_framebuffers_identical"])
    p18 = read(args.root / "p18_raw_batched_pose_001/ACCEPTANCE.json")
    stage("P18_BATCHED_POSE_INTERPOLATION", "p18_raw_batched_pose_001/ACCEPTANCE.json", p18["counted"]
          and p18["all7338_decisions_and_reasons_equal"] and p18["all1002_exported_JPEG_payloads_equal"] and p18["candidate_pair_file_equal"])
    p22 = read(args.root / "p22_serialization_001/ACCEPTANCE.json")
    stage("P22_VECTORIZED_PLY", "p22_serialization_001/ACCEPTANCE.json", p22["status"] == "accepted_export_stage"
          and p22["all_three_native_candidate_file_pairs_byte_identical"] and p22["all_six_native_loader_parameter_roundtrips_exact"]
          and p22["all_six_save_operations_preserve_rng"])
    p25 = read(args.root / "p25_filter_viewer_002/ACCEPTANCE.json")
    stage("P25_RENDER_FILTER_CONTRACT", "p25_filter_viewer_002/ACCEPTANCE.json", p25["counted"]
          and p25["new_maximum_alpha_error"] <= p25["new_limit"] and p25["browser_checks_passed"])
    p27 = read(args.root / "p27_matching_acceptance_002/ACCEPTANCE.json")
    stage("P27_LAZY_SIFT_FEATURES", "p27_matching_acceptance_002/ACCEPTANCE.json",
          p27["status"] == "accepted_stage_optimization" and all(p27["checks"].values()))
    check("fifteen_distinct_stage_mechanisms", len({item["id"] for item in mechanisms}) == len(mechanisms) == 15)
    admission = read(args.root / "new_map_export_003/TRAINING_ADMISSION.json")
    check("explicit_full_map_and_training_denominators", admission["raw_candidate_count"] == 7338
          and admission["initial_count"] == 1002 and admission["full_registered_denominator"] == 1021
          and admission["training_count"] == 1016 and len(admission["excluded_names"]) == 5
          and admission["protected_development_exclusion_count"] == 48 and admission["old_all_frame_failures_remain_failed"])
    state = read(args.pair / "STATE.json")
    check("new_route_actual_training_and_restore", state["status"] == "training_and_restore_passed_quality_pending"
          and all(state["checks"].values()) and [row["returncode"] for row in state["runs"]] == [0, 86, 0])
    rejected = read(args.pair / "rejection_checks/RESULT.json")
    check("ten_new_route_invalid_checkpoints_rejected", len(rejected["checks"]) == 10
          and all(item["rejected"] for item in rejected["checks"].values()))
    evaluation = read(args.pair / "evaluation/EVALUATION.json")
    check("new_route_all24_full_and_target_guards", evaluation["development_metric_guard_passed"]
          and all(model["summary"][region]["count"] == 24 for model in evaluation["results"].values() for region in ("full_image", "target")))
    check("new_route_six_actual_views_reviewed", read(args.pair / "evaluation/VISUAL_REVIEW.json")["passed"])
    localization = args.root / "p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
    contract = read(args.pair / "evaluation/EVALUATION_CONTRACT.json")
    check("new_map_localization_frozen_for_both_models", contract["input_version"] == "new_map_admission_v1"
          and contract["localization_source"]["sha256"] == data_identity(localization)["sha256"])
    inputs = [read(args.pair / label / "INPUT_MANIFEST.json") for label in ("reference", "combined")]
    check("same1016_inputs_in_both_models", all(item["image_count"] == 1016 and item["input_version"] == "new_map_admission_v1" for item in inputs)
          and [item["sha256"] for item in inputs[0]["files"]] == [item["sha256"] for item in inputs[1]["files"]])
    packaging = read(args.viewer / "PACKING_VALIDATION.json")
    check("four_actual_full_SH_assets_roundtrip_exact", packaging["status"] == "four_native_assets_exported_and_roundtrip_verified"
          and len(packaging["records"]) == 4 and all(len(row["checks"]) == 9 and all(item["passed"] for item in row["checks"])
          and row["maxSh"] == 3 for row in packaging["records"]))
    for label, iterations in (("reference", 10000), ("combined", 12000)):
        viewer_root = args.viewer / label
        assets = read(viewer_root / "ASSET_MANIFEST.json")
        model = args.pair / label / f"model/point_cloud/iteration_{iterations}/point_cloud.ply"
        check(label + "_same_training_evaluated_displayed_model", assets["scene"]["source_ply_sha256"]
              == evaluation["results"][label]["model"]["sha256"] == data_identity(model)["sha256"])
        for region in ("scene", "box"):
            check(label + "_" + region + "_actual_packed_identity", assets[region]["asset_sha256"]
                  == data_identity(viewer_root / "cases/packed" / (region + ".spk"))["sha256"])
        config = read(args.pair / label / "CONFIG.json")
        check(label + "_native_CPU_full_K_classic_training", not config["gpu_resident_inputs"]
              and not config["compact_host_images"] and not config["parameters"]["antialiasing"]
              and config["intervention"]["projection"] == "full")
    browser = read(args.viewer / "BROWSER_VALIDATION.json")
    check("actual_new_route_free_navigation_and_all1016_cameras", browser["status"] == "browser_checks_passed_visual_review_pending"
          and bool(browser["checks"]) and all(row["passed"] for row in browser["checks"]) and not browser["errors"])
    visual = read(args.viewer / "VISUAL_REVIEW.json")
    check("actual_new_route_free_view_visual_review", visual["passed"] and visual["actual_images_reviewed"] == 12)
    atomic_json(args.output, {"status": "accepted_fifteen_mechanisms_and_actual_new_route_combination",
        "full_route_combination_accepted": True, "accepted_count": 15, "mechanisms": mechanisms,
        "checks": checks, "new_route_pair": str(args.pair), "new_route_viewer": str(args.viewer),
        "source": data_identity(Path(__file__)), "quality_deltas": evaluation["candidate_minus_control"],
        "self_review_only": True, "strict_blind_test": False, "independent_geometry_accuracy_claim": False,
        "aggregate_speedup_claim": False, "new_sequence_or_Oxford_GT_evaluated": False,
        "boundary": "Usable within the verified Box2 engineering route and each explicitly scoped stage; prior failed branches remain failed"})
    print(json.dumps({"status": "accepted_fifteen_mechanisms_and_actual_new_route_combination", "checks": len(checks)}))


if __name__ == "__main__":
    main()
