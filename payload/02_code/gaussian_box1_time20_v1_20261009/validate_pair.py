"""Audit the complete frozen time, image, training, preview and scale requirements."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json, read_json
from sqlite_verified_backup import complete_bytes_equal
from run_side import windows_to_wsl
from runtime import data_identity


def audit_training_sources(campaigns, configs):
    cases, all_valid = [], True
    for campaign, config in zip(campaigns, configs):
        folder = Path(campaign).resolve() / "box1/optimized/training/source_snapshot"
        rows = config["implementation"]
        names = [row["path"].replace("\\", "/").rsplit("/", 1)[-1] for row in rows]
        actual_names = sorted(path.name for path in folder.iterdir() if path.is_file())
        population_ok = len(names) == len(set(names)) and sorted(names) == actual_names
        checks = []
        for name, declared in zip(names, rows):
            target = folder / name
            owned = target.resolve().parent == folder.resolve()
            bound = declared["path"] in (str(target), target.as_posix(), windows_to_wsl(target))
            current = data_identity(target) if target.is_file() else None
            valid = owned and bound and current is not None and all(
                current[key] == declared[key] for key in ("bytes", "sha256"))
            checks.append({"name": name, "owned_path_bound": owned and bound,
                           "complete_source_identity_unchanged": valid})
        valid = population_ok and all(row["complete_source_identity_unchanged"] for row in checks)
        all_valid = all_valid and valid
        cases.append({"folder": folder, "names": names, "checks": checks, "population_ok": population_ok})
    differences = []
    if cases[0]["names"] != cases[1]["names"]:
        differences.append("declared_source_order_or_population")
    else:
        for name in cases[0]["names"]:
            first, second = [case["folder"] / name for case in cases]
            if not first.is_file() or not second.is_file() or not complete_bytes_equal(first, second):
                differences.append(name)
    return {"passed": all_valid and not differences, "different_complete_sources": differences,
            "cases": [{**case, "folder": str(case["folder"])} for case in cases],
            "private_ownership_paths_differ_but_entire_executed_source_bytes_must_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-completion", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for campaign in (args.reference, args.candidate):
        side = campaign / "box1/optimized"
        manifest = read_json(side / "initial/INITIAL_FRAME_MANIFEST.json")
        admission = read_json(side / "export/TRAINING_ADMISSION.json")
        dataset = Path(admission["dataset"])
        actual_images = sorted(path.name for path in (dataset / "images").iterdir() if path.is_file())
        full_path = args.candidate_completion if campaign == args.candidate and args.candidate_completion else campaign / "RESULT.json"
        full = read_json(full_path)
        localization_path = Path(full["localization_path"]) if full.get("recovered_current_campaign_only") else side / "localization/LOCALIZATION.json"
        case = {"campaign": str(campaign), "full": full,
                "contract": read_json(campaign / "CONTRACT.json"), "manifest": manifest,
                "initial_contract": read_json(side / "initial/CONTRACT.json"),
                "exclusions": read_json(side / "initial/EXCLUDED_EVALUATION_IDS.json"),
                "evaluation_frames": read_json(side / "initial/EVALUATION_FRAMES.json"),
                "admission": admission, "dataset_images": actual_images,
                "roles": read_json(side / "roles/TRAINING_ROLES.json"),
                "training_config": read_json(side / "training/CONFIG.json"),
                "training": read_json(side / "training/STATE.json"),
                "evaluation": read_json(side / "evaluation/RESULT.json"),
                "localization": read_json(localization_path),
                "scale": read_json(side / "scale_consistency/RESULT.json"),
                "alignment": read_json(side / "metric/ALIGNMENT.json"),
                "packing": read_json(campaign / "native_packing/RESULT.json"),
                "pose_gap": read_json(side / "pose_gap/RESULT.json")}
        cases.append(case)
    reference, candidate = cases
    gates = {}

    def gate(name, passed, evidence):
        gates[name] = {"passed": bool(passed), "evidence": evidence}

    if candidate["full"].get("recovered_current_campaign_only"):
        completed = candidate["full"]
        original = read_json(Path(completed["original_failure"]))
        original_cost = read_json(Path(completed["original_failed_outer_receipt"]))
        costs = [read_json(Path(row["receipt"]))["elapsed_seconds"] for row in completed["recovery_costs"]]
        expected = original_cost["elapsed_seconds"]+sum(costs)+completed["diagnostic_cost"]["seconds"]
        gate("recovery_preserves_primary_and_counts_original_failure_and_all_retries",
             original["raw_to_preview_seconds"] == completed["raw_to_preview_seconds"] and
             original["primary_pipeline_finished"] and completed["raw_to_preview_plus_full_QA_seconds"] == expected and
             completed["all_actual_localization_failures_and_retries_included"] and
             not completed["historical_other_campaign_model_feature_or_image_cache_used"],
             {"full_QA_timing_kind": completed["full_QA_timing_kind"], "original_outer_seconds": original_cost["elapsed_seconds"],
              "recovery_seconds": costs, "diagnostic_seconds": completed["diagnostic_cost"]["seconds"]})

    gate("same_raw_recording", all(reference["contract"][key] == candidate["contract"][key]
         for key in ("raw", "raw_bytes", "raw_mtime_ns")), [case["contract"]["raw"] for case in cases])
    gate("fresh_independent_full_runs", all(case["contract"]["fresh_private_inputs"] and
         not case["contract"]["historical_derived_inputs_used"] and case["full"]["raw_identity_unchanged"] and
         case["full"]["status"] == "complete_raw_model_preview_and_QA_comparison_pending" for case in cases),
         [case["full"]["status"] for case in cases])
    gate("same_complete_initial_selection_and_policy", all(reference["manifest"][key] == candidate["manifest"][key]
         for key in ("selected_frames", "policy", "camera_intrinsics")),
         {"selected_counts": [case["manifest"]["receipt"]["selected_count"] for case in cases]})
    gate("initial_selection_uses_no_SfM_or_GT", all(not case["initial_contract"]["SfM_or_GT_for_initial_selection"]
         and not case["initial_contract"]["old_runtime_inputs_used"] for case in cases),
         [case["initial_contract"]["coarse_pose_source"] for case in cases])
    gate("all_original_and_additional_frames_registered", all(case["pose_gap"]["original_registration_guard_passed"] and
         case["pose_gap"]["pose_gap_registration_guard_passed"] and not case["roles"]["initial_not_registered"]
         for case in cases), [case["roles"]["registered_count"] for case in cases])
    gate("same_complete_training_image_set", reference["dataset_images"] == candidate["dataset_images"] and
         all(len(case["dataset_images"]) == case["admission"]["training_count"] == case["training"]["image_count"] ==
             case["roles"]["registered_count"] == case["roles"]["support_pass_count"] for case in cases),
         {"image_counts": [len(case["dataset_images"]) for case in cases]})
    gate("both_complete_same_12000_iterations", all(case["training"]["status"] == "completed" and
         case["training"]["iteration"] == case["training"]["target_iterations"] ==
         case["training"]["executed_iterations"] == 12000 and case["training"]["resumed_iteration"] == 0 for case in cases),
         [case["training"]["iteration"] for case in cases])
    configs = [case["training_config"] for case in cases]
    parameter_rows = [{key: value for key, value in config["parameters"].items()
                       if key not in ("source_path", "model_path")} for config in configs]
    sources = audit_training_sources([case["campaign"] for case in cases], configs)
    gate("same_entire_executed_training_sources", sources["passed"], sources)
    shared_config_fields = ("runtime", "intervention", "seed", "ply_serialization", "compact_host_images",
                            "trim_host_after_load", "gpu_resident_inputs", "target_density",
                            "optional_ssim_identity", "optional_gsplat_identity", "torch", "gpu",
                            "camera_optimization", "expected_image_count", "external_adapters", "input_version",
                            "gpu_budget_bytes", "ram_budget_bytes", "fixed_fused_ssim")
    gate("same_native_training_configuration_and_backend", parameter_rows[0] == parameter_rows[1] and
         all(configs[0][key] == configs[1][key] for key in shared_config_fields) and
         all(config["optional_gsplat_identity"] is not None for config in configs),
         {"shared_fields": shared_config_fields, "parameters_without_ownership_paths": parameter_rows,
          "seeds": [config["seed"] for config in configs]})
    datasets = [Path(case["admission"]["dataset"]) / "images" for case in cases]
    image_differences = [name for name in reference["dataset_images"] if
        not (datasets[1] / name).is_file() or not complete_bytes_equal(datasets[0] / name, datasets[1] / name)]
    gate("same_complete_training_image_bytes", not image_differences,
         {"complete_image_count": len(reference["dataset_images"]), "different_images": image_differences})
    protected_overlap = []
    protected_shape_ok = True
    for case in cases:
        exclusions = case["exclusions"]
        protected = {int(frame) for frame in exclusions["frames"]}
        evaluations = set(exclusions["evaluation_frames"])
        protected_shape_ok = protected_shape_ok and exclusions["frozen_before_scoring"] and len(protected) == 48 and \
            len(evaluations) == 24 and {int(row["camera_frame"]) for row in case["evaluation_frames"]["rows"]} == evaluations
        protected_shape_ok = protected_shape_ok and all(
            exclusions["frames"].get(str(frame)) == "development_evaluation" and
            exclusions["frames"].get(str(frame+1)) == "development_evaluation_adjacent" for frame in evaluations)
        overlaps = []
        for name in case["dataset_images"]:
            match = re.fullmatch(r"frame_(\d+)_\d+\.jpg", name)
            if match is None or int(match.group(1)) in protected:
                overlaps.append(name)
        protected_overlap.append(overlaps)
    gate("same48_frozen_protected_frames_excluded_from_all_training", protected_shape_ok and
         reference["exclusions"] == candidate["exclusions"] and not any(protected_overlap) and
         reference["evaluation_frames"] == candidate["evaluation_frames"],
         {"protected_counts": [len(case["exclusions"]["frames"]) for case in cases],
          "training_overlap_or_unparseable_names": protected_overlap})
    gate("native_complete_assets_and_full_scene_ready", all(case["packing"]["status"] == "native_assets_and_viewer_ready" and
         case["packing"]["viewer"]["ready"] and not case["packing"]["viewer"]["errors"] and
         case["packing"]["viewer"]["fullGaussianCount"] == case["training"]["gaussian_count"] and
         all(row["complete_native_arrays_equal"] for row in case["packing"]["native"]["records"]) for case in cases),
         [case["training"]["gaussian_count"] for case in cases])
    gate("same_protected_evaluation_frames", [row["image_name"] for row in reference["evaluation"]["rows"]] ==
         [row["image_name"] for row in candidate["evaluation"]["rows"]], "all24_frozen_raw_indices")
    summaries = [case["evaluation"]["summary"]["full_image"] for case in cases]
    all24 = all(summary["rendered_count"] == summary["complete_denominator"] == 24 for summary in summaries)
    gate("complete24_localization_and_rendering", all24 and all(case["localization"]["status"] == "all24_localized"
         for case in cases), summaries)
    delta_psnr = summaries[1]["mean_PSNR_dB"] - summaries[0]["mean_PSNR_dB"]
    delta_ssim = summaries[1]["mean_SSIM"] - summaries[0]["mean_SSIM"]
    gate("image_quality_preserved", all24 and delta_psnr >= -.1 and delta_ssim >= -.002,
         {"PSNR_delta_dB": delta_psnr, "SSIM_delta": delta_ssim, "PSNR_min": -.1, "SSIM_min": -.002})
    scan_ids = [[row["raw_scan_id"] for row in case["scale"]["rows"]] for case in cases]
    gate("same64_raw_LiDAR_scans", scan_ids[0] == scan_ids[1] and all(case["scale"]["selected_scan_denominator"] == 64
         and case["scale"]["source_unchanged"] for case in cases), scan_ids)
    for key in ("p50_m", "p95_m"):
        values = [case["scale"]["sparse_to_raw_LiDAR"][key] for case in cases]
        maximum = values[0] + max(.05*values[0], .002)
        gate("LiDAR_support_"+key, values[1] <= maximum,
             {"reference": values[0], "candidate": values[1], "maximum": maximum})
    residuals = [case["alignment"]["residuals_m"]["p50"] for case in cases]
    maximum = residuals[0] + max(.05*residuals[0], .002)
    gate("coarse_metric_position_consistency", residuals[1] <= maximum,
         {"reference": residuals[0], "candidate": residuals[1], "maximum": maximum})
    timing = {}
    for key in ("raw_to_model_seconds", "raw_to_preview_seconds", "raw_to_preview_plus_full_QA_seconds"):
        first, second = [case["full"][key] for case in cases]
        timing[key] = {"reference": first, "candidate": second, "saved_seconds": first-second,
                       "reduction_fraction": 1-second/first}
    gate("complete_raw_to_preview_over20_percent", timing["raw_to_preview_seconds"]["reduction_fraction"] > .20, timing)
    gate("full_total_including_QA_over20_percent", timing["raw_to_preview_plus_full_QA_seconds"]["reduction_fraction"] > .20,
         timing["raw_to_preview_plus_full_QA_seconds"])
    stage_costs = []
    for case in cases:
        side = Path(case["campaign"]) / "box1/optimized"
        rows = []
        for path in sorted(side.glob('*_process.json')):
            record = read_json(path)
            rows.append({"stage": path.stem.removesuffix('_process'), "status": record["status"],
                         "elapsed_seconds": record.get("elapsed_seconds"), "receipt": str(path)})
        stage_costs.append({"campaign": case["campaign"], "rows": rows})
    result = {"status": "paired_primary_gates_passed_additional_checks_pending" if all(g["passed"] for g in gates.values())
              else "paired_acceptance_failed_goal_still_active", "gates": gates, "timing": timing,
              "stage_costs": stage_costs, "all_primary_gates_passed": all(g["passed"] for g in gates.values()),
              "independent_GT_accuracy_claim": False, "strict_blind_quality_claim": False,
              "single_complete_pair_is_not_a_statistically_stable_or_all_object_speed_claim": True}
    atomic_json(args.output / "RESULT.json", result)
    print({"status": result["status"], "failed": [key for key, value in gates.items() if not value["passed"]],
           "timing": timing})


if __name__ == "__main__":
    main()
