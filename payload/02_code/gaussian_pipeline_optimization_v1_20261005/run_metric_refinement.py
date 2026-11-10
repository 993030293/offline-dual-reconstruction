"""Bind a fresh visual map to its current coarse metric positions with verified factors."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity
from pose_gap_candidates import select_pose_gap_candidates
from coarse_prior_provenance import (CURRENT_ALIGNMENT_SOURCE, CURRENT_METRIC_SOURCE,
                                    summarize_used_priors, unique_coarse_lookup)


def observation_identity(model):
    return [(int(image.image_id), index, int(point.point3D_id))
            for image in sorted(model.images.values(), key=lambda image: image.image_id)
            for index, point in enumerate(image.points2D) if point.has_point3D()]


def verified_pose_gap_names(initial_frames, receipt_path):
    if receipt_path is None:
        return set()
    manifest = json.loads((initial_frames / "INITIAL_FRAME_MANIFEST.json").read_text())
    rows = [json.loads(line) for line in (initial_frames / "frame_decisions.jsonl").read_text().splitlines()]
    candidates, decisions = select_pose_gap_candidates(rows, manifest["policy"], manifest["receipt"]["sharpness_min"])
    receipt = json.loads(receipt_path.read_text())
    if (receipt["rows"] != decisions or receipt["unchanged_quality_policy"] != manifest["policy"]
            or receipt["sharpness_min"] != manifest["receipt"]["sharpness_min"]
            or receipt["position_priors_added"] != 0 or receipt["extrapolation_used"] is not False):
        raise ValueError("Pose-gap authorization does not reproduce the raw quality decisions")
    return {row["image_name"] for row in candidates}


def bind_supported_priors(model, lookup, allowed_pose_gaps):
    priors, unsupported = [], []
    for image in model.images.values():
        if not image.has_pose:
            continue
        if image.name in lookup:
            if image.name in allowed_pose_gaps:
                raise ValueError("A pose-gap image cannot also have a position prior")
            priors.append({"image_id": image.image_id, "image_name": image.name,
                "C_W_m": lookup[image.name]["C_W_m"],
                "position_covariance_m2": (np.eye(3)*.15**2).tolist()})
        elif image.name in allowed_pose_gaps:
            unsupported.append(image.name)
        else:
            raise ValueError("Registered image has neither a valid raw prior nor a verified pose-gap role: " + image.name)
    return priors, sorted(unsupported)


def projection_validity(model):
    invalid = []
    observation_count = 0
    for image in model.images.values():
        if not image.has_pose:
            continue
        entries = [(index, point) for index, point in enumerate(image.points2D) if point.has_point3D()]
        observation_count += len(entries)
        if not entries:
            continue
        xyz = np.asarray([model.points3D[point.point3D_id].xyz for _, point in entries])
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        finite = np.isfinite(projected).all(axis=1)
        valid = finite & np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0)
        for slot in np.flatnonzero(~valid):
            index, point = entries[slot]
            invalid.append({"image_name": image.name, "point2D_index": index,
                "point3D_id": int(point.point3D_id),
                "depth_m": float(camera_xyz[slot, 2]) if np.isfinite(camera_xyz[slot, 2]) else None,
                "finite_camera_coordinates": bool(np.isfinite(camera_xyz[slot]).all()),
                "finite_projection": bool(finite[slot]), "track_length": model.points3D[point.point3D_id].track.length()})
    return {"valid": not invalid, "observation_count": observation_count, "invalid_count": len(invalid), "invalid": invalid}


def active_prior_names(model, rows):
    return {
        row["image_name"]
        for row in rows
        if any(point.has_point3D() for point in model.images[row["image_id"]].points2D)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-raw-candidates", action="store_true")
    parser.add_argument("--pose-gap-decisions", type=Path)
    parser.add_argument("--threads", type=int, default=8, choices=[2, 8])
    parser.add_argument("--offline-physical-recovery", action="store_true")
    parser.add_argument("--offline-metric-retriangulation", action="store_true")
    parser.add_argument("--retriangulation-database", type=Path)
    parser.add_argument("--retriangulation-images", type=Path)
    args = parser.parse_args()
    if args.offline_physical_recovery and args.offline_metric_retriangulation:
        parser.error("Select exactly one offline metric recovery implementation")
    if args.offline_metric_retriangulation and (
        args.retriangulation_database is None or args.retriangulation_images is None
    ):
        parser.error("Metric retriangulation requires its current-run database and images")
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    helper_path = project/"02_code/box2_lio_guided_sfm_v1_20261002"
    sys.path.insert(0, str(helper_path))
    common = importlib.import_module("common")
    metric_helpers = importlib.import_module("run_prior_ba")
    recovery = None
    recovery_kind = None
    if args.offline_metric_retriangulation:
        sys.path.insert(0, str(project/"02_code/reconstruction_dual_v1_20261009"))
        recovery = importlib.import_module("metric_retriangulation_recovery")
        recovery_kind = "coarse_center_retriangulation_joint_BA"
    elif args.offline_physical_recovery:
        sys.path.insert(0, str(project/"02_code/reconstruction_dual_v1_20261009"))
        recovery = importlib.import_module("metric_geometry_recovery")
        recovery_kind = "legacy_same_track_physical_recovery"
    model = pycolmap.Reconstruction(args.model)
    rows = json.loads((args.initial_frames/"INITIAL_FRAME_MANIFEST.json").read_text())["selected_frames"]
    candidate_source = None
    if args.include_raw_candidates:
        candidate_source = args.initial_frames/"frame_decisions.jsonl"
        rows = [json.loads(line) for line in candidate_source.read_text().splitlines()]
        rows = [row for row in rows if row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
                and row["coarse_pose_valid"] and not row["excluded_role"]]
    lookup = unique_coarse_lookup(rows)
    allowed_pose_gaps = verified_pose_gap_names(args.initial_frames, args.pose_gap_decisions)
    prior_rows, pose_gap_names = bind_supported_priors(model, lookup, allowed_pose_gaps)
    provenance = summarize_used_priors(lookup, prior_rows)
    registered_count = model.num_reg_images()
    common.validate_prior_rows(model, prior_rows)
    names = [row["image_name"] for row in prior_rows]
    centers = np.asarray([row["C_W_m"] for row in prior_rows])
    contract = {"source_model": str(args.model), "source_model_identity": [data_identity(args.model/name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")],
        "raw_frame_manifest": data_identity(args.initial_frames/"INITIAL_FRAME_MANIFEST.json"),
        "additional_raw_candidates": data_identity(candidate_source) if candidate_source else None,
        "input_camera_count": registered_count, "supported_position_prior_count": len(prior_rows),
        "pose_gap_images_without_priors": pose_gap_names,
        "pose_gap_decisions": data_identity(args.pose_gap_decisions) if args.pose_gap_decisions else None,
        "all_registered_cameras_remain_in_visual_BA": True, "CPU_threads": args.threads,
        "sigma_m": .15, "covariance_is_assumed_weight_not_sensor_estimate": True,
        "coarse_pose_producer": provenance["coarse_pose_producer"], "time_offset_verified": False,
        "coarse_prior_provenance": provenance,
        "position_only_priors": True, "joint_iterations": 60, "solver_seconds_per_case": 180,
        "failed_solver_models_preserved_as_unaccepted_diagnostics": True,
        "offline_metric_recovery_protocol": recovery.RECOVERY_PROTOCOL if recovery else None,
        "offline_metric_recovery_kind": recovery_kind,
        "recovery_failure_trigger": "actual invalid projection, metric radius, or coarse-center P95; original failure preserved",
        "prior_loss": "CAUCHY", "prior_loss_scale": 2.7954834829151074,
        "alignment": "new robust Sim3 using this visual map and raw coarse camera centers; no previous map or GT",
        "alignment_threshold_m": .6, "minimum_alignment_inlier_ratio": .2,
        "guard": "same observations and intrinsics; usable solver; exactly 3 residuals per active position prior; common-observation P95 delta versus visual BA <=0.05 px",
        "absolute_accuracy_claim": False, "GT_used": False, "old_metric_Sim3_used": False,
        "counts_as_new_optimization": False,
        "implementation": [data_identity(Path(__file__)), data_identity(helper_path/"common.py"),
                           data_identity(helper_path/"run_prior_ba.py"),
                           data_identity(Path(__file__).with_name("coarse_prior_provenance.py")),
                           data_identity(Path(__file__).with_name("pose_gap_candidates.py"))]}
    atomic_json(args.output/"CONTRACT.json", contract)
    if recovery:
        if recovery_kind == "coarse_center_retriangulation_joint_BA":
            recovery_sources = [data_identity(Path(recovery.__file__)),
                data_identity(project/"02_code/reconstruction_dual_v1_20261009/metric_point_policy.py")]
        else:
            recovery_sources = [data_identity(Path(recovery.__file__)),
                *[data_identity(project/"02_code/reconstruction_dual_v1_20261009"/name)
                  for name in ("native_positive_depth.py", "positive_depth_point_initialization.py", "metric_point_policy.py")]]
        atomic_json(args.output/"PHYSICAL_RECOVERY_IMPLEMENTATION.json", {
            "sources": recovery_sources, "kind": recovery_kind,
            "runtime": "offline_numeric_code_no_LLM_or_VLM_or_network",
            "downstream_point_filter_thresholds_unchanged": True})
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    shutil.copyfile(Path(__file__).with_name("pose_gap_candidates.py"), args.output/"pose_gap_candidates.py")
    shutil.copyfile(Path(__file__).with_name("coarse_prior_provenance.py"), args.output/"coarse_prior_provenance.py")
    started = time.monotonic()
    alignment_options = pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0)
    transform = pycolmap.align_reconstruction_to_locations(model, names, centers, 3, alignment_options)
    if transform is None or not np.isfinite(transform.scale) or transform.scale <= 0:
        atomic_json(args.output/"FAILURE.json", {"status": "alignment_failed_no_metric_claim"})
        raise RuntimeError("Fresh visual map could not be aligned to raw metric positions")
    model.transform(transform)
    initial_validity = projection_validity(model)
    atomic_json(args.output/"INPUT_PROJECTION_VALIDITY.json", initial_validity)
    if not initial_validity["valid"]:
        raise ValueError("The aligned input already has invalid observed projections")
    aligned_error = [np.linalg.norm(model.images[row["image_id"]].projection_center()-row["C_W_m"]) for row in prior_rows]
    atomic_json(args.output/"ALIGNMENT.json", common.serializable({"new_Sim3": transform.todict(),
        "source": CURRENT_ALIGNMENT_SOURCE, "coarse_prior_provenance": provenance,
        "residuals_m": metric_helpers.quantiles(aligned_error),
        "inliers_within_0_6_m": int(np.sum(np.asarray(aligned_error) <= .6)), "old_alignment_reused": False}))
    fixed_identity = observation_identity(model)
    intrinsics = {key: camera.params.copy() for key, camera in model.cameras.items()}
    results = {}
    for label in ["visual_metric_aligned", "weak_position_prior"]:
        local = pycolmap.Reconstruction(model)
        config = common.ba_config(local)
        visual_residuals = int(config.num_residuals(local))
        options = common.ba_options(60)
        options.ceres.solver_options.max_solver_time_in_seconds = 180
        options.ceres.solver_options.num_threads = args.threads
        atomic_json(args.output/"STATE.json", {"status": "running", "phase": label,
            "registered_images": registered_count, "supported_priors": len(prior_rows),
            "visual_only_pose_gap_images": pose_gap_names})
        case_started = time.monotonic()
        if label == "visual_metric_aligned":
            config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
            adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, local)
            prior_count = 0
            provided_prior_count = 0
        else:
            priors = common.make_priors(local, prior_rows, .15)
            provided_prior_count = len(priors)
            prior_count = len(active_prior_names(local, prior_rows))
            prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
            prior_options.prior_position_fallback_stddev = .15
            prior_options.alignment_ransac.max_error = .6
            prior_options.alignment_ransac.min_inlier_ratio = .2
            prior_options.alignment_ransac.random_seed = 0
            prior_options.alignment_ransac.max_num_trials = 10000
            prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
            prior_options.ceres.prior_position_loss_scale = contract["prior_loss_scale"]
            adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(options, prior_options, config, priors, local)
        solution = common.solver_receipt(adjuster.solve())
        actual_residuals = solution["num_residuals"]
        activated = actual_residuals == visual_residuals+3*prior_count and solution["num_residual_blocks"] == visual_residuals//2+prior_count
        folder = args.output/label
        folder.mkdir()
        if not activated or not solution["usable"]:
            atomic_json(folder/"FAILURE.json", common.serializable({"status": "solver_or_activation_failed",
                "solution": solution, "expected_visual_residuals": visual_residuals, "expected_priors": prior_count}))
            raise RuntimeError("Solver did not prove usable optimization with the intended residual factors")
        if fixed_identity != observation_identity(local) or local.num_reg_images() != registered_count:
            raise AssertionError("Common observation identity or registration changed during BA")
        if any(not np.array_equal(camera.params, intrinsics[key]) for key, camera in local.cameras.items()):
            raise AssertionError("Frozen camera calibration changed")
        if not all(np.isfinite(image.cam_from_world().matrix()).all()
                   for image in local.images.values() if image.has_pose):
            raise AssertionError("Bundle adjustment produced a nonfinite registered pose")
        local.update_point_3d_errors()
        destination = folder/"model"
        destination.mkdir()
        local.write(destination)
        validity = projection_validity(local)
        atomic_json(folder/"PROJECTION_VALIDITY.json", validity)
        recovery_result = None
        physical_residuals = 0
        observations_rebuilt = False
        if recovery and label == "weak_position_prior":
            radius = recovery.maximum_camera_radius(local)
            _, point_policy, _, _ = recovery.native_point_policy()
            original_center_errors = [np.linalg.norm(local.images[row["image_id"]].projection_center()-row["C_W_m"])
                for row in prior_rows]
            original_center_metrics = metric_helpers.quantiles(original_center_errors)
            if (not validity["valid"] or radius > point_policy["maximum_camera_radius_m"]
                    or original_center_metrics["p95"] > .6):
                original_failure = {"status": "original_candidate_rejected_known_physical_or_extent_failure",
                    "solver": solution, "model": str(destination), "projection": validity,
                    "maximum_camera_radius_m": radius, "active_priors": prior_count,
                    "provided_position_priors": provided_prior_count,
                    "coarse_prior_consistency_m": original_center_metrics,
                    "factor_count_verified": activated, "model_role": "unaccepted_diagnostic_do_not_train"}
                atomic_json(folder/"ORIGINAL_CANDIDATE_FAILURE.json", common.serializable(original_failure))
                if recovery_kind == "coarse_center_retriangulation_joint_BA":
                    local, recovery_result = recovery.recover_metric_geometry(
                        model, prior_rows, pose_gap_names,
                        args.retriangulation_database, args.retriangulation_images,
                        folder/"offline_metric_retriangulation", common,
                        sys.modules[__name__], metric_helpers.observation_metrics, args.threads)
                    observations_rebuilt = True
                else:
                    reference = results.get("visual_metric_aligned", {})
                    if "metrics" not in reference:
                        raise ValueError("Offline physical recovery requires a valid common-observation visual reference")
                    local, recovery_result = recovery.recover_metric_geometry(model, prior_rows,
                        reference["metrics"]["reprojection_px"]["p95"], folder/"offline_physical_recovery",
                        common, sys.modules[__name__], metric_helpers.observation_metrics, args.threads)
                solution = recovery_result["solver"]
                physical_residuals = recovery_result.get("physical_depth_residuals", 0)
                destination = Path(recovery_result["model"])
                validity = projection_validity(local)
                if observations_rebuilt:
                    visual_residuals = recovery_result["expected_visual_residuals"]
                    provided_prior_count = recovery_result["supported_position_priors"]
                    prior_count = recovery_result["active_position_priors"]
                    activated = recovery_result["factor_count_verified"]
                    if (not activated or recovery_result["old_points_reused"]
                            or not recovery_result["development_guard_passed"]):
                        raise AssertionError("Offline retriangulation recovery lacks its verified metric contract")
                else:
                    activated = (solution["num_residuals"] == visual_residuals+3*prior_count+physical_residuals
                        and solution["num_residual_blocks"] == visual_residuals//2+prior_count+physical_residuals)
                    if not activated or fixed_identity != observation_identity(local):
                        raise AssertionError("Offline recovery did not preserve actual position factors and observations")
        if not validity["valid"]:
            results[label] = {"status": "failed_projection_validation", "solver": solution,
                "factor_count_verified": activated, "active_priors": prior_count,
                "provided_position_priors": provided_prior_count,
                "expected_visual_residuals": visual_residuals, "registered_images": local.num_reg_images(),
                "common_observations_preserved": not observations_rebuilt,
                "observation_set_rebuilt_from_current_run_matches": observations_rebuilt,
                "invalid_projection_count": validity["invalid_count"],
                "model": str(destination), "model_role": "unaccepted_diagnostic_do_not_export_or_train",
                "elapsed_seconds": time.monotonic()-case_started}
            atomic_json(folder/"FAILURE.json", common.serializable(results[label]))
            continue
        metrics = metric_helpers.observation_metrics(local)
        errors = [np.linalg.norm(local.images[row["image_id"]].projection_center()-row["C_W_m"]) for row in prior_rows]
        results[label] = {"solver": solution, "metrics": metrics,
            "registered_images": local.num_reg_images(), "pose_gap_images_without_priors": pose_gap_names,
            "pose_gap_poses_finite": all(np.isfinite(image.cam_from_world().matrix()).all()
                for image in local.images.values() if image.has_pose and image.name in pose_gap_names),
            "active_priors": prior_count, "provided_position_priors": provided_prior_count,
            "expected_visual_residuals": visual_residuals,
            "separate_physical_depth_residuals": physical_residuals,
            "offline_physical_recovery": ({"protocol": recovery_result["protocol"],
                "result": str(folder/"offline_physical_recovery/RESULT.json"),
                "original_candidate_failure_preserved": True,
                "native_complete_metric_coordinate_lifecycle_used": True}
                if recovery_result and not observations_rebuilt else None),
            "offline_metric_retriangulation": ({"protocol": recovery_result["protocol"],
                "result": str(folder/"offline_metric_retriangulation/RESULT.json"),
                "original_candidate_failure_preserved": True,
                "old_points_reused": recovery_result["old_points_reused"],
                "common_observation_p95_delta_px": recovery_result["common_observation_p95_delta_px"],
                "common_observation_comparison_verified": recovery_result["common_observation_comparison_verified"],
                "development_guard_passed": recovery_result["development_guard_passed"]}
                if recovery_result and observations_rebuilt else None),
            "factor_count_verified": activated,
            "common_observations_preserved": not observations_rebuilt,
            "observation_set_rebuilt_from_current_run_matches": observations_rebuilt,
            "coarse_prior_consistency_m": metric_helpers.quantiles(errors),
            "elapsed_seconds": time.monotonic()-case_started, "model": str(destination)}
        atomic_json(folder/"RESULT.json", common.serializable(results[label]))
    if any(case.get("status") == "failed_projection_validation" for case in results.values()):
        failure = {"status": "failed_projection_validation", "results": results,
            "selected": None, "development_guard_passed": False, "counted": False,
            "no_observations_deleted_to_mask_failure": True, "elapsed_seconds": time.monotonic()-started}
        atomic_json(args.output/"RESULT.json", common.serializable(failure))
        atomic_json(args.output/"STATE.json", {"status": failure["status"], "selected": None})
        raise ValueError("One or both BA cases have invalid observed projections; preserve diagnostic models without promotion")
    retriangulation = results["weak_position_prior"].get("offline_metric_retriangulation")
    if retriangulation:
        delta = retriangulation["common_observation_p95_delta_px"]
        development_guard_passed = bool(
            retriangulation["common_observation_comparison_verified"]
            and retriangulation["development_guard_passed"]
        )
        selected = "weak_position_prior"
        comparison_basis = "common retained observations before and after current-run retriangulation BA"
        fallback_boundary = "Retriangulation failure remains fatal; the rejected original weak BA and its failure receipt are preserved"
    else:
        delta = (results["weak_position_prior"]["metrics"]["reprojection_px"]["p95"]-
                 results["visual_metric_aligned"]["metrics"]["reprojection_px"]["p95"])
        development_guard_passed = delta <= .05
        selected = "weak_position_prior" if development_guard_passed else "visual_metric_aligned"
        comparison_basis = "unchanged original observation identity across both BA cases"
        fallback_boundary = "If weak BA fails its guard, retain the separately identified post-hoc metric alignment; do not claim active metric BA success"
    atomic_json(args.output/"RESULT.json", common.serializable({"status": "metric_interface_verified",
        "results": results, "common_observation_p95_delta_px": delta,
        "common_observation_comparison_basis": comparison_basis,
        "development_guard_passed": development_guard_passed,
        "selected": selected, "metric_source": CURRENT_METRIC_SOURCE,
        "coarse_prior_provenance": provenance, "elapsed_seconds": time.monotonic()-started,
        "absolute_accuracy_claim": False, "counted": False,
        "fallback_boundary": fallback_boundary}))
    atomic_json(args.output/"STATE.json", {"status": "completed", "selected": selected,
        "registered_images": registered_count, "supported_priors": len(prior_rows),
        "visual_only_pose_gap_images": pose_gap_names,
        "development_guard_passed": development_guard_passed})
    print(json.dumps({"status": "metric_interface_verified", "selected": selected, "P95_delta_px": delta,
                      "priors": len(prior_rows), "counted": False}), flush=True)


if __name__ == "__main__":
    main()
