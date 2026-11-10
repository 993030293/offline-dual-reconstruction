"""Recover metric geometry by rebuilding points around supported coarse centers."""

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from metric_point_policy import native_point_policy


RECOVERY_PROTOCOL = "coarse_center_retriangulation_joint_BA_v1"
MAXIMUM_PRUNED_POINT_FRACTION = 0.001
MAXIMUM_PRUNED_OBSERVATION_FRACTION = 0.001


def _quantiles(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite metric values")
    return {
        "count": int(values.size),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _copy_database(source, destination):
    source = Path(source)
    destination = Path(destination)
    source_digest = _file_digest(source)
    with closing(
        sqlite3.connect("file:" + source.as_posix() + "?mode=ro", uri=True)
    ) as incoming:
        if incoming.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The metric recovery database failed integrity validation")
        database_names = {
            name for name, in incoming.execute("SELECT name FROM images")
        }
        with closing(sqlite3.connect(destination)) as owned:
            incoming.backup(owned)
    if _file_digest(source) != source_digest:
        raise RuntimeError("The source metric recovery database changed during its copy")
    return source_digest, database_names


def _registered(model):
    return {image.name: image for image in model.images.values() if image.has_pose}


def _observation_names(model):
    return {
        image.name
        for image in model.images.values()
        if image.has_pose and any(point.has_point3D() for point in image.points2D)
    }


def _projection_audit(model):
    errors = {}
    invalid = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        pose = image.cam_from_world()
        camera = model.cameras[image.camera_id]
        for index, point in enumerate(image.points2D):
            if not point.has_point3D():
                continue
            xyz = model.points3D[point.point3D_id].xyz
            camera_xyz = pose.rotation.matrix() @ xyz + pose.translation
            projected = camera.img_from_cam(camera_xyz)
            valid = (
                np.isfinite(camera_xyz).all()
                and camera_xyz[2] > 0
                and projected is not None
                and np.isfinite(projected).all()
            )
            key = (int(image.image_id), index)
            if valid:
                errors[key] = float(np.linalg.norm(projected - point.xy))
            else:
                invalid.append(
                    {
                        "image_name": image.name,
                        "point2D_index": index,
                        "point3D_id": int(point.point3D_id),
                        "depth_m": float(camera_xyz[2])
                        if np.isfinite(camera_xyz[2])
                        else None,
                        "track_length": model.points3D[point.point3D_id].track.length(),
                    }
                )
    return {
        "observation_count": model.compute_num_observations(),
        "valid_observation_count": len(errors),
        "invalid_count": len(invalid),
        "invalid": invalid,
        "errors": errors,
        "reprojection_px": _quantiles(errors.values()),
    }


def _public_audit(audit):
    return {key: value for key, value in audit.items() if key != "errors"}


def _camera_radii(model):
    values = [
        np.linalg.norm(image.projection_center())
        for image in model.images.values()
        if image.has_pose
    ]
    return _quantiles(values)


def maximum_camera_radius(model):
    return _camera_radii(model)["maximum"]


def _triangulation_options(threads):
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = threads
    options.random_seed = 0
    options.fix_existing_frames = True
    options.mapper.fix_existing_frames = True
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_refine_sensor_from_rig = False
    options.extract_colors = False
    options.max_runtime_seconds = 280
    options.ba_global_max_num_iterations = 60
    options.ba_global_max_refinements = 1
    if not options.check():
        raise ValueError("The installed point triangulator rejects the recovery options")
    return options


def _replace_supported_centers(model, rows_by_name):
    registered = _registered(model)
    for name, row in rows_by_name.items():
        image = registered[name]
        pose = image.cam_from_world()
        center = np.asarray(row["C_W_m"], dtype=np.float64)
        translation = -(pose.rotation.matrix() @ center)
        image.frame.set_cam_from_world(
            image.camera_id, pycolmap.Rigid3d(pose.rotation, translation)
        )
    errors = [
        np.linalg.norm(registered[name].projection_center() - rows_by_name[name]["C_W_m"])
        for name in sorted(rows_by_name)
    ]
    if max(errors) > 1e-9:
        raise ValueError("A replaced metric camera center differs from its coarse input")
    return _quantiles(errors)


def _factor_count_verified(solution, visual_residuals, active_prior_count):
    return bool(
        solution["num_residuals"] == visual_residuals + 3 * active_prior_count
        and solution["num_residual_blocks"]
        == visual_residuals // 2 + active_prior_count
    )


def _prune_invalid_tracks(model, audit):
    invalid_ids = sorted({row["point3D_id"] for row in audit["invalid"]})
    points_before = model.num_points3D()
    observations_before = model.compute_num_observations()
    removed_observations = sum(
        model.points3D[point_id].track.length() for point_id in invalid_ids
    )
    point_fraction = len(invalid_ids) / points_before if points_before else 1.0
    observation_fraction = (
        removed_observations / observations_before if observations_before else 1.0
    )
    receipt = {
        "invalid_point_ids": invalid_ids,
        "removed_points": len(invalid_ids),
        "removed_observations": removed_observations,
        "removed_point_fraction": point_fraction,
        "removed_observation_fraction": observation_fraction,
        "maximum_pruned_point_fraction": MAXIMUM_PRUNED_POINT_FRACTION,
        "maximum_pruned_observation_fraction": MAXIMUM_PRUNED_OBSERVATION_FRACTION,
        "all_removed_points_had_an_actual_invalid_projection": True,
    }
    if (
        point_fraction > MAXIMUM_PRUNED_POINT_FRACTION
        or observation_fraction > MAXIMUM_PRUNED_OBSERVATION_FRACTION
    ):
        raise ValueError("Invalid-track pruning exceeds the bounded recovery fractions")
    for point_id in invalid_ids:
        model.delete_point3D(point_id)
    return receipt


def recover_metric_geometry(
    source,
    prior_rows,
    pose_gap_names,
    database,
    images,
    output,
    helpers,
    checks,
    metrics_fn,
    threads,
):
    """Return a metric model rebuilt only from current-run matches and priors."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase = "validating_inputs"
    database = Path(database).resolve(strict=True)
    images = Path(images).resolve(strict=True)
    policy_path, point_policy, _, _ = native_point_policy()
    source_registered = _registered(source)
    source_names = set(source_registered)
    rows_by_name = {row["image_name"]: row for row in prior_rows}
    if len(rows_by_name) != len(prior_rows):
        raise ValueError("Metric recovery position priors contain duplicate image names")
    supported_names = set(rows_by_name)
    unsupported_names = source_names - supported_names
    if supported_names - source_names or unsupported_names != set(pose_gap_names):
        raise ValueError("Metric recovery camera roles differ from the verified prior contract")
    if not supported_names:
        raise ValueError("Metric recovery requires at least one supported coarse center")
    owned_database = output / "retriangulation_database.db"
    database_digest, database_names = _copy_database(database, owned_database)
    owned_database_digest_before = _file_digest(owned_database)
    if not source_names <= database_names:
        raise ValueError("The metric recovery database lacks a registered source image")
    if any(not (images / name).is_file() for name in source_names):
        raise FileNotFoundError("A registered metric recovery image is absent")
    source_observations = checks.observation_identity(source)
    source_intrinsics = {
        key: camera.params.copy() for key, camera in source.cameras.items()
    }
    contract = {
        "protocol": RECOVERY_PROTOCOL,
        "runtime": "offline_numeric_code_no_LLM_or_VLM_or_network",
        "registered_images": len(source_names),
        "supported_position_priors": len(prior_rows),
        "verified_pose_gap_images_without_priors": sorted(unsupported_names),
        "database": str(database),
        "owned_retriangulation_database": str(owned_database),
        "images": str(images),
        "database_sha256": database_digest,
        "owned_database_sha256_before_native_triangulation": owned_database_digest_before,
        "coarse_center_role": "metric camera-center initialization and active BA prior",
        "visual_rotation_role": "robustly aligned source rotation initialization",
        "point_role": "all source points discarded and retriangulated from current-run matches",
        "fixed_intrinsics": True,
        "prior_sigma_m": 0.15,
        "prior_loss": "CAUCHY",
        "joint_iterations": 60,
        "maximum_solver_seconds": 180,
        "maximum_pruned_point_fraction": MAXIMUM_PRUNED_POINT_FRACTION,
        "maximum_pruned_observation_fraction": MAXIMUM_PRUNED_OBSERVATION_FRACTION,
        "metric_camera_radius_guard_m": point_policy["maximum_camera_radius_m"],
        "downstream_point_reprojection_filter_px": point_policy[
            "maximum_point_reprojection_error_px"
        ],
        "downstream_point_filter_unchanged": True,
        "object_class_or_box_geometry_used": False,
        "GT_used": False,
    }
    checks.atomic_json(output / "CONTRACT.json", helpers.serializable(contract))
    try:
        phase = "coarse_center_initialization"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        metric_seed = pycolmap.Reconstruction(source)
        supported_order = sorted(supported_names)
        target_centers = np.asarray(
            [rows_by_name[name]["C_W_m"] for name in supported_order],
            dtype=np.float64,
        )
        transform = pycolmap.align_reconstruction_to_locations(
            metric_seed,
            supported_order,
            target_centers,
            3,
            pycolmap.RANSACOptions(
                max_error=0.6,
                min_inlier_ratio=0.2,
                max_num_trials=10000,
                random_seed=0,
            ),
        )
        if transform is None or not np.isfinite(transform.scale) or transform.scale <= 0:
            raise RuntimeError("The source visual model could not be robustly metric-aligned")
        metric_seed.transform(transform)
        aligned_errors = [
            np.linalg.norm(
                _registered(metric_seed)[name].projection_center()
                - rows_by_name[name]["C_W_m"]
            )
            for name in supported_order
        ]
        replacement = _replace_supported_centers(metric_seed, rows_by_name)
        seed_model = output / "metric_pose_seed"
        seed_model.mkdir()
        metric_seed.write(seed_model)

        phase = "fixed_pose_retriangulation"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        triangulated_path = output / "retriangulated_model"
        triangulated_path.mkdir()
        triangulation_started = time.monotonic()
        pycolmap.triangulate_points(
            metric_seed,
            owned_database,
            images,
            triangulated_path,
            clear_points=True,
            options=_triangulation_options(threads),
            refine_intrinsics=False,
        )
        triangulation_seconds = time.monotonic() - triangulation_started
        reference = pycolmap.Reconstruction(triangulated_path)
        if set(_registered(reference)) != source_names:
            raise ValueError("Fixed-pose retriangulation changed the registered camera denominator")
        if any(
            not np.array_equal(camera.params, source_intrinsics[key])
            for key, camera in reference.cameras.items()
        ):
            raise ValueError("Fixed-pose retriangulation changed camera calibration")
        reference_center_errors = [
            np.linalg.norm(
                _registered(reference)[name].projection_center()
                - rows_by_name[name]["C_W_m"]
            )
            for name in supported_order
        ]
        if max(reference_center_errors) > 1e-9:
            raise ValueError("Fixed-pose retriangulation changed a coarse metric center")
        reference_audit = _projection_audit(reference)
        if reference_audit["invalid_count"]:
            raise ValueError("Fixed-pose retriangulation produced invalid observed projections")

        phase = "joint_metric_bundle_adjustment"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        model = pycolmap.Reconstruction(reference)
        before_ba_identity = checks.observation_identity(model)
        config = helpers.ba_config(model)
        visual_residuals = int(config.num_residuals(model))
        options = helpers.ba_options(60)
        options.ceres.solver_options.max_solver_time_in_seconds = 180
        options.ceres.solver_options.num_threads = threads
        priors = helpers.make_priors(model, prior_rows, 0.15)
        active_prior_names = supported_names & _observation_names(model)
        prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
        prior_options.prior_position_fallback_stddev = 0.15
        prior_options.alignment_ransac = pycolmap.RANSACOptions(
            max_error=0.6,
            min_inlier_ratio=0.2,
            max_num_trials=10000,
            random_seed=0,
        )
        prior_options.ceres.prior_position_loss_function_type = (
            pycolmap.LossFunctionType.CAUCHY
        )
        prior_options.ceres.prior_position_loss_scale = 2.7954834829151074
        adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(
            options, prior_options, config, priors, model
        )
        solve_started = time.monotonic()
        native = adjuster.solve()
        solve_seconds = time.monotonic() - solve_started
        solution = helpers.solver_receipt(native)
        factors = _factor_count_verified(
            solution, visual_residuals, len(active_prior_names)
        )
        if not solution["usable"] or not factors:
            raise ValueError("Retriangulated metric BA lacks usable verified factors")
        if checks.observation_identity(model) != before_ba_identity:
            raise ValueError("Joint metric BA changed the retriangulated observation identity")
        if set(_registered(model)) != source_names:
            raise ValueError("Joint metric BA changed the registered camera denominator")
        if any(
            not np.array_equal(camera.params, source_intrinsics[key])
            for key, camera in model.cameras.items()
        ):
            raise ValueError("Joint metric BA changed fixed camera calibration")
        preprune = output / "post_ba_unaccepted_model"
        preprune.mkdir()
        model.write(preprune)
        preprune_audit = _projection_audit(model)
        pruning = _prune_invalid_tracks(model, preprune_audit)
        model.update_point_3d_errors()
        final_audit = _projection_audit(model)
        if final_audit["invalid_count"]:
            raise ValueError("Bounded invalid-track pruning did not restore valid projections")

        reference_errors = reference_audit["errors"]
        final_errors = final_audit["errors"]
        common_keys = sorted(set(reference_errors) & set(final_errors))
        if len(common_keys) != final_audit["observation_count"]:
            raise ValueError("The retained common-observation comparison is incomplete")
        reference_common = _quantiles(reference_errors[key] for key in common_keys)
        final_common = _quantiles(final_errors[key] for key in common_keys)
        common_delta = final_common["p95"] - reference_common["p95"]
        center_errors = [
            np.linalg.norm(
                _registered(model)[name].projection_center()
                - rows_by_name[name]["C_W_m"]
            )
            for name in supported_order
        ]
        center_metrics = _quantiles(center_errors)
        radius_metrics = _camera_radii(model)
        source_database_unchanged = _file_digest(database) == database_digest
        owned_database_digest_after = _file_digest(owned_database)
        source_geometry_unchanged = checks.observation_identity(source) == source_observations
        passed = bool(
            common_delta <= 0.05
            and center_metrics["p95"] <= 0.6
            and radius_metrics["maximum"] <= point_policy["maximum_camera_radius_m"]
            and source_database_unchanged
            and source_geometry_unchanged
        )
        candidate_role = "accepted_model" if passed else "unaccepted_model"
        destination = output / candidate_role
        destination.mkdir()
        model.write(destination)
        result = {
            "status": "completed_metric_retriangulation_recovery"
            if passed
            else "failed_metric_retriangulation_recovery_guards",
            "protocol": RECOVERY_PROTOCOL,
            "model": str(destination),
            "registered_images": model.num_reg_images(),
            "supported_position_priors": len(prior_rows),
            "active_position_priors": len(active_prior_names),
            "zero_observation_supported_images": sorted(
                supported_names - active_prior_names
            ),
            "verified_pose_gap_images_without_priors": sorted(unsupported_names),
            "source_points_discarded": source.num_points3D(),
            "source_observations_discarded": source.compute_num_observations(),
            "retriangulated_points": reference.num_points3D(),
            "retriangulated_observations": reference.compute_num_observations(),
            "final_points": model.num_points3D(),
            "final_observations": model.compute_num_observations(),
            "alignment_residuals_before_center_replacement_m": _quantiles(
                aligned_errors
            ),
            "center_replacement_readback_m": replacement,
            "retriangulated_input_metrics": _public_audit(reference_audit),
            "post_ba_preprune_metrics": _public_audit(preprune_audit),
            "pruning": pruning,
            "metrics": metrics_fn(model),
            "final_projection": _public_audit(final_audit),
            "coarse_prior_consistency_m": center_metrics,
            "camera_radius_m": radius_metrics,
            "common_observation_reference_p95_px": reference_common["p95"],
            "common_observation_final_p95_px": final_common["p95"],
            "common_observation_p95_delta_px": common_delta,
            "common_observation_count": len(common_keys),
            "common_observation_comparison_verified": True,
            "solver": solution,
            "expected_visual_residuals": visual_residuals,
            "factor_count_verified": factors,
            "fixed_intrinsics": True,
            "old_points_reused": False,
            "source_database_unchanged": source_database_unchanged,
            "source_geometry_unchanged": source_geometry_unchanged,
            "owned_database_sha256_before_native_triangulation": owned_database_digest_before,
            "owned_database_sha256_after_native_triangulation": owned_database_digest_after,
            "owned_database_modified_by_native_triangulation": (
                owned_database_digest_after != owned_database_digest_before
            ),
            "downstream_point_reprojection_filter_px": point_policy[
                "maximum_point_reprojection_error_px"
            ],
            "downstream_point_filter_unchanged": True,
            "development_guard_passed": passed,
            "triangulation_seconds": triangulation_seconds,
            "solver_seconds": solve_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "runtime_language_model_used": False,
            "manual_geometry_adoption_used": False,
            "GT_used": False,
            "counted_as_full_cold_delivery": False,
        }
        checks.atomic_json(output / "RESULT.json", helpers.serializable(result))
        checks.atomic_json(
            output / "STATE.json",
            {"status": result["status"], "phase": phase},
        )
        if not passed:
            raise ValueError("Retriangulated metric recovery failed its admission guards")
        return model, result
    except BaseException:
        failure = {
            "status": "failed_metric_retriangulation_recovery",
            "phase": phase,
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
            "source_failure_not_relabelled": True,
            "counted_as_full_cold_delivery": False,
        }
        checks.atomic_json(output / "FAILURE.json", failure)
        checks.atomic_json(
            output / "STATE.json",
            {"status": failure["status"], "phase": phase},
        )
        raise
