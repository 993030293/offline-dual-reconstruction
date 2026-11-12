"""Recover metric gauge per strongly supported visual-track component."""

from collections import deque
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pycolmap

from metric_point_policy import native_point_policy


RECOVERY_PROTOCOL = "visual_track_component_metric_gauge_v1"
ALIGNMENT_MAXIMUM_ERROR_M = 0.6
ALIGNMENT_MINIMUM_INLIER_RATIO = 0.2
REPROJECTION_INVARIANCE_TOLERANCE_PX = 1e-6


def _quantiles(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite component metric values")
    return {
        "count": int(values.size),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _registered(model):
    return {
        image.name: image for image in model.images.values() if image.has_pose
    }


def _load_policy():
    policy_path, point_policy, _, _ = native_point_policy()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return (
        policy_path,
        point_policy,
        policy["camera_geometry_admission"],
        policy["metric_time_support"],
    )


def _track_graph(model, registered):
    ordered = sorted(registered.values(), key=lambda image: image.name)
    names = [image.name for image in ordered]
    index_by_image_id = {
        int(image.image_id): index for index, image in enumerate(ordered)
    }
    weights = np.zeros((len(names), len(names)), dtype=np.uint32)
    point_indices = {}
    for point_id, point in model.points3D.items():
        indices = sorted(
            {
                index_by_image_id[int(element.image_id)]
                for element in point.track.elements
                if int(element.image_id) in index_by_image_id
            }
        )
        point_indices[int(point_id)] = indices
        if len(indices) >= 2:
            weights[np.ix_(indices, indices)] += 1
    np.fill_diagonal(weights, 0)
    return names, weights, point_indices


def _connected_components(weights, minimum_shared_tracks):
    unseen = np.ones(weights.shape[0], dtype=bool)
    components = []
    for seed in range(weights.shape[0]):
        if not unseen[seed]:
            continue
        unseen[seed] = False
        queue = deque([seed])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            neighbors = np.flatnonzero(
                unseen & (weights[node] >= minimum_shared_tracks)
            )
            if neighbors.size:
                unseen[neighbors] = False
                queue.extend(int(value) for value in neighbors)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def _transformed_center(transform, center):
    return (
        transform.scale * (transform.rotation.matrix() @ np.asarray(center))
        + transform.translation
    )


def _estimate_component_alignment(
    model,
    names,
    rows_by_name,
    minimum_priors,
    maximum_camera_radius_m,
):
    supported = sorted(name for name in names if name in rows_by_name)
    base = {
        "camera_count": len(names),
        "prior_count": len(supported),
        "minimum_required_priors": minimum_priors,
    }
    if len(supported) < minimum_priors:
        return None, {
            **base,
            "status": "excluded_insufficient_component_priors",
        }
    transform = pycolmap.align_reconstruction_to_locations(
        model,
        supported,
        np.asarray([rows_by_name[name]["C_W_m"] for name in supported]),
        3,
        pycolmap.RANSACOptions(
            max_error=ALIGNMENT_MAXIMUM_ERROR_M,
            min_inlier_ratio=ALIGNMENT_MINIMUM_INLIER_RATIO,
            max_num_trials=10000,
            random_seed=0,
        ),
    )
    if transform is None or not np.isfinite(transform.scale) or transform.scale <= 0:
        return None, {**base, "status": "excluded_failed_component_alignment"}
    registered = _registered(model)
    centers = {
        name: _transformed_center(
            transform, registered[name].projection_center()
        )
        for name in names
    }
    residuals = [
        np.linalg.norm(centers[name] - rows_by_name[name]["C_W_m"])
        for name in supported
    ]
    radii = [np.linalg.norm(center) for center in centers.values()]
    inlier_count = sum(value <= ALIGNMENT_MAXIMUM_ERROR_M for value in residuals)
    row = {
        **base,
        "status": "retained_component_metric_alignment",
        "scale": float(transform.scale),
        "inlier_count_at_0_6m": inlier_count,
        "inlier_fraction_at_0_6m": inlier_count / len(supported),
        "coarse_center_residual_m": _quantiles(residuals),
        "transformed_camera_radius_m": _quantiles(radii),
    }
    if (inlier_count < minimum_priors
            or inlier_count / len(supported) < ALIGNMENT_MINIMUM_INLIER_RATIO):
        row["status"] = "excluded_insufficient_alignment_inliers"
        return None, row
    if row["transformed_camera_radius_m"]["maximum"] > maximum_camera_radius_m:
        row["status"] = "excluded_component_metric_extent"
        return None, row
    return transform, row


def _reprojection_errors(model):
    chunks = []
    invalid_count = 0
    for image in sorted(model.images.values(), key=lambda value: value.image_id):
        if not image.has_pose:
            continue
        points = [point for point in image.points2D if point.has_point3D()]
        if not points:
            continue
        xyz = np.asarray([model.points3D[point.point3D_id].xyz for point in points])
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        valid = (
            np.isfinite(camera_xyz).all(axis=1)
            & (camera_xyz[:, 2] > 0)
            & np.isfinite(projected).all(axis=1)
        )
        invalid_count += int(np.sum(~valid))
        observations = np.asarray([point.xy for point in points], dtype=np.float64)
        chunks.append(
            np.linalg.norm(projected[valid] - observations[valid], axis=1)
        )
    values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return values, invalid_count


def _transform_component(model, names, point_ids, transform):
    rotation = transform.rotation.matrix()
    for point_id in point_ids:
        point = model.points3D[point_id]
        point.xyz = (
            transform.scale * (rotation @ point.xyz) + transform.translation
        )
    registered = _registered(model)
    for name in names:
        image = registered[name]
        source_pose = image.cam_from_world()
        center = _transformed_center(transform, image.projection_center())
        camera_rotation = source_pose.rotation.matrix() @ rotation.T
        image.frame.set_cam_from_world(
            image.camera_id,
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(camera_rotation), -camera_rotation @ center
            ),
        )


def maximum_camera_radius(model):
    radii = [
        np.linalg.norm(image.projection_center())
        for image in model.images.values()
        if image.has_pose
    ]
    return _quantiles(radii)["maximum"]


def _source_geometry_matches(model, poses, points):
    registered = _registered(model)
    return bool(
        set(registered) == set(poses)
        and set(model.points3D) == set(points)
        and all(
            np.array_equal(registered[name].cam_from_world().matrix(), pose)
            for name, pose in poses.items()
        )
        and all(
            np.array_equal(model.points3D[point_id].xyz, xyz)
            for point_id, xyz in points.items()
        )
    )


def recover_metric_geometry(
    source,
    prior_rows,
    pose_gap_names,
    output,
    helpers,
    checks,
    source_role="post_native_visual_BA_geometry",
):
    """Metric-gauge supported components while retaining every registered camera."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase = "validating_inputs"
    source_observation_identity = checks.observation_identity(source)
    source_intrinsics = {
        key: camera.params.copy() for key, camera in source.cameras.items()
    }
    source_registered = _registered(source)
    source_poses = {
        name: image.cam_from_world().matrix().copy()
        for name, image in source_registered.items()
    }
    source_points = {
        int(point_id): point.xyz.copy()
        for point_id, point in source.points3D.items()
    }
    allowed_source_roles = {
        "post_native_visual_BA_geometry",
        "verified_aligned_input_before_metric_BA",
    }
    if source_role not in allowed_source_roles:
        raise ValueError("Component metric alignment received an unverified visual source role")
    rows_by_name = {row["image_name"]: row for row in prior_rows}
    if len(rows_by_name) != len(prior_rows):
        raise ValueError("Component metric priors contain duplicate image names")
    if set(rows_by_name) - set(source_registered):
        raise ValueError("A component metric prior does not name a registered camera")
    unsupported = set(source_registered) - set(rows_by_name)
    if unsupported != set(pose_gap_names):
        raise ValueError("Component metric camera roles differ from the verified pose-gap contract")
    policy_path, point_policy, camera_policy, time_policy = _load_policy()
    minimum_shared_tracks = int(
        pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
    )
    minimum_component_priors = int(time_policy["minimum_supported_frames"])
    full_camera_count = len(source_registered)
    contract = {
        "protocol": RECOVERY_PROTOCOL,
        "runtime": "offline_numeric_code_no_LLM_or_VLM_or_network",
        "source_role": source_role,
        "minimum_shared_tracks_for_component_edge": minimum_shared_tracks,
        "component_edge_threshold_source": (
            "COLMAP IncrementalMapperOptions.abs_pose_min_num_inliers"
        ),
        "minimum_component_priors": minimum_component_priors,
        "component_prior_threshold_source": (
            "existing metric_time_support.minimum_supported_frames"
        ),
        "alignment_maximum_error_m": ALIGNMENT_MAXIMUM_ERROR_M,
        "alignment_minimum_inlier_ratio": ALIGNMENT_MINIMUM_INLIER_RATIO,
        "minimum_retained_camera_fraction": camera_policy[
            "minimum_retained_fraction_of_sim3_inliers"
        ],
        "maximum_camera_radius_m": point_policy["maximum_camera_radius_m"],
        "minimum_retained_sparse_points": point_policy[
            "minimum_retained_sparse_points"
        ],
        "reprojection_invariance_tolerance_px": (
            REPROJECTION_INVARIANCE_TOLERANCE_PX
        ),
        "cross_component_tracks_removed_before_independent_gauge_transform": True,
        "coarse_position_role": "component metric gauge estimation only",
        "camera_and_point_relative_geometry_role": "native visual BA authority",
        "position_prior_BA_factors_claimed": False,
        "thresholds_relaxed": False,
        "object_class_or_box_geometry_used": False,
        "GT_used": False,
        "policy_source": str(policy_path),
    }
    checks.atomic_json(output / "CONTRACT.json", helpers.serializable(contract))
    try:
        phase = "building_visual_track_components"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        names, weights, point_indices = _track_graph(source, source_registered)
        components = _connected_components(weights, minimum_shared_tracks)
        membership = np.empty(len(names), dtype=np.int32)
        for component_index, component in enumerate(components):
            membership[component] = component_index

        phase = "estimating_component_metric_gauges"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        transforms = {}
        component_rows = []
        retained_components = set()
        for component_index, component in enumerate(components):
            component_names = [names[index] for index in component]
            transform, row = _estimate_component_alignment(
                source,
                component_names,
                rows_by_name,
                minimum_component_priors,
                point_policy["maximum_camera_radius_m"],
            )
            row["component_index"] = component_index
            row["names_head"] = component_names[:10]
            row["camera_registration_role"] = (
                "metric_gauged_training_candidate"
                if transform is not None
                else "retained_nontraining_witness"
            )
            component_rows.append(row)
            if transform is not None:
                transforms[component_index] = transform
                retained_components.add(component_index)
        if not retained_components:
            raise ValueError("No visual-track component has sufficient metric support")

        point_component = {}
        cross_component_points = set()
        excluded_component_points = set()
        for point_id, indices in point_indices.items():
            component_ids = {int(membership[index]) for index in indices}
            if len(component_ids) != 1:
                cross_component_points.add(point_id)
                continue
            component_index = next(iter(component_ids))
            point_component[point_id] = component_index
            if component_index not in retained_components:
                excluded_component_points.add(point_id)

        metric_gauged_names = {
            names[index]
            for component_index, component in enumerate(components)
            if component_index in retained_components
            for index in component
        }
        witness_names = set(source_registered) - metric_gauged_names

        phase = "isolating_and_transforming_components"
        checks.atomic_json(output / "STATE.json", {"status": "running", "phase": phase})
        model = pycolmap.Reconstruction(source)
        for point_id in sorted(cross_component_points | excluded_component_points):
            model.delete_point3D(point_id)
        if set(_registered(model)) != set(source_registered):
            raise ValueError("Component isolation changed the full registered camera denominator")
        witness_observations = {
            name: int(_registered(model)[name].num_points3D)
            for name in sorted(witness_names)
        }
        if any(witness_observations.values()):
            raise ValueError("A nontraining component witness retained a three-dimensional observation")
        retained_observation_identity = checks.observation_identity(model)
        before_errors, before_invalid = _reprojection_errors(model)
        for component_index in sorted(retained_components):
            component_names = [names[index] for index in components[component_index]]
            retained_point_ids = [
                point_id
                for point_id, assigned in point_component.items()
                if assigned == component_index and point_id in model.points3D
            ]
            _transform_component(
                model,
                component_names,
                retained_point_ids,
                transforms[component_index],
            )
        model.update_point_3d_errors()
        after_errors, after_invalid = _reprojection_errors(model)
        if before_errors.shape != after_errors.shape:
            raise ValueError("Component metric alignment changed the retained observation count")
        if checks.observation_identity(model) != retained_observation_identity:
            raise ValueError("Component metric alignment changed retained observation identities")
        maximum_reprojection_delta = float(
            np.max(np.abs(before_errors - after_errors))
        ) if before_errors.size else 0.0
        metric_gauged_fraction = len(metric_gauged_names) / full_camera_count
        registered_after = _registered(model)
        full_map_radii = {
            name: np.linalg.norm(image.projection_center())
            for name, image in registered_after.items()
        }
        metric_gauged_radii = [full_map_radii[name] for name in sorted(metric_gauged_names)]
        witness_radii = [full_map_radii[name] for name in sorted(witness_names)]
        retained_registered = _registered(model)
        retained_prior_residuals = [
            np.linalg.norm(
                retained_registered[name].projection_center()
                - rows_by_name[name]["C_W_m"]
            )
            for name in sorted(metric_gauged_names & set(rows_by_name))
        ]
        full_map_prior_residuals = [
            np.linalg.norm(
                retained_registered[name].projection_center()
                - rows_by_name[name]["C_W_m"]
            )
            for name in sorted(set(retained_registered) & set(rows_by_name))
        ]
        if before_invalid or after_invalid:
            raise ValueError("Component metric alignment retained an invalid projection")
        if maximum_reprojection_delta > REPROJECTION_INVARIANCE_TOLERANCE_PX:
            raise ValueError("Component metric alignment changed retained reprojections")
        if metric_gauged_fraction < camera_policy["minimum_retained_fraction_of_sim3_inliers"]:
            raise ValueError("Component metric alignment fails the training-camera retention bound")
        if model.num_reg_images() != full_camera_count:
            raise ValueError("Component metric alignment lost a registered full-map camera")
        if model.num_points3D() < point_policy["minimum_retained_sparse_points"]:
            raise ValueError("Component metric alignment leaves too few sparse points")
        if max(metric_gauged_radii) > point_policy["maximum_camera_radius_m"]:
            raise ValueError("A metric-gauged training candidate exceeds the camera radius guard")
        if checks.observation_identity(source) != source_observation_identity:
            raise ValueError("Component metric alignment changed its source observations")
        if not _source_geometry_matches(source, source_poses, source_points):
            raise ValueError("Component metric alignment changed its source geometry")
        if any(
            not np.array_equal(camera.params, source_intrinsics[key])
            for key, camera in source.cameras.items()
        ):
            raise ValueError("Component metric alignment changed its source calibration")
        if any(
            not np.array_equal(camera.params, source_intrinsics[key])
            for key, camera in model.cameras.items()
        ):
            raise ValueError("Component metric alignment changed output calibration")

        phase = "publishing_candidate"
        accepted = output / "accepted_model"
        accepted.mkdir()
        model.write(accepted)
        metrics = {
            "reprojection_px": _quantiles(after_errors),
            "observations": model.compute_num_observations(),
            "negative_depth_observations": 0,
            "registered_images": model.num_reg_images(),
            "points": model.num_points3D(),
        }
        result = {
            "status": "completed_component_metric_guards",
            "protocol": RECOVERY_PROTOCOL,
            "model": str(accepted),
            "component_count": len(components),
            "component_rows": component_rows,
            "retained_component_indices": sorted(retained_components),
            "source_registered_cameras": full_camera_count,
            "retained_registered_cameras": model.num_reg_images(),
            "metric_gauged_registered_cameras": len(metric_gauged_names),
            "retained_camera_fraction": metric_gauged_fraction,
            "output_registered_camera_fraction": model.num_reg_images() / full_camera_count,
            "dropped_camera_names": [],
            "metric_gauge_excluded_camera_names": sorted(witness_names),
            "nontraining_witness_camera_names": sorted(witness_names),
            "nontraining_witness_observations": witness_observations,
            "full_registered_denominator_preserved": True,
            "supported_position_priors": len(prior_rows),
            "retained_position_priors": len(retained_prior_residuals),
            "position_prior_BA_factors": 0,
            "position_prior_factor_count_claimed": False,
            "coarse_prior_consistency_m": _quantiles(retained_prior_residuals),
            "full_map_coarse_prior_consistency_m": _quantiles(full_map_prior_residuals),
            "removed_cross_component_points": len(cross_component_points),
            "removed_excluded_component_points": len(excluded_component_points),
            "retained_points": model.num_points3D(),
            "retained_observations": model.compute_num_observations(),
            "expected_visual_residuals": 2 * model.compute_num_observations(),
            "metrics": metrics,
            "common_observation_p95_delta_px": (
                metrics["reprojection_px"]["p95"]
                - _quantiles(before_errors)["p95"]
            ),
            "maximum_reprojection_error_delta_px": maximum_reprojection_delta,
            "retained_observation_identity_preserved": (
                checks.observation_identity(model) == retained_observation_identity
            ),
            "new_observations_invented": False,
            "invalid_projection_count": after_invalid,
            "camera_radius_m": _quantiles(metric_gauged_radii),
            "full_map_camera_radius_m": _quantiles(full_map_radii.values()),
            "nontraining_witness_camera_radius_m": (
                _quantiles(witness_radii) if witness_radii else None
            ),
            "camera_radius_guard_scope": "metric_gauged_training_candidates_only",
            "nontraining_witnesses_may_exceed_training_extent": bool(
                witness_radii
                and max(witness_radii) > point_policy["maximum_camera_radius_m"]
            ),
            "development_guard_passed": True,
            "source_visual_geometry_authority": source_role,
            "source_visual_BA_required_and_retained_as_authority": (
                source_role == "post_native_visual_BA_geometry"
            ),
            "verified_aligned_input_required_and_retained_as_authority": (
                source_role == "verified_aligned_input_before_metric_BA"
            ),
            "source_geometry_unchanged": _source_geometry_matches(
                source, source_poses, source_points
            ),
            "fixed_intrinsics": True,
            "thresholds_relaxed": False,
            "absolute_accuracy_claim": False,
            "GT_used": False,
            "elapsed_seconds": time.monotonic() - started,
            "counted_as_full_cold_delivery": False,
        }
        checks.atomic_json(output / "RESULT.json", helpers.serializable(result))
        checks.atomic_json(
            output / "STATE.json",
            {"status": result["status"], "phase": phase},
        )
        return model, result
    except BaseException:
        failure = {
            "status": "failed_component_metric_alignment",
            "phase": phase,
            "traceback": traceback.format_exc(),
            "source_geometry_unchanged": _source_geometry_matches(
                source, source_poses, source_points
            ),
            "elapsed_seconds": time.monotonic() - started,
            "counted_as_full_cold_delivery": False,
        }
        checks.atomic_json(output / "FAILURE.json", helpers.serializable(failure))
        checks.atomic_json(
            output / "STATE.json",
            {"status": failure["status"], "phase": phase},
        )
        raise
