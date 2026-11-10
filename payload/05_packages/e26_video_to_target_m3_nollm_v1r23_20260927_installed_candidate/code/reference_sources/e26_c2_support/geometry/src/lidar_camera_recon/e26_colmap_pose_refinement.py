"""Metric camera-pose refinement for the E26 corrected COLMAP pipeline.

The lightweight pose graph supplies rotation priors and verified relative-pose
constraints that pycolmap's position-prior BA does not expose.  A subsequent
Ceres bundle adjustment jointly refines the camera poses and sparse points.
Both stages are required and are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseGraphEdge:
    left: int
    right: int
    rotation_left_right: np.ndarray
    translation_left_right_m: np.ndarray
    edge_type: str
    support_count: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_left_right, dtype=np.float64)
        translation = np.asarray(self.translation_left_right_m, dtype=np.float64)
        if self.left < 0 or self.right <= self.left:
            raise ValueError("pose graph edge indices must satisfy 0 <= left < right")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("invalid pose graph edge geometry")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("pose graph edge rotation is invalid")
        if self.edge_type not in {"local", "lio_local", "loop", "withheld_loop"}:
            raise ValueError("unsupported pose graph edge type")
        if self.support_count < 0:
            raise ValueError("edge support count cannot be negative")
        object.__setattr__(self, "rotation_left_right", rotation.copy())
        object.__setattr__(self, "translation_left_right_m", translation.copy())


@dataclass(frozen=True)
class PoseGraphOptions:
    lio_position_sigma_m: float = 0.05
    lio_rotation_sigma_deg: float = 2.0
    lio_position_weight: float = 1.0
    lio_rotation_weight: float = 1.0
    local_translation_sigma_m: float = 0.015
    local_rotation_sigma_deg: float = 0.6
    lio_local_translation_sigma_m: float = 0.05
    lio_local_rotation_sigma_deg: float = 1.2
    anchor_translation_sigma_m: float = 0.002
    anchor_rotation_sigma_deg: float = 0.1
    loop_translation_sigma_m: float = 0.03
    loop_rotation_sigma_deg: float = 1.0
    smooth_translation_sigma_m: float = 0.04
    smooth_rotation_sigma_deg: float = 1.5
    maximum_translation_correction_m: float = 0.30
    maximum_rotation_correction_deg: float = 10.0
    robust_loss: str = "cauchy"
    maximum_evaluations: int = 100

    def __post_init__(self) -> None:
        positive = (
            self.lio_position_sigma_m,
            self.lio_rotation_sigma_deg,
            self.local_translation_sigma_m,
            self.local_rotation_sigma_deg,
            self.lio_local_translation_sigma_m,
            self.lio_local_rotation_sigma_deg,
            self.anchor_translation_sigma_m,
            self.anchor_rotation_sigma_deg,
            self.loop_translation_sigma_m,
            self.loop_rotation_sigma_deg,
            self.smooth_translation_sigma_m,
            self.smooth_rotation_sigma_deg,
            self.maximum_translation_correction_m,
            self.maximum_rotation_correction_deg,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("pose graph sigmas and bounds must be positive")
        if self.lio_position_weight < 0 or self.lio_rotation_weight < 0:
            raise ValueError("LIO prior weights cannot be negative")
        if self.robust_loss not in {"huber", "cauchy", "soft_l1"}:
            raise ValueError("unsupported robust loss")
        if self.maximum_evaluations <= 0:
            raise ValueError("maximum_evaluations must be positive")


def relative_pose(
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_rotation = rotations_world_camera[left]
    relative_rotation = left_rotation.T @ rotations_world_camera[right]
    relative_translation = left_rotation.T @ (centers_m[right] - centers_m[left])
    return relative_rotation, relative_translation


def make_pose_graph_edge(
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    left: int,
    right: int,
    *,
    edge_type: str,
    support_count: int,
) -> PoseGraphEdge:
    rotation, translation = relative_pose(
        centers_m, rotations_world_camera, left, right
    )
    return PoseGraphEdge(
        left=left,
        right=right,
        rotation_left_right=rotation,
        translation_left_right_m=translation,
        edge_type=edge_type,
        support_count=support_count,
    )


def _candidate_poses(
    parameters: np.ndarray,
    initial_centers_m: np.ndarray,
    initial_rotations_world_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corrections = np.asarray(parameters, dtype=np.float64).reshape(-1, 6)
    delta_rotation = Rotation.from_rotvec(corrections[:, :3]).as_matrix()
    rotations = delta_rotation @ initial_rotations_world_camera
    centers = initial_centers_m + corrections[:, 3:]
    return centers, rotations


def apply_pose_graph_parameters(
    parameters: np.ndarray,
    initial_centers_m: np.ndarray,
    initial_rotations_world_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Public, validated conversion from correction parameters to poses."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.size != len(initial_centers_m) * 6 or not np.all(np.isfinite(values)):
        raise ValueError("invalid pose graph parameter vector")
    return _candidate_poses(
        values.reshape(-1, 6),
        np.asarray(initial_centers_m, dtype=np.float64),
        np.asarray(initial_rotations_world_camera, dtype=np.float64),
    )


def _rotation_vectors(matrices: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(matrices, dtype=np.float64)).as_rotvec()


def _residual_vector(
    parameters: np.ndarray,
    initial_centers_m: np.ndarray,
    initial_rotations_world_camera: np.ndarray,
    prior_centers_m: np.ndarray,
    prior_rotations_world_camera: np.ndarray,
    edges: list[PoseGraphEdge],
    options: PoseGraphOptions,
) -> np.ndarray:
    centers, rotations = _candidate_poses(
        parameters, initial_centers_m, initial_rotations_world_camera
    )
    corrections = np.asarray(parameters, dtype=np.float64).reshape(-1, 6)
    residuals: list[np.ndarray] = []
    if options.lio_position_weight > 0:
        residuals.append(
            np.sqrt(options.lio_position_weight)
            * (centers - prior_centers_m).ravel()
            / options.lio_position_sigma_m
        )
    if options.lio_rotation_weight > 0:
        prior_relative = np.transpose(prior_rotations_world_camera, (0, 2, 1)) @ rotations
        residuals.append(
            np.sqrt(options.lio_rotation_weight)
            * _rotation_vectors(prior_relative).ravel()
            / np.deg2rad(options.lio_rotation_sigma_deg)
        )
    residuals.append(corrections[0, 3:] / options.anchor_translation_sigma_m)
    residuals.append(
        corrections[0, :3] / np.deg2rad(options.anchor_rotation_sigma_deg)
    )
    for edge in edges:
        if edge.edge_type == "withheld_loop":
            continue
        rotation, translation = relative_pose(
            centers, rotations, edge.left, edge.right
        )
        if edge.edge_type == "loop":
            translation_sigma = options.loop_translation_sigma_m
            rotation_sigma = np.deg2rad(options.loop_rotation_sigma_deg)
        elif edge.edge_type == "lio_local":
            translation_sigma = options.lio_local_translation_sigma_m
            rotation_sigma = np.deg2rad(options.lio_local_rotation_sigma_deg)
        else:
            translation_sigma = options.local_translation_sigma_m
            rotation_sigma = np.deg2rad(options.local_rotation_sigma_deg)
        residuals.append(
            (translation - edge.translation_left_right_m) / translation_sigma
        )
        residuals.append(
            _rotation_vectors(
                edge.rotation_left_right.T @ rotation
            )
            / rotation_sigma
        )
    if len(corrections) > 1:
        residuals.append(
            np.diff(corrections[:, 3:], axis=0).ravel()
            / options.smooth_translation_sigma_m
        )
        residuals.append(
            np.diff(corrections[:, :3], axis=0).ravel()
            / np.deg2rad(options.smooth_rotation_sigma_deg)
        )
    return np.concatenate([np.atleast_1d(item) for item in residuals])


def pose_graph_raw_objective(
    parameters: np.ndarray,
    initial_centers_m: np.ndarray,
    initial_rotations_world_camera: np.ndarray,
    prior_centers_m: np.ndarray,
    prior_rotations_world_camera: np.ndarray,
    edges: Iterable[PoseGraphEdge],
    options: PoseGraphOptions,
) -> float:
    """Evaluate the un-robustified squared residual objective for auditing."""

    residual = _residual_vector(
        np.asarray(parameters, dtype=np.float64).reshape(-1),
        np.asarray(initial_centers_m, dtype=np.float64),
        np.asarray(initial_rotations_world_camera, dtype=np.float64),
        np.asarray(prior_centers_m, dtype=np.float64),
        np.asarray(prior_rotations_world_camera, dtype=np.float64),
        list(edges),
        options,
    )
    return float(0.5 * np.sum(residual**2))


def _jacobian_sparsity(
    camera_count: int,
    edges: list[PoseGraphEdge],
    options: PoseGraphOptions,
) -> lil_matrix:
    row_blocks: list[tuple[list[int], int]] = []
    if options.lio_position_weight > 0:
        row_blocks.extend(([index], 3) for index in range(camera_count))
    if options.lio_rotation_weight > 0:
        row_blocks.extend(([index], 3) for index in range(camera_count))
    row_blocks.append(([0], 3))
    row_blocks.append(([0], 3))
    for edge in edges:
        if edge.edge_type != "withheld_loop":
            row_blocks.append(([edge.left, edge.right], 3))
            row_blocks.append(([edge.left, edge.right], 3))
    row_blocks.extend(([index, index + 1], 3) for index in range(camera_count - 1))
    row_blocks.extend(([index, index + 1], 3) for index in range(camera_count - 1))
    row_count = sum(length for _, length in row_blocks)
    pattern = lil_matrix((row_count, camera_count * 6), dtype=np.int8)
    row = 0
    for cameras, length in row_blocks:
        for camera in cameras:
            pattern[row : row + length, camera * 6 : (camera + 1) * 6] = 1
        row += length
    return pattern


def build_lio_motion_edges(
    prior_centers_m: np.ndarray,
    prior_rotations_world_camera: np.ndarray,
    *,
    strides: tuple[int, ...] = (1, 2),
) -> list[PoseGraphEdge]:
    """Build local metric motion factors without treating LIO as global truth."""

    centers = np.asarray(prior_centers_m, dtype=np.float64)
    rotations = np.asarray(prior_rotations_world_camera, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("prior centers must have shape (N,3)")
    if rotations.shape != (len(centers), 3, 3):
        raise ValueError("prior rotations must have shape (N,3,3)")
    result: list[PoseGraphEdge] = []
    for stride in strides:
        if stride <= 0:
            raise ValueError("LIO motion strides must be positive")
        for left in range(len(centers) - stride):
            result.append(
                make_pose_graph_edge(
                    centers,
                    rotations,
                    left,
                    left + stride,
                    edge_type="lio_local",
                    support_count=1,
                )
            )
    return result


def evaluate_edge_type(
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    edges: Iterable[PoseGraphEdge],
    edge_type: str,
) -> dict[str, Any]:
    """Summarize residuals against one independent edge family."""

    translation: list[float] = []
    rotation: list[float] = []
    for edge in edges:
        if edge.edge_type != edge_type:
            continue
        current_rotation, current_translation = relative_pose(
            centers_m, rotations_world_camera, edge.left, edge.right
        )
        translation.append(
            float(np.linalg.norm(current_translation - edge.translation_left_right_m))
        )
        rotation.append(
            float(
                np.rad2deg(
                    Rotation.from_matrix(
                        edge.rotation_left_right.T @ current_rotation
                    ).magnitude()
                )
            )
        )

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "p50": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    return {
        "translation_residual_m": stats(translation),
        "rotation_residual_deg": stats(rotation),
    }


def refine_pose_graph(
    initial_centers_m: np.ndarray,
    initial_rotations_world_camera: np.ndarray,
    prior_centers_m: np.ndarray,
    prior_rotations_world_camera: np.ndarray,
    edges: Iterable[PoseGraphEdge],
    options: PoseGraphOptions,
    *,
    initial_parameters: np.ndarray | None = None,
) -> dict[str, Any]:
    initial_centers = np.asarray(initial_centers_m, dtype=np.float64)
    initial_rotations = np.asarray(initial_rotations_world_camera, dtype=np.float64)
    prior_centers = np.asarray(prior_centers_m, dtype=np.float64)
    prior_rotations = np.asarray(prior_rotations_world_camera, dtype=np.float64)
    camera_count = len(initial_centers)
    if initial_centers.shape != (camera_count, 3):
        raise ValueError("initial camera centers must have shape (N,3)")
    if initial_rotations.shape != (camera_count, 3, 3):
        raise ValueError("initial camera rotations must have shape (N,3,3)")
    if prior_centers.shape != initial_centers.shape or prior_rotations.shape != initial_rotations.shape:
        raise ValueError("LIO prior arrays must match initial camera poses")
    edge_list = list(edges)
    for edge in edge_list:
        if edge.right >= camera_count:
            raise ValueError("pose graph edge index exceeds camera count")
    if initial_parameters is None:
        initial = np.zeros(camera_count * 6, dtype=np.float64)
    else:
        initial = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
        if initial.shape != (camera_count * 6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial pose graph parameters have invalid shape")
    rotation_bound = np.deg2rad(options.maximum_rotation_correction_deg)
    lower = np.tile(
        [-rotation_bound] * 3 + [-options.maximum_translation_correction_m] * 3,
        camera_count,
    )
    upper = -lower
    result = least_squares(
        _residual_vector,
        initial,
        args=(
            initial_centers,
            initial_rotations,
            prior_centers,
            prior_rotations,
            edge_list,
            options,
        ),
        jac="2-point",
        jac_sparsity=_jacobian_sparsity(camera_count, edge_list, options),
        bounds=(lower, upper),
        loss=options.robust_loss,
        f_scale=1.0,
        x_scale="jac",
        tr_solver="lsmr",
        max_nfev=options.maximum_evaluations,
        verbose=0,
    )
    centers, rotations = _candidate_poses(
        result.x, initial_centers, initial_rotations
    )
    corrections = result.x.reshape(-1, 6)
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "optimality": float(result.optimality),
        "initial_cost": float(
            0.5
            * np.sum(
                _residual_vector(
                    initial,
                    initial_centers,
                    initial_rotations,
                    prior_centers,
                    prior_rotations,
                    edge_list,
                    options,
                )
                ** 2
            )
        ),
        "final_cost": float(result.cost),
        "final_raw_squared_cost": float(0.5 * np.sum(result.fun**2)),
        "centers_m": centers,
        "rotations_world_camera": rotations,
        "translation_corrections_m": corrections[:, 3:],
        "rotation_corrections_deg": np.rad2deg(
            np.linalg.norm(corrections[:, :3], axis=1)
        ),
        "parameters": corrections,
    }


def evaluate_withheld_edges(
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    edges: Iterable[PoseGraphEdge],
) -> dict[str, Any]:
    translation: list[float] = []
    rotation: list[float] = []
    for edge in edges:
        if edge.edge_type != "withheld_loop":
            continue
        current_rotation, current_translation = relative_pose(
            centers_m, rotations_world_camera, edge.left, edge.right
        )
        translation.append(
            float(np.linalg.norm(current_translation - edge.translation_left_right_m))
        )
        rotation.append(
            float(
                np.rad2deg(
                    Rotation.from_matrix(
                        edge.rotation_left_right.T @ current_rotation
                    ).magnitude()
                )
            )
        )

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "p50": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    return {
        "translation_residual_m": summary(translation),
        "rotation_residual_deg": summary(rotation),
    }


def build_colmap_pose_graph_edges(
    reconstruction: Any,
    ordered_image_ids: list[int],
    timestamps_ns: np.ndarray,
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    *,
    minimum_loop_separation_s: float,
    minimum_loop_track_support: int,
    maximum_loops_per_image: int = 2,
    withheld_stride: int = 5,
) -> list[PoseGraphEdge]:
    """Build local and independently withheld track-supported loop edges."""

    if len(ordered_image_ids) != len(timestamps_ns):
        raise ValueError("ordered image IDs and timestamps length mismatch")
    if len(set(ordered_image_ids)) != len(ordered_image_ids):
        raise ValueError("ordered image IDs must be unique")
    id_to_index = {image_id: index for index, image_id in enumerate(ordered_image_ids)}
    edge_pairs: set[tuple[int, int]] = set()
    edges: list[PoseGraphEdge] = []
    for stride in (1, 2):
        for left in range(len(ordered_image_ids) - stride):
            right = left + stride
            edge_pairs.add((left, right))
            edges.append(
                make_pose_graph_edge(
                    centers_m,
                    rotations_world_camera,
                    left,
                    right,
                    edge_type="local",
                    support_count=0,
                )
            )

    pair_support: dict[tuple[int, int], int] = {}
    for point in reconstruction.points3D.values():
        indices = sorted(
            {
                id_to_index[element.image_id]
                for element in point.track.elements
                if element.image_id in id_to_index
            }
        )
        for ordinal, left in enumerate(indices):
            for right in indices[ordinal + 1 :]:
                separation_s = (int(timestamps_ns[right]) - int(timestamps_ns[left])) / 1e9
                if separation_s < minimum_loop_separation_s:
                    continue
                pair_support[(left, right)] = pair_support.get((left, right), 0) + 1

    ranked = sorted(
        (
            (support, left, right)
            for (left, right), support in pair_support.items()
            if support >= minimum_loop_track_support and (left, right) not in edge_pairs
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used = np.zeros(len(ordered_image_ids), dtype=np.int64)
    accepted_loops: list[tuple[int, int, int]] = []
    for support, left, right in ranked:
        if used[left] >= maximum_loops_per_image or used[right] >= maximum_loops_per_image:
            continue
        accepted_loops.append((left, right, support))
        used[left] += 1
        used[right] += 1
    for ordinal, (left, right, support) in enumerate(accepted_loops):
        edge_type = "withheld_loop" if ordinal % withheld_stride == 0 else "loop"
        edges.append(
            make_pose_graph_edge(
                centers_m,
                rotations_world_camera,
                left,
                right,
                edge_type=edge_type,
                support_count=support,
            )
        )
    return edges


def apply_world_camera_poses(
    reconstruction: Any,
    ordered_image_ids: list[int],
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
) -> None:
    """Write T_camera_world poses into a pycolmap reconstruction in memory."""

    import pycolmap

    centers = np.asarray(centers_m, dtype=np.float64)
    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    if centers.shape != (len(ordered_image_ids), 3):
        raise ValueError("camera center array shape mismatch")
    if rotations.shape != (len(ordered_image_ids), 3, 3):
        raise ValueError("camera rotation array shape mismatch")
    for index, image_id in enumerate(ordered_image_ids):
        image = reconstruction.images[image_id]
        rotation_camera_world = rotations[index].T
        translation_camera_world = -rotation_camera_world @ centers[index]
        image.frame.set_cam_from_world(
            image.camera_id,
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(rotation_camera_world),
                translation_camera_world,
            ),
        )


def extract_world_camera_poses(
    reconstruction: Any, ordered_image_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for image_id in ordered_image_ids:
        image = reconstruction.images[image_id]
        centers.append(np.asarray(image.projection_center(), dtype=np.float64))
        rotations.append(
            np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64).T
        )
    return np.stack(centers), np.stack(rotations)


def make_pycolmap_pose_priors(
    reconstruction: Any,
    ordered_image_ids: list[int],
    prior_centers_m: np.ndarray,
    *,
    position_sigma_m: float,
) -> list[Any]:
    import pycolmap

    covariance = np.eye(3, dtype=np.float64) * position_sigma_m**2
    priors: list[Any] = []
    for prior_id, image_id in enumerate(ordered_image_ids):
        image = reconstruction.images[image_id]
        priors.append(
            pycolmap.PosePrior(
                pose_prior_id=prior_id,
                corr_data_id=image.data_id,
                position=np.asarray(prior_centers_m[prior_id], dtype=np.float64),
                position_covariance=covariance,
                coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
            )
        )
    return priors


def run_pycolmap_joint_ba(
    reconstruction: Any,
    ordered_image_ids: list[int],
    prior_centers_m: np.ndarray,
    *,
    position_sigma_m: float,
    maximum_iterations: int = 100,
) -> dict[str, Any]:
    """Run points-only warmup then Ceres camera/point BA with position priors."""

    import pycolmap

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    options.print_summary = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = 1.0
    options.ceres.solver_options.max_num_iterations = maximum_iterations
    options.ceres.solver_options.num_threads = -1

    warmup_config = pycolmap.BundleAdjustmentConfig()
    for image_id in ordered_image_ids:
        image = reconstruction.images[image_id]
        warmup_config.add_image(image_id)
        warmup_config.set_constant_cam_intrinsics(image.camera_id)
        warmup_config.set_constant_rig_from_world_pose(image.frame_id)
    warmup = pycolmap.create_default_ceres_bundle_adjuster(
        options, warmup_config, reconstruction
    ).solve()

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in ordered_image_ids:
        image = reconstruction.images[image_id]
        config.add_image(image_id)
        config.set_constant_cam_intrinsics(image.camera_id)
        config.set_variable_rig_from_world_pose(image.frame_id)
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    prior_options.prior_position_fallback_stddev = position_sigma_m
    prior_options.alignment_ransac.max_error = max(0.20, 4.0 * position_sigma_m)
    prior_options.alignment_ransac.min_inlier_ratio = 0.20
    prior_options.alignment_ransac.random_seed = 20260809
    prior_options.ceres.prior_position_loss_function_type = (
        pycolmap.LossFunctionType.CAUCHY
    )
    prior_options.ceres.prior_position_loss_scale = 2.5
    priors = make_pycolmap_pose_priors(
        reconstruction,
        ordered_image_ids,
        prior_centers_m,
        position_sigma_m=position_sigma_m,
    )
    adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(
        options, prior_options, config, priors, reconstruction
    )
    summary = adjuster.solve()
    return {
        "warmup": warmup.todict(),
        "joint": summary.todict(),
        "pose_prior_count": len(priors),
        "backend": "pycolmap_ceres_pose_prior_bundle_adjustment",
        "rotation_prior_stage": (
            "rotation priors and loop factors are supplied by the preceding "
            "pose-graph initialization; pycolmap Ceres exposes position priors"
        ),
    }


def run_pycolmap_points_only_ba(
    reconstruction: Any,
    ordered_image_ids: list[int],
    *,
    maximum_iterations: int = 200,
) -> dict[str, Any]:
    """Refresh sparse points with final camera poses and intrinsics fixed."""

    import pycolmap

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_rig_from_world = False
    options.refine_points3D = True
    options.print_summary = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = 1.0
    options.ceres.solver_options.max_num_iterations = maximum_iterations
    options.ceres.solver_options.num_threads = -1
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in ordered_image_ids:
        image = reconstruction.images[image_id]
        config.add_image(image_id)
        config.set_constant_cam_intrinsics(image.camera_id)
        config.set_constant_rig_from_world_pose(image.frame_id)
    summary_result = pycolmap.create_default_ceres_bundle_adjuster(
        options, config, reconstruction
    ).solve()
    reconstruction.update_point_3d_errors()
    return summary_result.todict()
