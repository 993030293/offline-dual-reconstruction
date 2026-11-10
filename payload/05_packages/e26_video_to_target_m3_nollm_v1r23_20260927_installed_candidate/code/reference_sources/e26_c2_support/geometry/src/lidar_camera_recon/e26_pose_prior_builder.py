"""Bridge COLMAP sparse poses and interpolated metric LIO camera priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .e26_sim3_initialization import SimilarityTransform


def load_binding_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e26_trajectory_binding_v1":
        raise ValueError("unsupported trajectory binding report")
    if payload.get("status") != "complete" or payload.get("abstained_count") != 0:
        raise ValueError("trajectory binding is not complete")
    by_name = {str(item["image_name"]): item for item in payload["images"]}
    if len(by_name) != len(payload["images"]):
        raise ValueError("trajectory binding contains duplicate image names")
    return payload, by_name


def colmap_world_camera_pose(image: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return camera center and R_world_camera from a pycolmap Image."""

    center = np.asarray(image.projection_center(), dtype=np.float64)
    rotation_camera_world = np.asarray(
        image.cam_from_world().rotation.matrix(), dtype=np.float64
    )
    rotation_world_camera = rotation_camera_world.T
    if center.shape != (3,) or rotation_world_camera.shape != (3, 3):
        raise ValueError("invalid COLMAP image pose")
    return center, rotation_world_camera


def transformed_colmap_camera_pose(
    image: Any, similarity: SimilarityTransform
) -> tuple[np.ndarray, np.ndarray]:
    center, rotation_world_camera = colmap_world_camera_pose(image)
    transformed_center = similarity.apply(center[None, :])[0]
    transformed_rotation = similarity.rotation @ rotation_world_camera
    return transformed_center, transformed_rotation


def prior_camera_pose(binding: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = binding["T_world_camera_interpolated"]
    if transform.get("source_frame") != "camera_optical":
        raise ValueError("pose prior source frame must be camera_optical")
    if transform.get("target_frame") != "world_lio":
        raise ValueError("pose prior target frame must be world_lio")
    pose = np.asarray(transform["pose_xyz_xyzw"], dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("invalid pose prior vector")
    return pose[:3], Rotation.from_quat(pose[3:]).as_matrix()


def rotation_residual_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.rad2deg(Rotation.from_matrix(np.asarray(left).T @ np.asarray(right)).magnitude())
    )


def build_pose_correspondences(
    reconstruction: Any,
    bindings_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for image_id, image in reconstruction.images.items():
        if not image.has_pose or image.name not in bindings_by_name:
            continue
        center, rotation_world_camera = colmap_world_camera_pose(image)
        prior_center, prior_rotation = prior_camera_pose(bindings_by_name[image.name])
        rows.append(
            {
                "image_id": int(image_id),
                "image_name": image.name,
                "image_timestamp_ns": int(bindings_by_name[image.name]["image_timestamp_ns"]),
                "sfm_camera_center": center,
                "sfm_R_world_camera": rotation_world_camera,
                "lio_camera_center_m": prior_center,
                "lio_R_world_camera": prior_rotation,
            }
        )
    rows.sort(key=lambda item: item["image_timestamp_ns"])
    if len(rows) < 3:
        raise ValueError("fewer than three registered images have pose priors")
    return {
        "rows": rows,
        "source_centers": np.stack([item["sfm_camera_center"] for item in rows]),
        "target_centers": np.stack([item["lio_camera_center_m"] for item in rows]),
        "timestamps_ns": np.asarray(
            [item["image_timestamp_ns"] for item in rows], dtype=np.int64
        ),
    }


def serialize_pose_prior_manifest(
    correspondences: dict[str, Any],
    similarity: SimilarityTransform,
    residuals_m: np.ndarray,
    inliers: np.ndarray,
    *,
    position_sigma_m: float,
    rotation_sigma_deg: float,
    binding_report_sha256: str,
    sparse_model_signature: str,
) -> dict[str, Any]:
    rows = correspondences["rows"]
    residuals = np.asarray(residuals_m, dtype=np.float64)
    inlier_mask = np.asarray(inliers, dtype=bool)
    if len(rows) != len(residuals) or len(rows) != len(inlier_mask):
        raise ValueError("pose prior diagnostics length mismatch")
    images: list[dict[str, Any]] = []
    rotation_residuals: list[float] = []
    for index, row in enumerate(rows):
        transformed_rotation = similarity.rotation @ row["sfm_R_world_camera"]
        rotation_error = rotation_residual_deg(
            transformed_rotation, row["lio_R_world_camera"]
        )
        rotation_residuals.append(rotation_error)
        images.append(
            {
                "image_id": row["image_id"],
                "image_name": row["image_name"],
                "image_timestamp_ns": row["image_timestamp_ns"],
                "T_world_camera_prior": {
                    "transform_direction": "T_target_source",
                    "source_frame": "camera_optical",
                    "target_frame": "world_lio",
                    "translation_unit": "m",
                    "quaternion_order": "xyzw",
                    "pose_xyz_xyzw": np.r_[
                        row["lio_camera_center_m"],
                        Rotation.from_matrix(row["lio_R_world_camera"]).as_quat(),
                    ].tolist(),
                },
                "position_sigma_m": [position_sigma_m] * 3,
                "rotation_sigma_deg": [rotation_sigma_deg] * 3,
                "sim3_position_residual_m": float(residuals[index]),
                "sim3_rotation_residual_deg": rotation_error,
                "sim3_ransac_inlier": bool(inlier_mask[index]),
            }
        )
    rotation_values = np.asarray(rotation_residuals, dtype=np.float64)
    return {
        "schema_version": "e26_colmap_lio_pose_prior_manifest_v1",
        "status": "complete_pose_prior_manifest",
        "binding_report_sha256": binding_report_sha256,
        "sparse_model_signature": sparse_model_signature,
        "sim3": similarity.to_json(),
        "registered_prior_count": len(images),
        "sim3_inlier_count": int(np.count_nonzero(inlier_mask)),
        "position_residual_m": {
            "p50": float(np.percentile(residuals, 50)),
            "p75": float(np.percentile(residuals, 75)),
            "p95": float(np.percentile(residuals, 95)),
            "max": float(np.max(residuals)),
        },
        "rotation_residual_deg": {
            "p50": float(np.percentile(rotation_values, 50)),
            "p95": float(np.percentile(rotation_values, 95)),
            "max": float(np.max(rotation_values)),
        },
        "images": images,
        "scientific_boundary": (
            "Sim(3) is an initialization and diagnostic. It cannot repair "
            "time-varying trajectory drift by itself."
        ),
    }

