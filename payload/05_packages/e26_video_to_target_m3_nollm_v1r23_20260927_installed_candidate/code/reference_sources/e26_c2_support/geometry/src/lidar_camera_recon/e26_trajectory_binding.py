"""Bind image timestamps to a metric LIO trajectory with continuous SE(3).

The public functions in this module deliberately use explicit frame and unit
contracts.  Nearest-neighbour poses are emitted only as an A/B diagnostic;
the operational prior is translation interpolation plus quaternion SLERP.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .e26.contracts import RigidTransformRecord
from .e26.trajectory_interpolation import interpolate_trajectory
from .e26.trajectory_io import Trajectory, sha256_file


SCHEMA_VERSION = "e26_trajectory_binding_v1"


def canonicalize_trajectory_frames(
    trajectory: Trajectory,
    *,
    source_frame: str = "base_link",
    target_frame: str = "world_lio",
) -> Trajectory:
    """Relabel a verified odometry transform without changing its numbers."""

    if not source_frame or not target_frame:
        raise ValueError("canonical trajectory frame names cannot be empty")
    return replace(
        trajectory,
        source_frame=source_frame,
        target_frame=target_frame,
    )


def load_image_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("image manifest must be a non-empty JSON list")
    names: set[str] = set()
    records: list[dict[str, Any]] = []
    previous_timestamp: int | None = None
    for ordinal, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"image manifest row {ordinal} is not an object")
        name = str(raw.get("image_name", "")).strip()
        if not name or name in names:
            raise ValueError(f"invalid or duplicate image_name at row {ordinal}")
        try:
            timestamp = int(raw["camera_header_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid camera_header_ns at image row {ordinal}"
            ) from error
        if timestamp <= 0:
            raise ValueError("camera_header_ns must be positive int64 nanoseconds")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("image timestamps must be strictly increasing")
        previous_timestamp = timestamp
        names.add(name)
        records.append(dict(raw))
    return records, sha256_file(resolved)


def _rotation_delta_deg(left: RigidTransformRecord, right: RigidTransformRecord) -> float:
    relative = left.matrix[:3, :3].T @ right.matrix[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def _check_timestamp_domain(records: list[dict[str, Any]], trajectory: Trajectory) -> None:
    query = np.asarray([int(item["camera_header_ns"]) for item in records], dtype=np.int64)
    inside = (query >= trajectory.start_ns) & (query <= trajectory.end_ns)
    if np.all(inside):
        return
    # A common integration error is passing seconds through a field named *_ns.
    # Detect it explicitly instead of returning a misleading range failure.
    scaled = query.astype(np.float64) * 1_000_000_000.0
    scaled_inside = (scaled >= trajectory.start_ns) & (scaled <= trajectory.end_ns)
    if np.mean(scaled_inside) > np.mean(inside):
        raise ValueError(
            "timestamp_unit_mismatch: camera_header_ns appears to contain seconds"
        )


def bind_image_records(
    records: list[dict[str, Any]],
    trajectory_world_body: Trajectory,
    transform_body_camera: RigidTransformRecord,
    *,
    maximum_bracket_span_ns: int,
    trajectory_role: str,
    image_manifest_sha256: str,
) -> dict[str, Any]:
    """Create auditable interpolated and nearest camera pose records."""

    if trajectory_world_body.source_frame != "base_link":
        raise ValueError("trajectory must encode T_world_lio_base_link")
    if trajectory_world_body.target_frame != "world_lio":
        raise ValueError("trajectory target frame must be world_lio")
    if transform_body_camera.source_frame != "camera_optical":
        raise ValueError("extrinsic source frame must be camera_optical")
    if transform_body_camera.target_frame != "base_link":
        raise ValueError("extrinsic target frame must be base_link")
    if len(image_manifest_sha256) != 64:
        raise ValueError("image manifest SHA256 is required")
    _check_timestamp_domain(records, trajectory_world_body)

    output: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    for ordinal, item in enumerate(records):
        timestamp_ns = int(item["camera_header_ns"])
        try:
            transform_world_body, bracket = interpolate_trajectory(
                trajectory_world_body,
                timestamp_ns,
                maximum_bracket_span_ns=maximum_bracket_span_ns,
            )
        except ValueError as error:
            abstained.append(
                {
                    "ordinal": ordinal,
                    "candidate_id": str(item.get("candidate_id") or item["image_name"]),
                    "image_name": str(item["image_name"]),
                    "image_timestamp_ns": timestamp_ns,
                    "reason": str(error),
                }
            )
            continue

        transform_world_camera = transform_world_body.compose(transform_body_camera)
        times = trajectory_world_body.timestamps_ns
        nearest_index = int(np.argmin(np.abs(times - timestamp_ns)))
        nearest_body = RigidTransformRecord.from_pose(
            trajectory_world_body.poses_xyz_xyzw[nearest_index],
            source_frame="base_link",
            target_frame="world_lio",
        )
        nearest_camera = nearest_body.compose(transform_body_camera)
        center_delta = float(
            np.linalg.norm(
                transform_world_camera.matrix[:3, 3]
                - nearest_camera.matrix[:3, 3]
            )
        )
        output.append(
            {
                "ordinal": ordinal,
                "candidate_id": str(item.get("candidate_id") or item["image_name"]),
                "image_name": str(item["image_name"]),
                "image_timestamp_ns": timestamp_ns,
                "left_pose_timestamp_ns": bracket.left_timestamp_ns,
                "right_pose_timestamp_ns": bracket.right_timestamp_ns,
                "alpha": bracket.alpha,
                "bracket_span_ms": bracket.bracket_span_ns / 1_000_000.0,
                "nearest_pose_index": nearest_index,
                "nearest_pose_timestamp_ns": int(times[nearest_index]),
                "nearest_pose_error_ms": abs(int(times[nearest_index]) - timestamp_ns)
                / 1_000_000.0,
                "nearest_vs_interpolated_camera_center_delta_m": center_delta,
                "nearest_vs_interpolated_rotation_delta_deg": _rotation_delta_deg(
                    nearest_camera, transform_world_camera
                ),
                "T_world_camera_interpolated": transform_world_camera.to_json(),
                "T_world_camera_nearest_diagnostic": nearest_camera.to_json(),
                "interpolation_status": "bound_interpolated",
            }
        )

    bracket_spans = np.asarray(
        [record["bracket_span_ms"] for record in output], dtype=np.float64
    )
    center_deltas = np.asarray(
        [record["nearest_vs_interpolated_camera_center_delta_m"] for record in output],
        dtype=np.float64,
    )
    rotation_deltas = np.asarray(
        [record["nearest_vs_interpolated_rotation_delta_deg"] for record in output],
        dtype=np.float64,
    )

    def summary(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"p50": None, "p95": None, "max": None}
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not abstained else "abstained_partial_binding",
        "trajectory_role": trajectory_role,
        "online_causality": "offline_noncausal_binding_uses_bracketing_pose_samples",
        "transform_convention": "T_target_source",
        "translation_unit": "meter",
        "rotation_representation": "quaternion_xyzw",
        "timestamp_unit": "int64_nanosecond",
        "image_manifest_sha256": image_manifest_sha256,
        "trajectory": trajectory_world_body.to_binding_json(),
        "T_base_link_camera_optical": transform_body_camera.to_json(),
        "maximum_bracket_span_ns": int(maximum_bracket_span_ns),
        "image_count": len(records),
        "bound_count": len(output),
        "abstained_count": len(abstained),
        "diagnostics": {
            "bracket_span_ms": summary(bracket_spans),
            "nearest_vs_interpolated_camera_center_delta_m": summary(center_deltas),
            "nearest_vs_interpolated_rotation_delta_deg": summary(rotation_deltas),
        },
        "images": output,
        "abstained_images": abstained,
        "scientific_boundary": (
            "Interpolated LIO camera poses are metric priors. They are not a "
            "claim that corrected COLMAP poses or dense geometry are complete."
        ),
    }

