"""Continuous SE(3) trajectory interpolation with auditable brackets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..e26_paperbox_precision import slerp_xyzw
from .contracts import RigidTransformRecord
from .trajectory_io import Trajectory


@dataclass(frozen=True)
class PoseBinding:
    query_timestamp_ns: int
    left_index: int
    right_index: int
    left_timestamp_ns: int
    right_timestamp_ns: int
    alpha: float
    bracket_span_ns: int
    maximum_time_gap_ns: int
    interpolation_method: str = "translation_linear__rotation_slerp"
    extrapolated: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "query_timestamp_ns": int(self.query_timestamp_ns),
            "left_index": int(self.left_index),
            "right_index": int(self.right_index),
            "left_timestamp_ns": int(self.left_timestamp_ns),
            "right_timestamp_ns": int(self.right_timestamp_ns),
            "alpha": float(self.alpha),
            "bracket_span_ns": int(self.bracket_span_ns),
            "maximum_time_gap_ns": int(self.maximum_time_gap_ns),
            "interpolation_method": self.interpolation_method,
            "extrapolated": bool(self.extrapolated),
        }


class TrajectoryRangeError(ValueError):
    """Raised when a query would require trajectory extrapolation."""


class TrajectoryGapError(ValueError):
    """Raised when the interpolation bracket is too wide."""


def interpolate_trajectory(
    trajectory: Trajectory,
    query_timestamp_ns: int,
    *,
    maximum_bracket_span_ns: int,
) -> tuple[RigidTransformRecord, PoseBinding]:
    if maximum_bracket_span_ns <= 0:
        raise ValueError("maximum_bracket_span_ns must be positive")
    timestamp = int(query_timestamp_ns)
    times = trajectory.timestamps_ns
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and int(times[right]) == timestamp:
        binding = PoseBinding(
            query_timestamp_ns=timestamp,
            left_index=right,
            right_index=right,
            left_timestamp_ns=timestamp,
            right_timestamp_ns=timestamp,
            alpha=0.0,
            bracket_span_ns=0,
            maximum_time_gap_ns=0,
        )
        return (
            RigidTransformRecord.from_pose(
                trajectory.poses_xyz_xyzw[right],
                source_frame=trajectory.source_frame,
                target_frame=trajectory.target_frame,
            ),
            binding,
        )
    if right == 0 or right == len(times):
        raise TrajectoryRangeError(
            "abstain_out_of_trajectory_range: extrapolation is forbidden"
        )
    left = right - 1
    left_ns = int(times[left])
    right_ns = int(times[right])
    span_ns = right_ns - left_ns
    if span_ns > maximum_bracket_span_ns:
        raise TrajectoryGapError(
            "abstain_trajectory_bracket_too_wide: "
            f"{span_ns} ns > {maximum_bracket_span_ns} ns"
        )
    alpha = float((timestamp - left_ns) / span_ns)
    left_pose = trajectory.poses_xyz_xyzw[left]
    right_pose = trajectory.poses_xyz_xyzw[right]
    pose = np.r_[
        (1.0 - alpha) * left_pose[:3] + alpha * right_pose[:3],
        slerp_xyzw(left_pose[3:], right_pose[3:], alpha),
    ]
    binding = PoseBinding(
        query_timestamp_ns=timestamp,
        left_index=left,
        right_index=right,
        left_timestamp_ns=left_ns,
        right_timestamp_ns=right_ns,
        alpha=alpha,
        bracket_span_ns=span_ns,
        maximum_time_gap_ns=max(timestamp - left_ns, right_ns - timestamp),
    )
    return (
        RigidTransformRecord.from_pose(
            pose,
            source_frame=trajectory.source_frame,
            target_frame=trajectory.target_frame,
        ),
        binding,
    )


def bind_timestamps(
    trajectory: Trajectory,
    query_timestamps_ns: np.ndarray,
    *,
    maximum_bracket_span_ns: int,
) -> tuple[list[RigidTransformRecord], list[PoseBinding]]:
    transforms: list[RigidTransformRecord] = []
    bindings: list[PoseBinding] = []
    for timestamp in np.asarray(query_timestamps_ns, dtype=np.int64):
        transform, binding = interpolate_trajectory(
            trajectory,
            int(timestamp),
            maximum_bracket_span_ns=maximum_bracket_span_ns,
        )
        transforms.append(transform)
        bindings.append(binding)
    return transforms, bindings
