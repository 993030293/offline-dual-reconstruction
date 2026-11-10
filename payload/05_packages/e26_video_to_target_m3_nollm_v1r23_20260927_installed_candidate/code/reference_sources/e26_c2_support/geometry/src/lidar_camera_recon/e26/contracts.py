"""Coordinate and transform contracts shared by the E26 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


TRANSFORM_CONVENTION = "T_target_source"
TRANSLATION_UNIT = "m"
QUATERNION_ORDER = "xyzw"


def _frame(value: str, field: str) -> str:
    name = str(value).strip()
    if not name or any(character.isspace() for character in name):
        raise ValueError(f"{field} must be a non-empty frame token")
    return name


def validate_rotation_matrix(matrix: np.ndarray, *, atol: float = 1e-8) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(value), 1.0, atol=atol):
        raise ValueError("rotation determinant is not +1")
    return value


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return value / norm


def pose_xyz_xyzw_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("pose must be a finite xyz_xyzw vector")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        normalize_quaternion_xyzw(value[3:])
    ).as_matrix()
    result[:3, 3] = value[:3]
    return result


def matrix_to_pose_xyz_xyzw(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("transform has an invalid homogeneous row")
    rotation = validate_rotation_matrix(value[:3, :3])
    return np.r_[value[:3, 3], Rotation.from_matrix(rotation).as_quat()]


@dataclass(frozen=True)
class RigidTransformRecord:
    """A rigid transform with explicit ``T_target_source`` direction."""

    matrix: np.ndarray
    source_frame: str
    target_frame: str
    transform_direction: str = TRANSFORM_CONVENTION
    translation_unit: str = TRANSLATION_UNIT
    quaternion_order: str = QUATERNION_ORDER

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be a finite 4x4 transform")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("matrix has an invalid homogeneous row")
        validate_rotation_matrix(matrix[:3, :3])
        if self.transform_direction != TRANSFORM_CONVENTION:
            raise ValueError(f"transform_direction must be {TRANSFORM_CONVENTION}")
        if self.translation_unit != TRANSLATION_UNIT:
            raise ValueError(f"translation_unit must be {TRANSLATION_UNIT}")
        if self.quaternion_order != QUATERNION_ORDER:
            raise ValueError(f"quaternion_order must be {QUATERNION_ORDER}")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "source_frame", _frame(self.source_frame, "source_frame"))
        object.__setattr__(self, "target_frame", _frame(self.target_frame, "target_frame"))

    @classmethod
    def from_pose(
        cls, pose_xyz_xyzw: np.ndarray, *, source_frame: str, target_frame: str
    ) -> "RigidTransformRecord":
        return cls(
            pose_xyz_xyzw_to_matrix(pose_xyz_xyzw),
            source_frame=source_frame,
            target_frame=target_frame,
        )

    @property
    def pose_xyz_xyzw(self) -> np.ndarray:
        return matrix_to_pose_xyz_xyzw(self.matrix)

    def inverse(self) -> "RigidTransformRecord":
        rotation = self.matrix[:3, :3]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation.T
        result[:3, 3] = -rotation.T @ self.matrix[:3, 3]
        return RigidTransformRecord(
            result,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def compose(self, other: "RigidTransformRecord") -> "RigidTransformRecord":
        """Return ``self @ other`` while enforcing the frame chain.

        If ``self`` is ``T_AB`` and ``other`` is ``T_BC``, the result is
        ``T_AC``.
        """

        if self.source_frame != other.target_frame:
            raise ValueError(
                "transform chain mismatch: "
                f"{self.target_frame}<-{self.source_frame} cannot compose with "
                f"{other.target_frame}<-{other.source_frame}"
            )
        return RigidTransformRecord(
            self.matrix @ other.matrix,
            source_frame=other.source_frame,
            target_frame=self.target_frame,
        )

    def apply(self, points_source: np.ndarray) -> np.ndarray:
        points = np.asarray(points_source, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points_source must be a finite array with shape (N, 3)")
        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    def to_json(self) -> dict[str, object]:
        return {
            "transform_direction": self.transform_direction,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "translation_unit": self.translation_unit,
            "quaternion_order": self.quaternion_order,
            "pose_xyz_xyzw": self.pose_xyz_xyzw.tolist(),
        }
