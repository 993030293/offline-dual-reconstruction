"""Robust metric Sim(3) initialization for COLMAP camera centers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SimilarityTransform:
    """Map source coordinates into target coordinates as ``s R x + t``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("similarity scale must be finite and positive")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("invalid similarity transform shape")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("similarity rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("similarity rotation must have determinant +1")
        if not np.all(np.isfinite(translation)):
            raise ValueError("similarity translation must be finite")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", translation.copy())

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")
        return self.scale * (values @ self.rotation.T) + self.translation

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": "e26_sim3_v1",
            "transform_name": "T_world_lio_from_world_sfm",
            "source_frame": "world_sfm",
            "target_frame": "world_lio",
            "translation_unit": "meter",
            "scale": float(self.scale),
            "rotation_xyzw": Rotation.from_matrix(self.rotation).as_quat().tolist(),
            "translation_m": self.translation.tolist(),
        }


def estimate_similarity_umeyama(
    source: np.ndarray, target: np.ndarray
) -> SimilarityTransform:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape (N,3)")
    if len(source) < 3 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("at least three finite correspondences are required")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-15 or np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("similarity correspondences are geometrically degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.dot(singular, sign) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return SimilarityTransform(scale, rotation, translation)


def robust_similarity_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_error_m: float,
    minimum_inliers: int,
    iterations: int = 2000,
    seed: int = 20260809,
) -> tuple[SimilarityTransform, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) < max(3, minimum_inliers):
        raise ValueError("not enough Sim(3) correspondences")
    if maximum_error_m <= 0 or iterations <= 0:
        raise ValueError("invalid RANSAC configuration")
    generator = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_median = np.inf
    for _ in range(iterations):
        selected = generator.choice(len(source), 3, replace=False)
        try:
            candidate = estimate_similarity_umeyama(source[selected], target[selected])
        except ValueError:
            continue
        residuals = np.linalg.norm(candidate.apply(source) - target, axis=1)
        inliers = residuals <= maximum_error_m
        count = int(np.count_nonzero(inliers))
        if count < minimum_inliers:
            continue
        median = float(np.median(residuals[inliers]))
        if best_inliers is None or count > np.count_nonzero(best_inliers) or (
            count == np.count_nonzero(best_inliers) and median < best_median
        ):
            best_inliers = inliers
            best_median = median
    if best_inliers is None:
        raise ValueError("abstain_sim3_ransac_no_consensus")
    refined = estimate_similarity_umeyama(source[best_inliers], target[best_inliers])
    residuals = np.linalg.norm(refined.apply(source) - target, axis=1)
    final_inliers = residuals <= maximum_error_m
    if np.count_nonzero(final_inliers) < minimum_inliers:
        raise ValueError("abstain_sim3_refinement_lost_consensus")
    refined = estimate_similarity_umeyama(source[final_inliers], target[final_inliers])
    residuals = np.linalg.norm(refined.apply(source) - target, axis=1)
    return refined, residuals, final_inliers


def time_binned_residuals(
    timestamps_ns: np.ndarray,
    residuals_m: np.ndarray,
    *,
    bin_count: int = 8,
) -> list[dict[str, float | int]]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    residuals = np.asarray(residuals_m, dtype=np.float64)
    if timestamps.shape != residuals.shape or len(timestamps) == 0:
        raise ValueError("timestamps and residuals must be non-empty matching vectors")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    edges = np.linspace(int(timestamps.min()), int(timestamps.max()) + 1, bin_count + 1)
    result: list[dict[str, float | int]] = []
    for index in range(bin_count):
        selected = (timestamps >= edges[index]) & (timestamps < edges[index + 1])
        if not np.any(selected):
            continue
        values = residuals[selected]
        result.append(
            {
                "bin": index,
                "start_timestamp_ns": int(edges[index]),
                "end_timestamp_ns": int(edges[index + 1]),
                "count": int(len(values)),
                "p50_m": float(np.percentile(values, 50)),
                "p75_m": float(np.percentile(values, 75)),
                "p95_m": float(np.percentile(values, 95)),
            }
        )
    return result

