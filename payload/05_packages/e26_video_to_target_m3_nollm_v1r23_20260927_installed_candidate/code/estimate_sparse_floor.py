"""Estimate a broad low floor plane from a metric sparse scene."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor


CELL_SIZE_M = 0.10
LOW_QUANTILE = 0.15
FLOOR_SEARCH_RADIUS_M = 3.0
MAXIMUM_CAMERA_FLOOR_HEIGHT_M = 2.5
MINIMUM_CAMERA_FLOOR_HEIGHT_M = 0.7
MINIMUM_INLIER_CELL_COUNT = 60
MINIMUM_INLIER_CELL_FRACTION = 0.25
MAXIMUM_PLANE_TILT_DEGREES = 15.0
MAXIMUM_INLIER_RESIDUAL_M = 0.035


def estimate_sparse_floor(
    points: np.ndarray,
    focus_xy: np.ndarray,
    median_camera_z_m: float,
    search_radius_m: float,
) -> dict:
    points = np.asarray(points, dtype=np.float64)
    focus_xy = np.asarray(focus_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Metric sparse floor input is invalid")
    if focus_xy.shape != (2,) or not np.isfinite(focus_xy).all():
        raise ValueError("Metric optical focus is invalid")
    if not np.isfinite(median_camera_z_m) or search_radius_m <= 0:
        raise ValueError("Metric camera geometry is invalid")

    radius = min(FLOOR_SEARCH_RADIUS_M, search_radius_m)
    radius_from_focus = np.linalg.norm(points[:, :2] - focus_xy[None, :], axis=1)
    candidate = points[
        (radius_from_focus < radius)
        & (points[:, 2] > median_camera_z_m - MAXIMUM_CAMERA_FLOOR_HEIGHT_M)
        & (points[:, 2] < median_camera_z_m - 0.55)
    ]
    if len(candidate) < 1000:
        raise RuntimeError(f"Too few metric sparse floor candidates: {len(candidate)}")

    keys = np.floor(candidate[:, :2] / CELL_SIZE_M).astype(np.int32)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.concatenate((
        [0],
        np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1,
        [len(order)],
    ))
    envelope = np.empty((len(boundaries) - 1, 3), dtype=np.float64)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        group = candidate[order[start:end]]
        envelope[index, :2] = np.median(group[:, :2], axis=0)
        envelope[index, 2] = np.quantile(group[:, 2], LOW_QUANTILE)
    if len(envelope) < MINIMUM_INLIER_CELL_COUNT:
        raise RuntimeError("Metric sparse floor has too few occupied XY cells")

    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=3,
        residual_threshold=MAXIMUM_INLIER_RESIDUAL_M,
        max_trials=5000,
        stop_probability=0.999999,
        random_state=20260927,
    ).fit(envelope[:, :2], envelope[:, 2])
    inlier_mask = model.inlier_mask_
    inliers = envelope[inlier_mask]
    fraction = len(inliers) / len(envelope)
    if len(inliers) < MINIMUM_INLIER_CELL_COUNT or fraction < MINIMUM_INLIER_CELL_FRACTION:
        raise RuntimeError(
            "Sparse floor lacks a dominant low spatial plane: "
            f"inlier_cells={len(inliers)}/{len(envelope)}"
        )
    x_slope, y_slope = model.estimator_.coef_
    intercept = float(model.estimator_.intercept_)
    normal = np.asarray([-x_slope, -y_slope, 1.0], dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    normal /= magnitude
    offset_m = -intercept / magnitude
    if normal[2] < np.cos(np.deg2rad(MAXIMUM_PLANE_TILT_DEGREES)):
        raise RuntimeError(f"Sparse floor is not gravity aligned: normal_z={normal[2]:.6f}")
    extent = np.ptp(inliers[:, :2], axis=0)
    if np.any(extent < 1.0):
        raise RuntimeError(f"Sparse floor lacks two-axis coverage: {extent.tolist()}")
    ground_z = float(intercept + np.dot(model.estimator_.coef_, focus_xy))
    camera_height = median_camera_z_m - ground_z
    if not MINIMUM_CAMERA_FLOOR_HEIGHT_M <= camera_height <= MAXIMUM_CAMERA_FLOOR_HEIGHT_M:
        raise RuntimeError(f"Sparse floor implies implausible camera height: {camera_height:.6f} m")
    residuals = np.abs(inliers @ normal + offset_m)
    support_point_count = int(np.count_nonzero(np.abs(candidate @ normal + offset_m) <= 0.05))
    return {
        "method": "deterministic RANSAC on low spatial envelope of metric sparse scene",
        "candidate_point_count": len(candidate),
        "envelope_cell_size_m": CELL_SIZE_M,
        "envelope_low_quantile": LOW_QUANTILE,
        "envelope_cell_count": len(envelope),
        "inlier_cell_count": len(inliers),
        "inlier_cell_fraction": fraction,
        "support_point_count_within_5cm": support_point_count,
        "inlier_xy_extent_m": extent.tolist(),
        "search_radius_m": radius,
        "normal": normal.tolist(),
        "offset_m": float(offset_m),
        "ground_z_at_focus_m": ground_z,
        "camera_height_above_ground_m": float(camera_height),
        "inlier_residual_p95_m": float(np.quantile(residuals, 0.95)),
    }
