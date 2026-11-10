#!/usr/bin/env python3
"""Derive a scale-normalized target ROI from a metric COLMAP text model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from estimate_sparse_floor import estimate_sparse_floor


DENSE_COMPONENT_MIN_NEIGHBORS = 8
DENSE_COMPONENT_MIN_FRACTION = 0.75
ROI_PADDING_M = 0.15
HEIGHT_THRESHOLDS_M = (0.08, 0.12, 0.18, 0.25)
GROUND_HEIGHT_BIN_M = 0.02
GROUND_MODE_SUPPORT_CELL_M = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_images(path: Path) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    directions = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) < 10 or index + 1 >= len(lines):
            raise RuntimeError("Malformed COLMAP images.txt")
        qw, qx, qy, qz = map(float, fields[1:5])
        translation = np.asarray(list(map(float, fields[5:8])))
        world_to_camera = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        centers.append(-world_to_camera.T @ translation)
        directions.append(world_to_camera.T[:, 2])
        index += 2
    return np.asarray(centers), np.asarray(directions)


def load_points(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            points.append([float(value) for value in fields[1:4]])
    result = np.asarray(points)
    if len(result) < 10000 or not np.isfinite(result).all():
        raise RuntimeError("Metric sparse model is insufficient")
    return result


def solve_focus(centers: np.ndarray, directions: np.ndarray):
    weights = np.ones(len(centers))
    identity = np.eye(3)
    focus = centers.mean(axis=0)
    for iteration in range(1, 21):
        projectors = identity[None] - directions[:, :, None] * directions[:, None, :]
        matrix = np.sum(weights[:, None, None] * projectors, axis=0)
        vector = np.sum(weights[:, None] * np.einsum("nij,nj->ni", projectors, centers), axis=0)
        focus = np.linalg.solve(matrix, vector)
        residuals = np.linalg.norm(np.cross(focus[None] - centers, directions), axis=1)
        weights = np.minimum(1.0, 0.35 / np.maximum(residuals, 1e-12))
    return focus, residuals, iteration


def select_ground_height_mode(
    local_ground: np.ndarray,
    focus_xy: np.ndarray,
    search_radius_m: float,
) -> tuple[float, dict]:
    """Resolve count-ambiguous low modes using near-target spatial coverage."""
    low = float(np.floor(local_ground[:, 2].min() / GROUND_HEIGHT_BIN_M) * GROUND_HEIGHT_BIN_M)
    high = float(np.ceil(local_ground[:, 2].max() / GROUND_HEIGHT_BIN_M) * GROUND_HEIGHT_BIN_M)
    counts, edges = np.histogram(
        local_ground[:, 2],
        bins=np.arange(low, high + GROUND_HEIGHT_BIN_M, GROUND_HEIGHT_BIN_M),
    )
    maximum_count = int(np.max(counts))
    if maximum_count <= 0:
        raise RuntimeError("No populated sparse ground-height mode")
    # Sparse reconstruction can move a small number of points between modes.
    ambiguity_band = 2.0 * np.sqrt(maximum_count)
    candidates = np.flatnonzero(counts >= maximum_count - ambiguity_band)
    inner_radius = search_radius_m / 2.0
    inner = local_ground[
        np.linalg.norm(local_ground[:, :2] - focus_xy[None, :], axis=1) <= inner_radius
    ]
    rows = []
    for index in candidates:
        inside = inner[
            (inner[:, 2] >= edges[index]) & (inner[:, 2] < edges[index + 1])
        ]
        cells = np.unique(
            np.floor(inside[:, :2] / GROUND_MODE_SUPPORT_CELL_M).astype(np.int64),
            axis=0,
        )
        rows.append({
            "bin_index": int(index),
            "center_z_m": float((edges[index] + edges[index + 1]) / 2.0),
            "local_point_count": int(counts[index]),
            "inner_support_point_count": len(inside),
            "inner_occupied_cell_count": len(cells),
        })
    selected = max(
        rows,
        key=lambda row: (
            row["inner_occupied_cell_count"],
            row["local_point_count"],
            -row["center_z_m"],
        ),
    )
    return selected["center_z_m"], {
        "method": "near_target_spatial_coverage_for_count_ambiguous_height_modes",
        "height_bin_m": GROUND_HEIGHT_BIN_M,
        "maximum_bin_count": maximum_count,
        "count_ambiguity_band_points": float(ambiguity_band),
        "inner_radius_m": inner_radius,
        "support_cell_m": GROUND_MODE_SUPPORT_CELL_M,
        "candidates": rows,
        "selected_bin_index": selected["bin_index"],
    }


def largest_dense_component(
    points: np.ndarray, focus_xy: np.ndarray, *, defer_fraction_gate: bool = False
) -> tuple[np.ndarray, dict]:
    if len(points) < DENSE_COMPONENT_MIN_NEIGHBORS:
        raise RuntimeError("Too few points for a dense non-ground component")
    tree = cKDTree(points)
    neighbor_distances = tree.query(points, k=DENSE_COMPONENT_MIN_NEIGHBORS, workers=1)[0][:, -1]
    density_p50, density_p90 = np.quantile(neighbor_distances, [0.50, 0.90])
    epsilon = float(min(2.0 * density_p90, 8.0 * density_p50))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError("Non-ground point spacing cannot define a density neighborhood")
    neighbors = tree.query_ball_point(points, r=epsilon)
    core = np.asarray(
        [len(indices) >= DENSE_COMPONENT_MIN_NEIGHBORS for indices in neighbors],
        dtype=bool,
    )
    labels = np.full(len(points), -1, dtype=np.int32)
    component_id = 0
    for seed in range(len(points)):
        if not core[seed] or labels[seed] >= 0:
            continue
        labels[seed] = component_id
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            if not core[current]:
                continue
            for neighbor in neighbors[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component_id
                    if core[neighbor]:
                        queue.append(neighbor)
        component_id += 1
    if component_id == 0:
        raise RuntimeError("No dense non-ground component supports the target ROI")
    counts = np.bincount(labels[labels >= 0], minlength=component_id)
    selected_id = int(np.argmax(counts))
    selected = points[labels == selected_id]
    fraction = len(selected) / len(points)
    runner_up = int(np.partition(counts, -2)[-2]) if len(counts) > 1 else 0
    dominance = float(len(selected) / runner_up) if runner_up else None
    if len(selected) < 1000 or (
        fraction < DENSE_COMPONENT_MIN_FRACTION and not defer_fraction_gate
    ):
        raise RuntimeError(
            "No dominant non-ground component supports the target ROI: "
            f"largest={len(selected)}/{len(points)}, runner_up={runner_up}"
        )
    lower = np.quantile(selected[:, :2], 0.01, axis=0) - ROI_PADDING_M
    upper = np.quantile(selected[:, :2], 0.99, axis=0) + ROI_PADDING_M
    if not np.all((focus_xy >= lower) & (focus_xy <= upper)):
        raise RuntimeError("Dominant non-ground component does not contain the optical focus")
    return selected, {
        "algorithm": "deterministic adaptive 3D density component with cKDTree",
        "epsilon_m": epsilon,
        "eighth_neighbor_distance_p50_m": float(density_p50),
        "eighth_neighbor_distance_p90_m": float(density_p90),
        "minimum_neighbors": DENSE_COMPONENT_MIN_NEIGHBORS,
        "minimum_fraction": DENSE_COMPONENT_MIN_FRACTION,
        "component_count": component_id,
        "noise_count": int(np.count_nonzero(labels < 0)),
        "selected_component_id": selected_id,
        "selected_count": len(selected),
        "selected_fraction": fraction,
        "runner_up_count": runner_up,
        "dominance_over_runner_up": dominance,
    }


def select_roi_support(
    points: np.ndarray,
    focus_xy: np.ndarray,
    median_depth: float,
    *,
    defer_fraction_gate: bool = False,
) -> tuple[np.ndarray, dict]:
    broad_lower = np.quantile(points[:, :2], 0.01, axis=0) - ROI_PADDING_M
    broad_upper = np.quantile(points[:, :2], 0.99, axis=0) + ROI_PADDING_M
    broad_extent = broad_upper - broad_lower
    component_points, component = largest_dense_component(
        points, focus_xy, defer_fraction_gate=defer_fraction_gate
    )
    component_lower = np.quantile(component_points[:, :2], 0.01, axis=0) - ROI_PADDING_M
    component_upper = np.quantile(component_points[:, :2], 0.99, axis=0) + ROI_PADDING_M
    overhang = np.maximum.reduce(
        [component_lower - broad_lower, broad_upper - component_upper, np.zeros(2)]
    )
    use_all = bool(np.all(overhang <= ROI_PADDING_M))
    selected = points if use_all else component_points
    lower = broad_lower if use_all else component_lower
    upper = broad_upper if use_all else component_upper
    selection = {
        "mode": "all_non_ground_quantiles" if use_all else "dominant_adaptive_3d_component",
        "broad_candidate_count": len(points),
        "broad_horizontal_extent_m": broad_extent.tolist(),
        "broad_overhang_beyond_dense_support_m": overhang.tolist(),
        "component": component,
        "selected_horizontal_extent_m": (upper - lower).tolist(),
        "selected_extent_over_median_depth_diagnostic": ((upper - lower) / median_depth).tolist(),
    }
    return selected, selection


def select_height_consistent_roi_support(
    points: np.ndarray,
    focus_xy: np.ndarray,
    ground_z: float,
    search_radius: float,
    median_depth: float,
) -> tuple[np.ndarray, dict, list[dict], float, float, float, dict]:
    selected_candidates = []
    selections = []
    bounds = []
    for index, threshold in enumerate(HEIGHT_THRESHOLDS_M):
        candidates = points[
            (points[:, 2] > ground_z + threshold)
            & (points[:, 2] < ground_z + 2.20)
            & (np.linalg.norm(points[:, :2] - focus_xy[None], axis=1) < search_radius)
        ]
        if len(candidates) < 1000:
            raise RuntimeError(
                f"Too few non-ground points at threshold {threshold}: {len(candidates)}"
            )
        selected, selection = select_roi_support(
            candidates,
            focus_xy,
            median_depth,
            defer_fraction_gate=index == 0,
        )
        selected_candidates.append(selected)
        selections.append(selection)
        bounds.append(
            np.concatenate(
                (
                    np.quantile(selected[:, :2], 0.01, axis=0),
                    np.quantile(selected[:, :2], 0.99, axis=0),
                )
            )
        )
    bound_array = np.stack(bounds)
    bound_drift = float(np.max(np.ptp(bound_array, axis=0)))
    core_bound_drift = float(np.max(np.ptp(bound_array[1:], axis=0)))
    # A low box can disappear at a higher clearance without making the low ROI invalid.
    lower_expansion = np.maximum(bounds[0][:2] - bound_array[:, :2], 0.0)
    upper_expansion = np.maximum(bound_array[:, 2:] - bounds[0][2:], 0.0)
    bound_expansion = float(max(np.max(lower_expansion), np.max(upper_expansion)))
    if bound_expansion > 0.20:
        raise RuntimeError(
            f"Higher-clearance ROI expands outside low-clearance support: {bound_expansion:.6f} m"
        )
    low_tree = cKDTree(selected_candidates[0])
    low_coverage = float(
        np.mean(low_tree.query(selected_candidates[1], k=1, workers=1)[0] < 1e-8)
    )
    if low_coverage < 0.98:
        raise RuntimeError(f"Low-clearance target support is inconsistent: {low_coverage:.6f}")
    low_selection = selections[0]
    low_component = low_selection["component"]
    low_fraction = float(low_component["selected_fraction"])
    if low_fraction < DENSE_COMPONENT_MIN_FRACTION:
        dominance = low_component["dominance_over_runner_up"]
        if (
            low_selection["mode"] != "dominant_adaptive_3d_component"
            or low_fraction <= 0.50
            or (dominance is not None and dominance < 2.0)
            or bound_expansion > 0.20
        ):
            raise RuntimeError(
                "Low-clearance component remains ambiguous after height-consistency checks: "
                f"fraction={low_fraction:.6f}, dominance={dominance}, "
                f"bound_expansion={bound_expansion:.6f} m"
            )
        admission = {
            "mode": "majority_component_with_stable_higher_clearance_core",
            "minimum_low_fraction": 0.50,
            "minimum_dominance_over_runner_up": 2.0,
            "maximum_all_height_bound_expansion_m": 0.20,
            "minimum_core_coverage_by_low_component": 0.98,
            "low_fraction": low_fraction,
            "dominance_over_runner_up": dominance,
            "all_height_bound_drift_m": bound_drift,
            "all_height_bound_expansion_m": bound_expansion,
            "core_coverage_by_low_component": low_coverage,
        }
    else:
        admission = {
            "mode": "ordinary_dense_fraction",
            "low_fraction": low_fraction,
            "all_height_bound_expansion_m": bound_expansion,
        }
    return (
        selected_candidates[0],
        low_selection,
        selections,
        bound_drift,
        core_bound_drift,
        low_coverage,
        admission,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-sparse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sparse = args.metric_sparse.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    images_path = sparse / "images.txt"
    points_path = sparse / "points3D.txt"
    centers, directions = load_images(images_path)
    points = load_points(points_path)
    focus, residuals, iterations = solve_focus(centers, directions)
    forward_depths = np.sum((focus[None] - centers) * directions, axis=1)
    if np.median(forward_depths) <= 0:
        directions = -directions
        focus, residuals, iterations = solve_focus(centers, directions)
        forward_depths = np.sum((focus[None] - centers) * directions, axis=1)
    if np.count_nonzero(forward_depths > 0) < int(np.ceil(0.95 * len(centers))):
        raise RuntimeError("Robust focus is not in front of enough cameras")
    median_depth = float(np.median(forward_depths))
    median_camera_z = float(np.median(centers[:, 2]))
    search_radius = float(np.clip(0.75 * median_depth, 2.5, 5.0))
    local_ground = points[
        (points[:, 2] > median_camera_z - 2.0)
        & (points[:, 2] < median_camera_z - 0.35)
        & (np.linalg.norm(points[:, :2] - focus[None, :2], axis=1) < search_radius)
    ]
    if len(local_ground) < 1000:
        raise RuntimeError("Too few local points for the ground-height estimate")
    ground_z, ground_mode_selection = select_ground_height_mode(
        local_ground, focus[:2], search_radius
    )
    sparse_floor = estimate_sparse_floor(
        points, focus[:2], median_camera_z, search_radius
    )
    mode_height = ground_z
    floor_height = float(sparse_floor["ground_z_at_focus_m"])
    height_difference = mode_height - floor_height
    if height_difference < -0.15:
        raise RuntimeError(
            "Sparse height mode lies below the spatial floor plane: "
            f"difference={height_difference:.6f} m"
        )
    ground_selection = "spatial_floor_override" if height_difference > 0.15 else "height_mode_agrees_with_spatial_floor"
    if ground_selection == "spatial_floor_override":
        ground_z = floor_height
    (
        selected,
        roi_selection,
        selections,
        bound_drift,
        core_bound_drift,
        low_coverage,
        low_component_admission,
    ) = select_height_consistent_roi_support(
        points, focus[:2], ground_z, search_radius, median_depth
    )
    lower_xy = np.quantile(selected[:, :2], 0.01, axis=0) - ROI_PADDING_M
    upper_xy = np.quantile(selected[:, :2], 0.99, axis=0) + ROI_PADDING_M
    top = float(np.quantile(selected[:, 2], 0.99))
    minimum_z = ground_z - 0.15
    if ground_selection == "spatial_floor_override":
        corners = np.asarray([
            [lower_xy[0], lower_xy[1]], [lower_xy[0], upper_xy[1]],
            [upper_xy[0], lower_xy[1]], [upper_xy[0], upper_xy[1]],
        ])
        floor_normal = np.asarray(sparse_floor["normal"])
        floor_corners = -(
            float(sparse_floor["offset_m"]) + corners @ floor_normal[:2]
        ) / floor_normal[2]
        minimum_z = float(np.min(floor_corners) - 0.15)
    minimum = np.asarray([lower_xy[0], lower_xy[1], minimum_z])
    maximum = np.asarray([upper_xy[0], upper_xy[1], min(ground_z + 2.20, max(focus[2] + 0.35, top + 0.15))])
    if not np.all(maximum > minimum) or not np.all((focus >= minimum) & (focus <= maximum)):
        raise RuntimeError("Derived vehicle ROI is invalid")
    receipt = {
        "schema_version": "e26_scale_normalized_vehicle_roi_v1",
        "status": "complete",
        "metric_sparse_model": str(sparse),
        "metric_sparse_model_files": {
            "images_sha256": sha256_file(images_path),
            "points3D_sha256": sha256_file(points_path),
        },
        "roi_derivation": {
            "method": "robust optical-axis focus, sparse-height ground mode, and adaptive density support",
            "camera_count": len(centers),
            "focus_world_lio_m": focus.tolist(),
            "axis_residual_m": {
                "p50": float(np.quantile(residuals, 0.50)),
                "p95": float(np.quantile(residuals, 0.95)),
                "maximum": float(np.max(residuals)),
            },
            "forward_depth_m": {
                "positive_count": int(np.count_nonzero(forward_depths > 0)),
                "p50": median_depth,
                "p05": float(np.quantile(forward_depths, 0.05)),
                "p95": float(np.quantile(forward_depths, 0.95)),
            },
            "median_camera_z_m": median_camera_z,
            "ground_height_mode_selection": ground_mode_selection,
            "spatial_sparse_floor_plane": sparse_floor,
            "ground_estimate_selection": {
                "mode": ground_selection,
                "height_mode_z_m": mode_height,
                "spatial_floor_z_at_focus_m": floor_height,
                "height_mode_above_spatial_floor_m": height_difference,
            },
            "roi_support_selection": roi_selection,
            "roi_support_selection_by_height_threshold": selections,
            "low_clearance_component_admission": low_component_admission,
            # Keep this field name for the existing downstream receipt contract.
            "metric_sparse_vehicle_support": {
                "search_radius_m": search_radius,
                "ground_z_m": ground_z,
                "ground_support_count": (
                    sparse_floor["support_point_count_within_5cm"]
                    if ground_selection == "spatial_floor_override" else len(local_ground)
                ),
                "candidate_count": len(selected),
                "height_thresholds_checked_m": HEIGHT_THRESHOLDS_M,
                "maximum_xy_bound_drift_across_height_thresholds_m": bound_drift,
                "maximum_xy_core_bound_drift_m": core_bound_drift,
                "maximum_xy_bound_expansion_across_height_thresholds_m": (
                    low_component_admission["all_height_bound_expansion_m"]
                ),
                "low_clearance_coverage_of_core": low_coverage,
                "quantile_minimum_xyz_m": np.quantile(selected, 0.01, axis=0).tolist(),
                "quantile_maximum_xyz_m": np.quantile(selected, 0.99, axis=0).tolist(),
            },
        },
        "roi_world_lio_m": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "language_model_used": False,
        "semantic_model_used": False,
        "manual_selection_used": False,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
