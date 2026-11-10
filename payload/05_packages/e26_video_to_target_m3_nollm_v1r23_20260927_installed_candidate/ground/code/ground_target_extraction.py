"""Extract a ground-removed target-ROI mesh with deterministic contact protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GroundFitPolicy:
    low_centroid_quantile: float
    candidate_triangle_maximum_tilt_degrees: float
    distance_threshold_m: float
    iterations: int
    random_seed: int
    maximum_accepted_plane_tilt_degrees: float
    minimum_accepted_inlier_fraction: float
    minimum_accepted_inlier_xy_extent_m: float
    minimum_accepted_inlier_roi_span_fraction: float
    maximum_ground_roi_bottom_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.low_centroid_quantile < 1.0:
            raise ValueError("Ground-candidate quantile is invalid")
        if not 0.0 < self.minimum_accepted_inlier_fraction <= 1.0:
            raise ValueError("Ground inlier fraction is invalid")
        if not 0.0 < self.minimum_accepted_inlier_roi_span_fraction <= 1.0:
            raise ValueError("Ground ROI span fraction is invalid")
        if not 0.0 < self.maximum_ground_roi_bottom_fraction <= 1.0:
            raise ValueError("Ground ROI bottom fraction is invalid")
        if self.distance_threshold_m <= 0.0 or self.iterations <= 0:
            raise ValueError("Ground RANSAC parameters are invalid")
        if self.minimum_accepted_inlier_xy_extent_m <= 0.0:
            raise ValueError("Ground minimum XY extent is invalid")


@dataclass(frozen=True)
class GroundClassificationPolicy:
    maximum_signed_height_m: float
    minimum_normal_cosine: float
    unconditional_low_artifact_maximum_height_m: float
    contact_band_minimum_m: float
    contact_band_maximum_m: float
    vertical_support_minimum_m: float
    vertical_support_maximum_m: float
    vertical_normal_cosine: float
    ground_grid_cell_m: float
    support_dilation_cells: int


@dataclass(frozen=True)
class TargetComponentPolicy:
    anchor_minimum_height_m: float
    retain_contact_supported_low_components: bool
    reject_low_only_unsupported_components: bool
    require_dominant_component_envelope: bool
    dominant_component_minimum_area_fraction: float
    dominant_envelope_lower_quantile: float
    dominant_envelope_upper_quantile: float
    dominant_envelope_margin_m: float
    minimum_inside_triangle_fraction: float
    require_dominant_surface_proximity: bool
    dominant_surface_voxel_m: float
    maximum_dominant_surface_distance_m: float
    minimum_proximal_triangle_fraction: float
    proximity_gate_maximum_component_height_m: float

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.dominant_envelope_lower_quantile
            < self.dominant_envelope_upper_quantile
            <= 1.0
        ):
            raise ValueError("Dominant-envelope quantiles are invalid")
        if not 0.0 < self.dominant_component_minimum_area_fraction <= 1.0:
            raise ValueError("Dominant-component area fraction is invalid")
        if self.dominant_envelope_margin_m < 0.0:
            raise ValueError("Dominant-envelope margin must be non-negative")
        if not 0.0 < self.minimum_inside_triangle_fraction <= 1.0:
            raise ValueError("Envelope inside-triangle fraction is invalid")
        if self.dominant_surface_voxel_m <= 0.0:
            raise ValueError("Dominant-surface voxel size must be positive")
        if self.maximum_dominant_surface_distance_m <= 0.0:
            raise ValueError("Dominant-surface distance must be positive")
        if not 0.0 < self.minimum_proximal_triangle_fraction <= 1.0:
            raise ValueError("Dominant-surface proximal fraction is invalid")
        if self.proximity_gate_maximum_component_height_m <= 0.0:
            raise ValueError("Proximity-gate maximum height must be positive")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(normal, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(seed @ unit)) > 0.85:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = seed - float(seed @ unit) * unit
    axis_u /= np.linalg.norm(axis_u)
    return axis_u, np.cross(unit, axis_u)


def load_policies(config: dict[str, Any]) -> tuple[
    GroundFitPolicy, GroundClassificationPolicy, TargetComponentPolicy
]:
    return (
        GroundFitPolicy(**config["ground_fit"]),
        GroundClassificationPolicy(**config["ground_classification"]),
        TargetComponentPolicy(**config["target_components"]),
    )


def validate_no_llm_input(tsdf_receipt: dict[str, Any]) -> dict[str, str]:
    """Reject known model or mask use and disclose legacy provenance gaps."""
    required_false = (
        "language_model_used",
        "semantic_model_used",
        "object_mask_used",
    )
    optional_false = (
        "visual_language_model_used",
        "manual_visual_decision_used",
        "known_box_dimension_used",
    )
    violations = [
        name
        for name in required_false + optional_false
        if name in tsdf_receipt and tsdf_receipt[name] is not False
    ]
    if violations:
        raise RuntimeError(
            "Upstream TSDF receipt violates the machine-only contract: "
            + ", ".join(violations)
        )
    if "geometry_mask_usage" in tsdf_receipt and tsdf_receipt["geometry_mask_usage"] != "none":
        raise RuntimeError("TSDF receipt reports geometry-mask use")
    if "known_box_dimension_usage" in tsdf_receipt and tsdf_receipt["known_box_dimension_usage"] != "none":
        raise RuntimeError("TSDF receipt reports box-dimension use")
    if all(tsdf_receipt.get(name) is False for name in required_false):
        return {"status": "explicit_machine_only_flags"}
    if tsdf_receipt.get("schema_version") == "e26_pgsr_streaming_roi_tsdf_v3":
        if tsdf_receipt.get("geometry_mask_usage") != "none":
            raise RuntimeError("Legacy TSDF receipt lacks geometry-mask provenance")
        if tsdf_receipt.get("known_box_dimension_usage") != "none":
            raise RuntimeError("Legacy TSDF receipt lacks box-dimension provenance")
        return {
            "status": "legacy_geometry_only_upstream_llm_unconfirmed",
            "reason": "Legacy S07 receipt has no explicit language-model provenance fields",
        }
    raise RuntimeError(
        "Upstream TSDF receipt lacks the machine-only contract: "
        + ", ".join(name for name in required_false if name not in tsdf_receipt)
    )


def validate_mesh_receipt(
    mesh_path: Path,
    mesh: o3d.geometry.TriangleMesh,
    tsdf_receipt: dict[str, Any],
) -> str:
    """Bind the current mesh to the supplied S07 receipt before extraction."""
    declared = tsdf_receipt.get("mesh")
    if not isinstance(declared, dict):
        raise RuntimeError("TSDF receipt has no mesh identity")
    if len(mesh.vertices) != declared.get("vertices"):
        raise RuntimeError("Mesh vertex count disagrees with TSDF receipt")
    if len(mesh.triangles) != declared.get("triangles"):
        raise RuntimeError("Mesh triangle count disagrees with TSDF receipt")
    digest = sha256_file(mesh_path)
    if digest != declared.get("sha256"):
        raise RuntimeError("Mesh SHA-256 disagrees with TSDF receipt")
    return digest


PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M = 0.05
MINIMUM_SPARSE_GROUND_PRIOR_SUPPORT = 1000
MAXIMUM_UNCONSTRAINED_GROUND_PRIOR_DIFFERENCE_M = 0.15
MAXIMUM_SPATIAL_FLOOR_MESH_DIFFERENCE_M = 0.08
MINIMUM_SPATIAL_FLOOR_INLIER_CELLS = 60
MINIMUM_SPATIAL_FLOOR_INLIER_FRACTION = 0.25


def _fit_ground_plane_once(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    tsdf_receipt: dict[str, Any],
    policy: GroundFitPolicy,
    prior_ground_z_m: float | None,
    raw_mcap: Path | None = None,
    corroborated_minimum_inlier_fraction: float | None = None,
    prior_ground_plane: dict[str, Any] | None = None,
    registered_scan_receipt: Path | None = None,
) -> dict[str, Any]:
    centroids = vertices[triangles].mean(axis=1)
    low_cutoff = float(np.quantile(centroids[:, 2], policy.low_centroid_quantile))
    horizontal = np.abs(triangle_normals[:, 2]) >= np.cos(
        np.deg2rad(policy.candidate_triangle_maximum_tilt_degrees)
    )
    candidate_triangles = (centroids[:, 2] <= low_cutoff) & horizontal
    if prior_ground_plane is not None:
        prior_normal = np.asarray(prior_ground_plane["normal"], dtype=np.float64)
        prior_offset = float(prior_ground_plane["offset_m"])
        candidate_triangles &= (
            np.abs(centroids @ prior_normal + prior_offset)
            <= PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M
        )
    elif prior_ground_z_m is not None:
        candidate_triangles &= (
            np.abs(centroids[:, 2] - prior_ground_z_m)
            <= PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M
        )
    candidate_vertices = np.unique(triangles[candidate_triangles].reshape(-1))
    if len(candidate_vertices) < 1000:
        raise RuntimeError(
            f"Too few low horizontal ground candidates: {len(candidate_vertices)}"
        )

    candidate_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(vertices[candidate_vertices])
    )
    o3d.utility.random.seed(policy.random_seed)
    model, inliers = candidate_cloud.segment_plane(
        distance_threshold=policy.distance_threshold_m,
        ransac_n=3,
        num_iterations=policy.iterations,
        probability=0.999999,
    )
    normal = np.asarray(model[:3], dtype=np.float64)
    offset = float(model[3])
    magnitude = float(np.linalg.norm(normal))
    normal /= magnitude
    offset /= magnitude
    if normal[2] < 0:
        normal = -normal
        offset = -offset

    inlier_points = np.asarray(candidate_cloud.points)[np.asarray(inliers, dtype=np.int64)]
    inlier_fraction = len(inliers) / len(candidate_vertices)
    inlier_xy_extent = np.ptp(inlier_points[:, :2], axis=0)
    maximum_tilt_cosine = float(
        np.cos(np.deg2rad(policy.maximum_accepted_plane_tilt_degrees))
    )
    if normal[2] < maximum_tilt_cosine:
        raise RuntimeError(
            f"Fitted low plane is not gravity aligned: normal_z={normal[2]:.6f}"
        )
    below_mesh_dominance_gate = inlier_fraction < policy.minimum_accepted_inlier_fraction
    if below_mesh_dominance_gate and (
        prior_ground_z_m is None
        or raw_mcap is None
        or corroborated_minimum_inlier_fraction is None
        or inlier_fraction < corroborated_minimum_inlier_fraction
    ):
        raise RuntimeError(
            f"Fitted low plane is not dominant: inlier_fraction={inlier_fraction:.6f}"
        )
    below_mesh_absolute_gate = bool(np.any(
        inlier_xy_extent < policy.minimum_accepted_inlier_xy_extent_m
    ))
    if below_mesh_absolute_gate and (prior_ground_z_m is None or raw_mcap is None):
        raise RuntimeError(
            "Fitted low plane lacks horizontal support: "
            f"xy_extent_m={inlier_xy_extent.tolist()}"
        )

    roi_minimum = np.asarray(
        tsdf_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64
    )
    roi_maximum = np.asarray(
        tsdf_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64
    )
    if (
        roi_minimum.shape != (3,)
        or roi_maximum.shape != (3,)
        or not np.isfinite(roi_minimum).all()
        or not np.isfinite(roi_maximum).all()
        or np.any(roi_maximum <= roi_minimum)
    ):
        raise RuntimeError("TSDF receipt has invalid metric ROI bounds")
    roi_extent = roi_maximum - roi_minimum
    roi_span_fraction = inlier_xy_extent / roi_extent[:2]
    roi_center = (roi_minimum + roi_maximum) / 2.0
    ground_z_at_roi_center = float(
        -(offset + normal[0] * roi_center[0] + normal[1] * roi_center[1])
        / normal[2]
    )
    ground_roi_bottom_fraction = float(
        (ground_z_at_roi_center - roi_minimum[2]) / roi_extent[2]
    )
    if not (
        0.0 <= ground_roi_bottom_fraction
        <= policy.maximum_ground_roi_bottom_fraction
    ):
        raise RuntimeError(
            "Fitted ground is outside the lower metric ROI band: "
            f"fraction={ground_roi_bottom_fraction:.6f}"
        )
    if prior_ground_z_m is not None and abs(
        ground_z_at_roi_center - prior_ground_z_m
    ) > PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M:
        raise RuntimeError("Prior-guided ground plane shifted outside its height window")

    registered_lidar_support = None
    below_mesh_span_gate = bool(np.any(
        roi_span_fraction < policy.minimum_accepted_inlier_roi_span_fraction
    ))
    if below_mesh_dominance_gate or below_mesh_absolute_gate or below_mesh_span_gate:
        if prior_ground_z_m is None or raw_mcap is None:
            raise RuntimeError(
                "Fitted low plane does not span enough of the ROI: "
                f"xy_fraction={roi_span_fraction.tolist()}"
            )
        from registered_lidar_ground_support import validate_registered_lidar_support

        registered_lidar_support = validate_registered_lidar_support(
            raw_mcap, normal, offset, roi_minimum, roi_maximum, prior_ground_z_m,
            policy.minimum_accepted_inlier_xy_extent_m,
            policy.minimum_accepted_inlier_roi_span_fraction,
            PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M,
            registered_scan_receipt=registered_scan_receipt,
            expected_run_root=(Path(tsdf_receipt['source_model']).resolve(strict=True).parents[1]
                               if registered_scan_receipt is not None else None),
            expand_evidence=True,
        )

    residuals = np.abs(inlier_points @ normal + offset)
    return {
        "candidate_selection_mode": (
            "spatial_sparse_floor_plane_window" if prior_ground_plane is not None
            else "sparse_ground_prior_window" if prior_ground_z_m is not None
            else "unconstrained_low_mesh"
        ),
        "normal": normal,
        "offset_m": offset,
        "candidate_vertex_count": len(candidate_vertices),
        "inlier_count": len(inliers),
        "inlier_fraction": inlier_fraction,
        "inlier_fraction_acceptance": (
            "registered_lidar_corroborated" if below_mesh_dominance_gate
            else "mesh_dominance"
        ),
        "corroborated_minimum_inlier_fraction": corroborated_minimum_inlier_fraction,
        "mesh_absolute_extent_gate_passed": not below_mesh_absolute_gate,
        "mesh_roi_span_gate_passed": not below_mesh_span_gate,
        "inlier_xy_extent_m": inlier_xy_extent.tolist(),
        "inlier_roi_span_fraction": roi_span_fraction.tolist(),
        "inlier_residual_m": {
            "p50": float(np.quantile(residuals, 0.50)),
            "p95": float(np.quantile(residuals, 0.95)),
            "maximum": float(np.max(residuals)),
        },
        "ground_z_at_roi_center_m": ground_z_at_roi_center,
        "ground_roi_bottom_fraction": ground_roi_bottom_fraction,
        "registered_lidar_spatial_support": registered_lidar_support,
    }


def fit_ground_plane(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    tsdf_receipt: dict[str, Any],
    policy: GroundFitPolicy,
    raw_mcap: Path | None = None,
    corroborated_minimum_inlier_fraction: float | None = None,
    registered_scan_receipt: Path | None = None,
) -> dict[str, Any]:
    registered_binding = None
    if registered_scan_receipt is not None:
        if raw_mcap is None:
            raise ValueError('Generated ground scan support requires its original MCAP')
        from registered_lidar_ground_support import validate_generated_scan_source
        model = Path(tsdf_receipt['source_model']).resolve(strict=True)
        if model.name != 'model' or model.parent.name != 'pgsr':
            raise ValueError('Generated ground scans require the current native PGSR source model')
        _, registered_binding = validate_generated_scan_source(
            raw_mcap, registered_scan_receipt, model.parents[1])
    if corroborated_minimum_inlier_fraction is not None and not (
        0.0 < corroborated_minimum_inlier_fraction
        <= policy.minimum_accepted_inlier_fraction
    ):
        raise ValueError("Corroborated ground inlier fraction is invalid")
    sparse_support = tsdf_receipt.get("roi_derivation", {}).get(
        "metric_sparse_vehicle_support", {}
    )
    prior_z_value = sparse_support.get("ground_z_m")
    prior_support = int(sparse_support.get("ground_support_count", 0))
    prior_z = float(prior_z_value) if prior_z_value is not None else None
    spatial_floor = tsdf_receipt.get("roi_derivation", {}).get(
        "spatial_sparse_floor_plane"
    )
    valid_spatial_floor = bool(
        isinstance(spatial_floor, dict)
        and int(spatial_floor.get("inlier_cell_count", 0)) >= MINIMUM_SPATIAL_FLOOR_INLIER_CELLS
        and float(spatial_floor.get("inlier_cell_fraction", 0))
        >= MINIMUM_SPATIAL_FLOOR_INLIER_FRACTION
    )
    if valid_spatial_floor:
        spatial_z = float(spatial_floor["ground_z_at_focus_m"])
        spatial_normal = np.asarray(spatial_floor["normal"], dtype=np.float64)
        spatial_offset = float(spatial_floor["offset_m"])
        if (
            not np.isfinite(spatial_z)
            or spatial_normal.shape != (3,)
            or not np.isfinite(spatial_normal).all()
            or not np.isclose(np.linalg.norm(spatial_normal), 1.0, atol=1e-3)
            or spatial_normal[2] <= 0
            or not np.isfinite(spatial_offset)
        ):
            raise RuntimeError("Spatial sparse floor receipt has an invalid plane")
        prior_z = spatial_z
        prior_support = int(spatial_floor["support_point_count_within_5cm"])
    valid_prior = (
        prior_z is not None
        and np.isfinite(prior_z)
        and (
            valid_spatial_floor
            or prior_support >= MINIMUM_SPARSE_GROUND_PRIOR_SUPPORT
        )
    )
    maximum_difference = (
        MAXIMUM_SPATIAL_FLOOR_MESH_DIFFERENCE_M if valid_spatial_floor
        else MAXIMUM_UNCONSTRAINED_GROUND_PRIOR_DIFFERENCE_M
    )
    try:
        fit = _fit_ground_plane_once(
            vertices, triangles, triangle_normals, tsdf_receipt, policy, None
        )
        if valid_prior and abs(fit["ground_z_at_roi_center_m"] - prior_z) > maximum_difference:
            raise RuntimeError("Unconstrained mesh plane disagrees with sparse ground height")
        if valid_spatial_floor:
            fit["spatial_sparse_floor_difference_m"] = abs(
                fit["ground_z_at_roi_center_m"] - prior_z
            )
        if registered_binding is not None:
            fit['provided_registered_scan_source'] = registered_binding
        return fit
    except RuntimeError as original_error:
        if not valid_prior:
            raise
        try:
            fit = _fit_ground_plane_once(
                vertices, triangles, triangle_normals, tsdf_receipt, policy, prior_z,
                raw_mcap, corroborated_minimum_inlier_fraction,
                spatial_floor if valid_spatial_floor else None,
                registered_scan_receipt,
            )
        except RuntimeError as fallback_error:
            raise RuntimeError(
                f"Ground plane failed both unconstrained and prior-guided fits: "
                f"unconstrained={original_error}; prior_guided={fallback_error}"
            ) from fallback_error
        difference = abs(fit["ground_z_at_roi_center_m"] - prior_z)
        if difference > PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M:
            raise RuntimeError(
                f"Prior-guided ground plane shifted outside its height window: {difference:.6f} m"
            )
        fit["independent_sparse_ground_z_m"] = prior_z
        fit["independent_sparse_ground_support_count"] = prior_support
        fit["ground_estimate_difference_m"] = difference
        fit["candidate_height_half_window_m"] = PRIOR_GROUND_CANDIDATE_HALF_WINDOW_M
        fit["unconstrained_rejection"] = str(original_error)
        if valid_spatial_floor:
            fit["spatial_sparse_floor_difference_m"] = difference
        if registered_binding is not None:
            fit['provided_registered_scan_source'] = registered_binding
        return fit


def build_vertical_support_grid(
    centroids: np.ndarray,
    signed_height: np.ndarray,
    normal_alignment: np.ndarray,
    normal: np.ndarray,
    policy: GroundClassificationPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    axis_u, axis_v = plane_basis(normal)
    coordinates = np.column_stack((centroids @ axis_u, centroids @ axis_v))
    keys = np.floor(coordinates / policy.ground_grid_cell_m).astype(np.int64)
    minimum_key = np.min(keys, axis=0)
    shifted = keys - minimum_key
    shape_array = np.max(shifted, axis=0) + 1
    shape = (int(shape_array[0]), int(shape_array[1]))
    vertical_seed = (
        (signed_height >= policy.vertical_support_minimum_m)
        & (signed_height <= policy.vertical_support_maximum_m)
        & (normal_alignment <= policy.vertical_normal_cosine)
    )
    support_grid = np.zeros(shape, dtype=bool)
    if np.any(vertical_seed):
        support_grid[shifted[vertical_seed, 0], shifted[vertical_seed, 1]] = True
    if policy.support_dilation_cells:
        width = 2 * policy.support_dilation_cells + 1
        support_grid = binary_dilation(
            support_grid, structure=np.ones((width, width), dtype=bool)
        )
    return support_grid, shifted, vertical_seed, shape


def classify_ground_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    normal: np.ndarray,
    offset_m: float,
    policy: GroundClassificationPolicy,
) -> dict[str, np.ndarray | tuple[int, int]]:
    centroids = vertices[triangles].mean(axis=1)
    centroid_height = centroids @ normal + offset_m
    vertex_height = vertices[triangles] @ normal + offset_m
    normal_alignment = np.abs(triangle_normals @ normal)
    support_grid, shifted, vertical_seed, shape = build_vertical_support_grid(
        centroids, centroid_height, normal_alignment, normal, policy
    )
    column_supported = support_grid[shifted[:, 0], shifted[:, 1]]
    in_contact_band = (
        (centroid_height >= policy.contact_band_minimum_m)
        & (centroid_height <= policy.contact_band_maximum_m)
    )
    non_horizontal = normal_alignment < policy.minimum_normal_cosine
    preserve_contact = in_contact_band & column_supported & non_horizontal
    confident_ground = (
        (centroid_height <= policy.maximum_signed_height_m)
        & (normal_alignment >= policy.minimum_normal_cosine)
    )
    unconditional_low_artifact = (
        np.max(vertex_height, axis=1)
        <= policy.unconditional_low_artifact_maximum_height_m
    )
    remove = (confident_ground | unconditional_low_artifact) & ~preserve_contact
    return {
        "remove": remove,
        "preserve_contact": preserve_contact,
        "confident_ground": confident_ground,
        "unconditional_low_artifact": unconditional_low_artifact,
        "vertical_seed": vertical_seed,
        "column_supported": column_supported,
        "centroid_height_m": centroid_height,
        "normal_alignment": normal_alignment,
        "grid_shape": shape,
    }


def component_keep_decisions(
    mesh: o3d.geometry.TriangleMesh,
    normal: np.ndarray,
    offset_m: float,
    classification: GroundClassificationPolicy,
    policy: TargetComponentPolicy,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    labels_raw, counts_raw, areas_raw = mesh.cluster_connected_triangles()
    labels = np.asarray(labels_raw, dtype=np.int64)
    counts = np.asarray(counts_raw, dtype=np.int64)
    areas = np.asarray(areas_raw, dtype=np.float64)
    component_count = len(counts)
    if component_count == 0:
        raise RuntimeError("Ground-removed mesh has no connected components")

    centroids = vertices[triangles].mean(axis=1)
    centroid_height = centroids @ normal + offset_m
    vertex_height = vertices[triangles] @ normal + offset_m
    normal_alignment = np.abs(normals @ normal)
    support_grid, shifted, _, _ = build_vertical_support_grid(
        centroids, centroid_height, normal_alignment, normal, classification
    )
    column_supported = support_grid[shifted[:, 0], shifted[:, 1]]
    non_horizontal = normal_alignment < classification.minimum_normal_cosine
    contact_evidence = (
        (centroid_height >= classification.contact_band_minimum_m)
        & (centroid_height <= policy.anchor_minimum_height_m)
        & column_supported
        & non_horizontal
    )

    height_max = np.full(component_count, -np.inf, dtype=np.float64)
    height_min = np.full(component_count, np.inf, dtype=np.float64)
    np.maximum.at(height_max, labels, np.max(vertex_height, axis=1))
    np.minimum.at(height_min, labels, np.min(vertex_height, axis=1))
    contact_count = np.bincount(
        labels, weights=contact_evidence.astype(np.int64), minlength=component_count
    ).astype(np.int64)
    anchored = height_max >= policy.anchor_minimum_height_m
    contact_supported = contact_count > 0
    base_keep = anchored.copy()
    if policy.retain_contact_supported_low_components:
        base_keep |= contact_supported
    if not policy.reject_low_only_unsupported_components:
        base_keep[:] = True

    dominant_component = int(np.argmax(areas))
    total_area = float(np.sum(areas))
    dominant_area_fraction = float(areas[dominant_component] / total_area)
    if dominant_area_fraction < policy.dominant_component_minimum_area_fraction:
        raise RuntimeError(
            "No unambiguous dominant target component: "
            f"area_fraction={dominant_area_fraction:.6f}"
        )

    envelope_supported = np.ones(component_count, dtype=bool)
    envelope_inside_fraction = np.ones(component_count, dtype=np.float64)
    envelope_receipt: dict[str, Any] = {
        "applied": False,
        "dominant_component": dominant_component,
        "dominant_component_area_fraction": dominant_area_fraction,
    }
    if policy.require_dominant_component_envelope:
        axis_u, axis_v = plane_basis(normal)
        horizontal = np.column_stack((centroids @ axis_u, centroids @ axis_v))
        triangle_vertices = vertices[triangles]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        dominant_triangles = labels == dominant_component
        dominant_horizontal = horizontal[dominant_triangles]
        dominant_weights = triangle_areas[dominant_triangles]
        envelope_center = np.average(
            dominant_horizontal, axis=0, weights=dominant_weights
        )
        centered = dominant_horizontal - envelope_center
        covariance = (
            (centered * dominant_weights[:, None]).T @ centered
            / float(np.sum(dominant_weights))
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        local = (horizontal - envelope_center) @ principal_axes
        dominant_local = local[dominant_triangles]
        lower = np.quantile(
            dominant_local, policy.dominant_envelope_lower_quantile, axis=0
        )
        upper = np.quantile(
            dominant_local, policy.dominant_envelope_upper_quantile, axis=0
        )
        inside_envelope = np.all(
            (local >= lower - policy.dominant_envelope_margin_m)
            & (local <= upper + policy.dominant_envelope_margin_m),
            axis=1,
        )
        inside_count = np.bincount(
            labels,
            weights=inside_envelope.astype(np.int64),
            minlength=component_count,
        )
        envelope_inside_fraction = inside_count / np.maximum(counts, 1)
        envelope_supported = (
            envelope_inside_fraction >= policy.minimum_inside_triangle_fraction
        )
        envelope_supported[dominant_component] = True
        envelope_receipt = {
            "applied": True,
            "dominant_component": dominant_component,
            "dominant_component_area_fraction": dominant_area_fraction,
            "center_plane_coordinates_m": envelope_center.tolist(),
            "principal_axes_2d": principal_axes.tolist(),
            "robust_lower_m": lower.tolist(),
            "robust_upper_m": upper.tolist(),
            "robust_extent_m": (upper - lower).tolist(),
            "margin_m": policy.dominant_envelope_margin_m,
            "minimum_inside_triangle_fraction": (
                policy.minimum_inside_triangle_fraction
            ),
        }

    proximity_supported = np.ones(component_count, dtype=bool)
    proximity_fraction = np.ones(component_count, dtype=np.float64)
    proximity_receipt: dict[str, Any] = {"applied": False}
    if policy.require_dominant_surface_proximity:
        dominant_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(centroids[labels == dominant_component])
        )
        proximity_reference = np.asarray(
            dominant_cloud.voxel_down_sample(
                policy.dominant_surface_voxel_m
            ).points,
            dtype=np.float64,
        )
        if len(proximity_reference) == 0:
            raise RuntimeError("Dominant target surface is empty after proximity sampling")
        proximity_tree = cKDTree(proximity_reference)
        proximal_count = np.zeros(component_count, dtype=np.int64)
        chunk_size = 500_000
        for start in range(0, len(centroids), chunk_size):
            end = min(start + chunk_size, len(centroids))
            distance, _ = proximity_tree.query(
                centroids[start:end], k=1, workers=1
            )
            proximal_count += np.bincount(
                labels[start:end],
                weights=(
                    distance <= policy.maximum_dominant_surface_distance_m
                ).astype(np.int64),
                minlength=component_count,
            ).astype(np.int64)
        proximity_fraction = proximal_count / np.maximum(counts, 1)
        proximity_required = (
            height_max <= policy.proximity_gate_maximum_component_height_m
        )
        proximity_supported = ~proximity_required | (
            proximity_fraction >= policy.minimum_proximal_triangle_fraction
        )
        proximity_supported[dominant_component] = True
        proximity_receipt = {
            "applied": True,
            "voxel_m": policy.dominant_surface_voxel_m,
            "reference_point_count": int(len(proximity_reference)),
            "maximum_distance_m": policy.maximum_dominant_surface_distance_m,
            "minimum_proximal_triangle_fraction": (
                policy.minimum_proximal_triangle_fraction
            ),
            "maximum_gated_component_height_m": (
                policy.proximity_gate_maximum_component_height_m
            ),
            "low_component_count_subject_to_gate": int(
                np.count_nonzero(proximity_required)
            ),
        }

    keep_components = base_keep & envelope_supported & proximity_supported
    keep_components[dominant_component] = True
    if not np.any(keep_components):
        raise RuntimeError("Component policy rejected every observed surface")

    triangle_keep = keep_components[labels]
    decisions = []
    for index in range(component_count):
        if index == dominant_component:
            reason = "retain_dominant_target_component"
        elif not base_keep[index]:
            reason = "reject_low_only_without_vertical_support"
        elif not envelope_supported[index]:
            reason = "reject_outside_dominant_target_envelope"
        elif not proximity_supported[index]:
            reason = "reject_far_from_dominant_target_surface"
        elif anchored[index]:
            reason = "retain_above_ground_anchor_within_dominant_envelope"
        else:
            reason = "retain_contact_support_within_dominant_envelope"
        decisions.append(
            {
                "component": index,
                "decision": "retain" if keep_components[index] else "reject",
                "reason": reason,
                "triangles": int(counts[index]),
                "area_m2": float(areas[index]),
                "minimum_signed_height_m": float(height_min[index]),
                "maximum_signed_height_m": float(height_max[index]),
                "contact_supported_triangle_count": int(contact_count[index]),
                "dominant_envelope_inside_triangle_fraction": float(
                    envelope_inside_fraction[index]
                ),
                "dominant_surface_proximal_triangle_fraction": float(
                    proximity_fraction[index]
                ),
            }
        )
    summary = {
        "component_count": component_count,
        "retained_component_count": int(np.count_nonzero(keep_components)),
        "rejected_component_count": int(np.count_nonzero(~keep_components)),
        "retained_triangle_count": int(np.count_nonzero(triangle_keep)),
        "rejected_triangle_count": int(np.count_nonzero(~triangle_keep)),
        "retained_area_m2": float(np.sum(areas[keep_components])),
        "rejected_area_m2": float(np.sum(areas[~keep_components])),
        "above_ground_anchor_component_count": int(np.count_nonzero(anchored)),
        "contact_supported_low_component_count": int(
            np.count_nonzero(keep_components & ~anchored & contact_supported)
        ),
        "dominant_component": dominant_component,
        "dominant_component_area_fraction": dominant_area_fraction,
        "dominant_envelope": envelope_receipt,
        "dominant_surface_proximity": proximity_receipt,
        "envelope_rejected_component_count": int(
            np.count_nonzero(base_keep & ~envelope_supported)
        ),
        "envelope_rejected_triangle_count": int(
            np.sum(counts[base_keep & ~envelope_supported])
        ),
        "envelope_rejected_area_m2": float(
            np.sum(areas[base_keep & ~envelope_supported])
        ),
        "proximity_rejected_component_count": int(
            np.count_nonzero(base_keep & envelope_supported & ~proximity_supported)
        ),
        "proximity_rejected_triangle_count": int(
            np.sum(counts[base_keep & envelope_supported & ~proximity_supported])
        ),
        "proximity_rejected_area_m2": float(
            np.sum(areas[base_keep & envelope_supported & ~proximity_supported])
        ),
        "high_component_count_bypassing_proximity_gate": int(
            np.count_nonzero(
                base_keep
                & envelope_supported
                & (height_max > policy.proximity_gate_maximum_component_height_m)
            )
        ),
    }
    return triangle_keep, decisions, summary


def write_xyzrgb_f32_ply(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Sampled point cloud has invalid XYZ values")
    if cloud.has_colors():
        colors = np.rint(
            np.clip(np.asarray(cloud.colors), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 190, dtype=np.uint8)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    packed = np.empty(points.shape[0], dtype=dtype)
    packed["x"], packed["y"], packed["z"] = points.T
    packed["red"], packed["green"], packed["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment E26 ground-removed target ROI XYZRGB\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        packed.tofile(stream)


def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    temporary = path.with_name(path.name + ".partial.ply")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    written = o3d.io.write_triangle_mesh(
        str(temporary),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    if not written or not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("Failed to write target mesh")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    raw_mesh_path = args.raw_mesh.resolve(strict=True)
    tsdf_receipt_path = args.tsdf_receipt.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    log_path = output / "RUN_LOG.jsonl"
    append_event(log_path, {"stage": "initialize", "status": "running"})

    config = read_json(config_path)
    tsdf_receipt = read_json(tsdf_receipt_path)
    upstream_provenance = validate_no_llm_input(tsdf_receipt)
    fit_policy, classification_policy, component_policy = load_policies(config)
    sample_count = args.sample_count or int(config["preview"]["sample_count"])
    random_seed = int(config["preview"]["random_seed"])
    if sample_count <= 0:
        raise ValueError("sample count must be positive")

    mesh = o3d.io.read_triangle_mesh(str(raw_mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("Raw metric mesh is empty")
    raw_mesh_sha256 = validate_mesh_receipt(raw_mesh_path, mesh, tsdf_receipt)
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    if not np.isfinite(vertices).all() or not np.isfinite(triangle_normals).all():
        raise RuntimeError("Raw metric mesh contains nonfinite values")

    fit = fit_ground_plane(
        vertices, triangles, triangle_normals, tsdf_receipt, fit_policy
    )
    normal = np.asarray(fit.pop("normal"), dtype=np.float64)
    offset_m = float(fit.pop("offset_m"))
    ground_receipt = {
        "schema_version": "e26_ground_target_ground_fit_v1",
        "status": "ground_plane_fitted",
        "source_mesh": str(raw_mesh_path),
        "source_mesh_sha256": raw_mesh_sha256,
        "source_tsdf_receipt": str(tsdf_receipt_path),
        "source_tsdf_receipt_sha256": sha256_file(tsdf_receipt_path),
        "language_model_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "manual_review_used": False,
        "upstream_provenance": upstream_provenance,
        "fit": fit,
        "ground_plane_world_lio": {
            "normal": normal.tolist(),
            "offset_m": offset_m,
        },
        "policy": config["ground_fit"],
        "scientific_boundary": (
            "The plane is a deterministic engineering estimate from low, broad, "
            "approximately horizontal mesh support and metric ROI plausibility gates."
        ),
    }
    atomic_write_json(output / "ground_plane_receipt.json", ground_receipt)
    append_event(log_path, {"stage": "ground_fit", "status": "complete"})

    classification = classify_ground_triangles(
        vertices,
        triangles,
        triangle_normals,
        normal,
        offset_m,
        classification_policy,
    )
    remove = np.asarray(classification["remove"], dtype=bool)
    ground_removed = o3d.geometry.TriangleMesh(mesh)
    ground_removed.remove_triangles_by_mask(remove)
    ground_removed.remove_unreferenced_vertices()
    if not ground_removed.has_vertices() or not ground_removed.has_triangles():
        raise RuntimeError("Ground removal produced an empty mesh")
    append_event(
        log_path,
        {
            "stage": "ground_removal",
            "status": "complete",
            "removed_triangles": int(np.count_nonzero(remove)),
            "preserved_contact_triangles": int(
                np.count_nonzero(classification["preserve_contact"])
            ),
        },
    )

    triangle_keep, decisions, component_summary = component_keep_decisions(
        ground_removed,
        normal,
        offset_m,
        classification_policy,
        component_policy,
    )
    target_mesh = o3d.geometry.TriangleMesh(ground_removed)
    target_mesh.remove_triangles_by_mask(~triangle_keep)
    target_mesh.remove_unreferenced_vertices()
    target_mesh.compute_vertex_normals()
    if not target_mesh.has_vertices() or not target_mesh.has_triangles():
        raise RuntimeError("Target component extraction produced an empty mesh")
    append_event(
        log_path,
        {"stage": "target_components", "status": "complete", **component_summary},
    )

    decisions_path = output / "component_decisions.json"
    atomic_write_json(
        decisions_path,
        {
            "schema_version": "e26_ground_target_component_decisions_v1",
            "status": "complete",
            "policy": config["target_components"],
            "summary": component_summary,
            "components": decisions,
        },
    )
    mesh_path = output / "ground_removed_target_observed_surface_mesh.ply"
    write_mesh(mesh_path, target_mesh)
    o3d.utility.random.seed(random_seed)
    cloud = target_mesh.sample_points_uniformly(number_of_points=sample_count)
    point_path = output / "ground_removed_target_xyzrgb_f32_1m.ply"
    write_xyzrgb_f32_ply(point_path, cloud)

    bounds_min = np.asarray(target_mesh.get_min_bound(), dtype=np.float64)
    bounds_max = np.asarray(target_mesh.get_max_bound(), dtype=np.float64)
    receipt = {
        "schema_version": "e26_ground_target_extraction_receipt_v2",
        "status": "complete_ground_removed_target_roi_observed_surface",
        "language_model_used": False,
        "visual_language_model_used": False,
        "manual_visual_decision_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "glass_processing_used": False,
        "input": {
            "raw_mesh": str(raw_mesh_path),
            "raw_mesh_sha256": raw_mesh_sha256,
            "tsdf_receipt": str(tsdf_receipt_path),
            "tsdf_receipt_sha256": sha256_file(tsdf_receipt_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "algorithm_code_sha256": sha256_file(Path(__file__)),
            "upstream_provenance": upstream_provenance,
        },
        "ground_plane": ground_receipt["ground_plane_world_lio"],
        "ground_classification": {
            "input_triangle_count": len(triangles),
            "removed_triangle_count": int(np.count_nonzero(remove)),
            "preserved_vertical_or_ambiguous_contact_triangle_count": int(
                np.count_nonzero(classification["preserve_contact"])
            ),
            "confident_ground_triangle_count": int(
                np.count_nonzero(classification["confident_ground"])
            ),
            "unconditional_low_artifact_triangle_count": int(
                np.count_nonzero(classification["unconditional_low_artifact"])
            ),
            "vertical_support_seed_triangle_count": int(
                np.count_nonzero(classification["vertical_seed"])
            ),
            "support_grid_shape": list(classification["grid_shape"]),
            "policy": config["ground_classification"],
        },
        "component_filter": component_summary,
        "output": {
            "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "mesh_vertices": len(target_mesh.vertices),
            "mesh_triangles": len(target_mesh.triangles),
            "point_cloud": str(point_path),
            "point_cloud_sha256": sha256_file(point_path),
            "point_count": len(cloud.points),
            "axis_aligned_bounds_m": {
                "minimum": bounds_min.tolist(),
                "maximum": bounds_max.tolist(),
            },
        },
        "contact_boundary_policy": (
            "Near-plane triangles are removed only when ground-aligned. Vertical or "
            "ambiguous contact triangles require a machine-detected vertical support "
            "column above them. Low-only components without that support are rejected."
        ),
        "unknown_surface_policy": "leave_open_without_completion",
        "runtime_seconds": time.monotonic() - started,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(output / "EXTRACTION_RECEIPT.json", receipt)
    atomic_write_json(
        output / "FINAL_STATUS.json",
        {
            "schema_version": "e26_ground_target_final_status_v1",
            "status": receipt["status"],
            "language_model_used": False,
            "visual_language_model_used": False,
            "manual_visual_decision_used": False,
            "glass_processing_used": False,
            "mesh": receipt["output"]["mesh"],
            "point_cloud": receipt["output"]["point_cloud"],
            "receipt": str(output / "EXTRACTION_RECEIPT.json"),
            "claim_boundary": receipt["claim_boundary"],
        },
    )
    append_event(log_path, {"stage": "finalize", "status": receipt["status"]})
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", required=True, type=Path)
    parser.add_argument("--tsdf-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    try:
        receipt = run(args)
    except Exception as exc:
        failure = {
            "schema_version": "e26_ground_target_failure_v1",
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "language_model_used": False,
            "derived_output_reusable": False,
        }
        failure_path = (
            output / "FAILURE.json"
            if output.exists()
            else output.with_name(output.name + ".FAILURE.json")
        )
        atomic_write_json(failure_path, failure)
        raise
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
