"""Fit a gravity-aligned ground plane to a raw metric TSDF mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DISTANCE_THRESHOLD_M = 0.012
MAXIMUM_TILT_DEGREES = 15.0
MINIMUM_INLIER_FRACTION = 0.50
MINIMUM_INLIER_XY_EXTENT_M = 0.50
RANDOM_SEED = 20260920


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", required=True, type=Path)
    parser.add_argument("--tsdf-receipt", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    args = parser.parse_args()

    raw_mesh = args.raw_mesh.resolve(strict=True)
    tsdf_receipt_path = args.tsdf_receipt.resolve(strict=True)
    output_receipt = args.output_receipt.resolve()
    if output_receipt.exists():
        raise FileExistsError(output_receipt)
    tsdf_receipt = read_json(tsdf_receipt_path)
    if tsdf_receipt.get("language_model_used") is not False:
        raise RuntimeError("TSDF receipt does not satisfy the no-language-model contract")
    if tsdf_receipt.get("object_mask_used") is not False:
        raise RuntimeError("TSDF receipt does not satisfy the no-object-mask contract")

    mesh = o3d.io.read_triangle_mesh(str(raw_mesh))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("Raw TSDF mesh is empty")
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    centroids = vertices[triangles].mean(axis=1)
    if not np.isfinite(vertices).all() or not np.isfinite(triangle_normals).all():
        raise RuntimeError("Raw TSDF mesh contains nonfinite values")

    low_cutoff = float(np.quantile(centroids[:, 2], 0.45))
    horizontal = np.abs(triangle_normals[:, 2]) >= np.cos(np.deg2rad(25.0))
    candidate_triangles = (centroids[:, 2] <= low_cutoff) & horizontal
    candidate_vertices = np.unique(triangles[candidate_triangles].reshape(-1))
    if len(candidate_vertices) < 1000:
        raise RuntimeError(f"Too few low horizontal ground candidates: {len(candidate_vertices)}")
    candidate_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(vertices[candidate_vertices])
    )
    o3d.utility.random.seed(RANDOM_SEED)
    model, inliers = candidate_cloud.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD_M,
        ransac_n=3,
        num_iterations=4096,
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
    minimum_normal_z = float(np.cos(np.deg2rad(MAXIMUM_TILT_DEGREES)))
    if normal[2] < minimum_normal_z:
        raise RuntimeError(
            f"Fitted low plane is not gravity aligned: normal_z={normal[2]:.6f}"
        )
    if inlier_fraction < MINIMUM_INLIER_FRACTION:
        raise RuntimeError(
            f"Fitted low plane is not dominant: inlier_fraction={inlier_fraction:.6f}"
        )
    if bool(np.any(inlier_xy_extent < MINIMUM_INLIER_XY_EXTENT_M)):
        raise RuntimeError(
            "Fitted low plane lacks horizontal support: "
            f"xy_extent_m={inlier_xy_extent.tolist()}"
        )
    distances = np.abs(inlier_points @ normal + offset)
    focus = np.asarray(tsdf_receipt["roi_derivation"]["focus_world_lio_m"], dtype=np.float64)
    ground_z_at_focus = float(-(offset + normal[0] * focus[0] + normal[1] * focus[1]) / normal[2])
    sparse_floor = tsdf_receipt.get("roi_derivation", {}).get("spatial_sparse_floor_plane")
    sparse_floor_difference = None
    if isinstance(sparse_floor, dict):
        sparse_floor_difference = abs(
            ground_z_at_focus - float(sparse_floor["ground_z_at_focus_m"])
        )
        if sparse_floor_difference > 0.08:
            raise RuntimeError(
                "Mesh-derived plane disagrees with the broad sparse floor plane: "
                f"difference={sparse_floor_difference:.6f} m"
            )
    roi_minimum = np.asarray(tsdf_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    roi_maximum = np.asarray(tsdf_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    if not (roi_minimum[2] + 0.02 <= ground_z_at_focus <= focus[2] - 0.25):
        raise RuntimeError(
            "Fitted ground height is outside the mechanical ROI/focus plausibility interval: "
            f"z={ground_z_at_focus:.6f}"
        )

    receipt = {
        "schema_version": "e26_c1_machine_ground_plane_v1",
        "status": "ground_plane_fitted_from_current_raw_mesh",
        "source_mesh": str(raw_mesh),
        "source_mesh_sha256": sha256_file(raw_mesh),
        "source_tsdf_receipt": str(tsdf_receipt_path),
        "source_tsdf_receipt_sha256": sha256_file(tsdf_receipt_path),
        "language_model_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "manual_review_used": False,
        "fit": {
            "method": "deterministic RANSAC on low, approximately horizontal mesh triangles",
            "random_seed": RANDOM_SEED,
            "low_centroid_quantile": 0.45,
            "candidate_triangle_maximum_tilt_degrees": 25.0,
            "candidate_vertex_count": len(candidate_vertices),
            "inlier_count": len(inliers),
            "inlier_fraction": inlier_fraction,
            "minimum_accepted_inlier_fraction": MINIMUM_INLIER_FRACTION,
            "inlier_xy_extent_m": inlier_xy_extent.tolist(),
            "minimum_accepted_inlier_xy_extent_m": MINIMUM_INLIER_XY_EXTENT_M,
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "iterations": 4096,
            "maximum_accepted_plane_tilt_degrees": MAXIMUM_TILT_DEGREES,
            "inlier_residual_m": {
                "p50": float(np.quantile(distances, 0.50)),
                "p95": float(np.quantile(distances, 0.95)),
                "maximum": float(np.max(distances)),
            },
            "ground_z_at_robust_focus_m": ground_z_at_focus,
            "spatial_sparse_floor_difference_m": sparse_floor_difference,
        },
        "ground_plane_world_lio": {
            "normal": normal.tolist(),
            "offset_m": offset,
        },
        "ground_triangle_rule": {
            "maximum_signed_height_m": 0.04,
            "minimum_normal_cosine": 0.65,
            "unconditional_low_artifact_maximum_height_m": 0.015,
        },
        "connected_component_filter_applied": False,
        "scientific_boundary": (
            "Ground is estimated from the current mesh by low position, approximate gravity alignment, "
            "dominant planar support, and height only. No semantic label, object mask, manual visual "
            "choice, or component filtering is used."
        ),
    }
    write_json_new(output_receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
