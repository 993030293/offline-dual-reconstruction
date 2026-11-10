"""Fit a gravity-aligned ground plane to a raw metric TSDF mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground" / "code"))
import ground_target_extraction as ground


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
    parser.add_argument("--raw-mcap", type=Path)
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
    if not np.isfinite(vertices).all() or not np.isfinite(triangle_normals).all():
        raise RuntimeError("Raw TSDF mesh contains nonfinite values")

    config = ground.read_json(Path(__file__).resolve().parents[1] / "ground" / "config.json")
    fit_policy = ground.GroundFitPolicy(
        **{**config["ground_fit"], "minimum_accepted_inlier_fraction": MINIMUM_INLIER_FRACTION}
    )
    fit = ground.fit_ground_plane(
        vertices, triangles, triangle_normals, tsdf_receipt, fit_policy,
        raw_mcap=args.raw_mcap,
        corroborated_minimum_inlier_fraction=float(config["ground_fit"]["minimum_accepted_inlier_fraction"]),
    )
    normal = np.asarray(fit["normal"], dtype=np.float64)
    offset = float(fit["offset_m"])
    focus = np.asarray(tsdf_receipt["roi_derivation"]["focus_world_lio_m"], dtype=np.float64)
    ground_z_at_focus = float(-(offset + normal[0] * focus[0] + normal[1] * focus[1]) / normal[2])
    roi_minimum = np.asarray(tsdf_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    roi_maximum = np.asarray(tsdf_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    estimated_ground_z = float(
        tsdf_receipt["roi_derivation"]["metric_sparse_vehicle_support"]["ground_z_m"]
    )
    ground_estimate_difference = abs(ground_z_at_focus - estimated_ground_z)
    if not (roi_minimum[2] + 0.02 <= ground_z_at_focus <= roi_minimum[2] + 0.30):
        raise RuntimeError(
            "Fitted ground height is outside the mechanical ROI-bottom plausibility interval: "
            f"z={ground_z_at_focus:.6f}"
        )
    if ground_estimate_difference > 0.15:
        raise RuntimeError(
            "RANSAC ground disagrees with the independent sparse-height estimate: "
            f"difference={ground_estimate_difference:.6f} m"
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
            "method": "deterministic RANSAC on low horizontal mesh triangles with conditional registered LiDAR support",
            "candidate_selection_mode": fit["candidate_selection_mode"],
            "random_seed": RANDOM_SEED,
            "low_centroid_quantile": 0.45,
            "candidate_triangle_maximum_tilt_degrees": 25.0,
            "candidate_vertex_count": fit["candidate_vertex_count"],
            "inlier_count": fit["inlier_count"],
            "inlier_fraction": fit["inlier_fraction"],
            "inlier_fraction_acceptance": fit["inlier_fraction_acceptance"],
            "minimum_accepted_inlier_fraction": MINIMUM_INLIER_FRACTION,
            "corroborated_minimum_inlier_fraction": fit["corroborated_minimum_inlier_fraction"],
            "inlier_xy_extent_m": fit["inlier_xy_extent_m"],
            "inlier_roi_span_fraction": fit["inlier_roi_span_fraction"],
            "mesh_absolute_extent_gate_passed": fit["mesh_absolute_extent_gate_passed"],
            "mesh_roi_span_gate_passed": fit["mesh_roi_span_gate_passed"],
            "registered_lidar_spatial_support": fit["registered_lidar_spatial_support"],
            "minimum_accepted_inlier_xy_extent_m": MINIMUM_INLIER_XY_EXTENT_M,
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "iterations": fit_policy.iterations,
            "maximum_accepted_plane_tilt_degrees": MAXIMUM_TILT_DEGREES,
            "inlier_residual_m": fit["inlier_residual_m"],
            "ground_z_at_robust_focus_m": ground_z_at_focus,
            "independent_sparse_ground_z_m": estimated_ground_z,
            "ground_estimate_difference_m": ground_estimate_difference,
            "maximum_accepted_ground_estimate_difference_m": 0.15,
            "candidate_height_half_window_m": fit.get("candidate_height_half_window_m"),
            "unconstrained_rejection": fit.get("unconstrained_rejection"),
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
            "Ground is estimated from the current mesh by low position, gravity alignment, "
            "planar support, ROI span, and independent sparse-height agreement. If the broad "
            "candidate set fails, a 5 cm band around the sparse ground prior is evaluated. "
            "When its mesh support falls below the S08 0.50 inlier or either spatial "
            "gate, it must still meet the M3 0.35 inlier gate, and registered LiDAR "
            "must satisfy the original absolute extent and ROI-span requirements "
            "plus temporal and grid-area support. "
            "No semantic label, object mask, "
            "manual visual choice, or component filtering is used."
        ),
    }
    write_json_new(output_receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
