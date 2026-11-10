"""Fail closed when ground removal discards observed elevated target support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


MAXIMUM_NEAREST_VERTEX_DISTANCE_M = 0.05
MINIMUM_SPARSE_SUPPORT_POINTS = 1000
MINIMUM_RAW_SUPPORT_FRACTION = 0.80
MINIMUM_RETAINED_RAW_SUPPORT_FRACTION = 0.95


def read_points(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        points = [
            [float(value) for value in line.split()[1:4]]
            for line in stream if line and not line.startswith("#")
        ]
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise RuntimeError("Metric sparse support is invalid")
    return result


def near_mesh(points: np.ndarray, path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"Mesh is empty: {path}")
    vertices = np.asarray(mesh.vertices)
    distance = cKDTree(vertices).query(points, k=1, workers=-1)[0]
    return distance <= MAXIMUM_NEAREST_VERTEX_DISTANCE_M


def audit(
    sparse_path: Path,
    raw_mesh_path: Path,
    target_mesh_path: Path,
    tsdf_receipt_path: Path,
) -> dict:
    receipt = json.loads(tsdf_receipt_path.read_text(encoding="utf-8-sig"))
    floor = receipt.get("roi_derivation", {}).get("spatial_sparse_floor_plane")
    if not isinstance(floor, dict):
        raise RuntimeError("Spatial sparse floor evidence is missing from TSDF receipt")
    roi = receipt["roi_world_lio_m"]
    minimum = np.asarray(roi["minimum"], dtype=np.float64)
    maximum = np.asarray(roi["maximum"], dtype=np.float64)
    normal = np.asarray(floor["normal"], dtype=np.float64)
    offset = float(floor["offset_m"])
    points = read_points(sparse_path)
    height = points @ normal + offset
    relevant = (
        np.all((points >= minimum) & (points <= maximum), axis=1)
        & (height > 0.10)
        & (height < 2.20)
    )
    support = points[relevant]
    if len(support) < MINIMUM_SPARSE_SUPPORT_POINTS:
        raise RuntimeError(f"Too few elevated metric sparse support points: {len(support)}")
    raw_near = near_mesh(support, raw_mesh_path)
    target_near = near_mesh(support, target_mesh_path)
    raw_count = int(raw_near.sum())
    retained_count = int((raw_near & target_near).sum())
    raw_fraction = raw_count / len(support)
    retained_fraction = retained_count / raw_count if raw_count else 0.0
    passed = (
        raw_fraction >= MINIMUM_RAW_SUPPORT_FRACTION
        and retained_fraction >= MINIMUM_RETAINED_RAW_SUPPORT_FRACTION
    )
    return {
        "schema_version": "e26_target_support_preservation_v1",
        "status": "pass" if passed else "failed_quality_gate",
        "method": "metric sparse above broad spatial floor versus S07 and M3 mesh vertices",
        "metric_sparse_points": str(sparse_path),
        "raw_s07_mesh": str(raw_mesh_path),
        "final_m3_mesh": str(target_mesh_path),
        "tsdf_receipt": str(tsdf_receipt_path),
        "support_point_count": len(support),
        "raw_supported_point_count": raw_count,
        "final_retained_raw_supported_point_count": retained_count,
        "raw_support_fraction": raw_fraction,
        "retained_raw_support_fraction": retained_fraction,
        "maximum_nearest_vertex_distance_m": MAXIMUM_NEAREST_VERTEX_DISTANCE_M,
        "minimum_raw_support_fraction": MINIMUM_RAW_SUPPORT_FRACTION,
        "minimum_retained_raw_support_fraction": MINIMUM_RETAINED_RAW_SUPPORT_FRACTION,
        "language_model_used": False,
        "evaluation_boundary": (
            "This is a self-consistency gate against metric sparse tracks, not independent "
            "ground truth or a completeness/accuracy metric. It cannot detect target parts "
            "missing from both the sparse model and the meshes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-sparse", required=True, type=Path)
    parser.add_argument("--raw-mesh", required=True, type=Path)
    parser.add_argument("--target-mesh", required=True, type=Path)
    parser.add_argument("--tsdf-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    evidence = audit(
        args.metric_sparse.resolve(strict=True),
        args.raw_mesh.resolve(strict=True),
        args.target_mesh.resolve(strict=True),
        args.tsdf_receipt.resolve(strict=True),
    )
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if evidence["status"] != "pass":
        raise RuntimeError(
            "Target support preservation failed: "
            f"raw={evidence['raw_support_fraction']:.4f}, "
            f"retained={evidence['retained_raw_support_fraction']:.4f}"
        )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
