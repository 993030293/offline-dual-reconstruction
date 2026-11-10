"""Materialize the S08 observed mesh and build its point-cloud preview."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from common import atomic_write_json, read_json, sha256_file


def write_xyzrgb_f32_ply(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("sampled S08 point cloud has invalid XYZ values")
    if cloud.has_colors():
        colors = np.rint(np.clip(np.asarray(cloud.colors), 0.0, 1.0) * 255.0).astype(np.uint8)
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
        "comment S08 terminal preview XYZRGB\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", type=Path, required=True)
    parser.add_argument("--ground-receipt", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=20260920)
    parser.add_argument("--expected-triangles", type=int)
    args = parser.parse_args()

    raw_mesh = args.raw_mesh.resolve(strict=True)
    ground_path = args.ground_receipt.resolve(strict=True)
    output_mesh = args.output_mesh.resolve() if args.output_mesh is not None else None
    output_ply = args.output_ply.resolve()
    output_receipt = args.output_receipt.resolve()
    if output_ply.exists() or output_receipt.exists() or (
        output_mesh is not None and output_mesh.exists()
    ):
        raise FileExistsError("S08 preview output already exists")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    if output_mesh is not None:
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_count <= 0:
        raise ValueError("sample count must be positive")

    started = time.monotonic()
    receipt = read_json(ground_path)
    plane = receipt["ground_plane_world_lio"]
    rule = receipt["ground_triangle_rule"]
    normal = np.asarray(plane["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    offset = float(plane["offset_m"])
    mesh = o3d.io.read_triangle_mesh(str(raw_mesh))
    if not mesh.has_triangles() or not mesh.has_vertices():
        raise ValueError("raw S08 input mesh is empty")
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)
    if not np.isfinite(vertices).all() or not np.isfinite(triangle_normals).all():
        raise ValueError("raw S08 mesh contains nonfinite values")

    centroids = vertices[triangles].mean(axis=1)
    signed_distance = centroids @ normal + offset
    vertex_height = vertices[triangles] @ normal + offset
    alignment = np.abs(triangle_normals @ normal)
    removal = (
        (signed_distance <= float(rule["maximum_signed_height_m"]))
        & (alignment >= float(rule["minimum_normal_cosine"]))
    ) | (
        np.max(vertex_height, axis=1)
        <= float(rule["unconditional_low_artifact_maximum_height_m"])
    )
    step08 = o3d.geometry.TriangleMesh(mesh)
    step08.remove_triangles_by_mask(removal)
    step08.remove_unreferenced_vertices()
    actual_triangles = len(step08.triangles)
    if not step08.has_vertices() or not step08.has_triangles():
        raise RuntimeError("S08 ground removal produced an empty mesh")
    if args.expected_triangles is not None and actual_triangles != args.expected_triangles:
        raise RuntimeError(
            f"S08 triangle count {actual_triangles} != expected {args.expected_triangles}"
        )
    step08.compute_vertex_normals()
    mesh_artifact = None
    if output_mesh is not None:
        temporary_mesh = output_mesh.with_name(output_mesh.name + ".partial.ply")
        if temporary_mesh.exists():
            raise FileExistsError(temporary_mesh)
        written = o3d.io.write_triangle_mesh(
            str(temporary_mesh),
            step08,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
        if not written or not temporary_mesh.is_file() or temporary_mesh.stat().st_size == 0:
            raise RuntimeError("Failed to materialize the S08 observed-surface mesh")
        temporary_mesh.replace(output_mesh)
        bounds_min = np.asarray(step08.get_min_bound(), dtype=np.float64)
        bounds_max = np.asarray(step08.get_max_bound(), dtype=np.float64)
        mesh_artifact = {
            "path": str(output_mesh),
            "sha256": sha256_file(output_mesh),
            "bytes": output_mesh.stat().st_size,
            "vertices": len(step08.vertices),
            "triangles": len(step08.triangles),
            "has_vertex_normals": step08.has_vertex_normals(),
            "has_vertex_colors": step08.has_vertex_colors(),
            "axis_aligned_bounds_m": {
                "minimum": bounds_min.tolist(),
                "maximum": bounds_max.tolist(),
            },
            "role": "observed_surface_after_s08_ground_removal",
            "watertightness_claim": False,
        }

    o3d.utility.random.seed(args.random_seed)
    cloud = step08.sample_points_uniformly(number_of_points=args.sample_count)
    if len(cloud.points) != args.sample_count:
        raise RuntimeError("S08 surface sampling returned an unexpected point count")
    write_xyzrgb_f32_ply(output_ply, cloud)
    result = {
        "schema_version": "e26_s08_preview_derivative_v2",
        "status": "verified_preview_derivative",
        "language_model_used": False,
        "meaning": "S08 only: ground and unconditional low-artifact triangles removed; connected-component filtering not applied.",
        "source_mesh": str(raw_mesh),
        "source_mesh_sha256": sha256_file(raw_mesh),
        "ground_receipt": str(ground_path),
        "ground_receipt_sha256": sha256_file(ground_path),
        "input_vertices": len(mesh.vertices),
        "input_triangles": len(mesh.triangles),
        "removed_ground_or_low_artifact_triangles": int(np.count_nonzero(removal)),
        "step08_vertices": len(step08.vertices),
        "step08_triangles": actual_triangles,
        "expected_step08_triangles": args.expected_triangles,
        "triangle_count_match": args.expected_triangles is None
        or actual_triangles == args.expected_triangles,
        "preview_surface_points": len(cloud.points),
        "random_seed": args.random_seed,
        "connected_component_filter_applied": False,
        "component_filter_code_executed": False,
        "mesh_generated": mesh_artifact is not None,
        "observed_surface_mesh": mesh_artifact,
        "unknown_surface_policy": "leave_open_without_completion",
        "output_ply": str(output_ply),
        "output_ply_sha256": sha256_file(output_ply),
        "runtime_seconds": time.monotonic() - started,
        "scientific_boundary": (
            "The triangle mesh and point preview stop immediately after the S08 ground-triangle rule. "
            "They do not execute S09 component filtering, infer missing surfaces, or establish watertightness."
        ),
    }
    atomic_write_json(output_receipt, result)
    print(output_ply)


if __name__ == "__main__":
    main()
