"""Render a trained PGSR scene into an automatically bounded metric TSDF mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import resource
except ModuleNotFoundError:
    resource = None

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimate_sparse_floor import estimate_sparse_floor


VOXEL_SIZE_M = 0.005
SDF_TRUNCATION_M = 0.020
MAX_DEPTH_M = 10.0
MAX_NORMAL_VIEW_ANGLE_DEGREES = 80.0
FOCUS_HUBER_DELTA_M = 0.35
SPARSE_CLUSTER_EPS_M = 0.15
SPARSE_CLUSTER_MIN_NEIGHBORS = 8
SPARSE_CLUSTER_HORIZONTAL_PADDING_M = 0.15
SPARSE_CLUSTER_TOP_PADDING_M = 0.12


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def camera_center_and_forward(view) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.asarray(view.R, dtype=np.float64).T
    world_to_camera[:3, 3] = np.asarray(view.T, dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    center = camera_to_world[:3, 3]
    forward = camera_to_world[:3, 2]
    forward /= np.linalg.norm(forward)
    return center, forward


def solve_viewing_axis_focus(
    centers: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.ones(len(centers), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    focus = centers.mean(axis=0)
    iterations = 0
    for iterations in range(1, 21):
        projectors = identity[None, :, :] - directions[:, :, None] * directions[:, None, :]
        matrix = np.sum(weights[:, None, None] * projectors, axis=0)
        vector = np.sum(
            weights[:, None] * np.einsum("nij,nj->ni", projectors, centers), axis=0
        )
        candidate = np.linalg.solve(matrix, vector)
        residuals = np.linalg.norm(
            np.cross(candidate[None, :] - centers, directions), axis=1
        )
        next_weights = np.minimum(1.0, FOCUS_HUBER_DELTA_M / np.maximum(residuals, 1e-12))
        if np.linalg.norm(candidate - focus) <= 1e-9:
            focus = candidate
            weights = next_weights
            break
        focus = candidate
        weights = next_weights
    residuals = np.linalg.norm(np.cross(focus[None, :] - centers, directions), axis=1)
    return focus, residuals, iterations


def load_sparse_points(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        points.append([float(value) for value in fields[1:4]])
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 1000:
        raise RuntimeError("Metric sparse model has too few valid points for automatic ROI")
    if not np.isfinite(result).all():
        raise RuntimeError("Metric sparse model contains nonfinite points")
    return result


def largest_density_cluster(points: np.ndarray) -> tuple[np.ndarray, dict]:
    neighbors = cKDTree(points).query_ball_point(points, r=SPARSE_CLUSTER_EPS_M)
    core = np.asarray(
        [len(indices) >= SPARSE_CLUSTER_MIN_NEIGHBORS for indices in neighbors], dtype=bool
    )
    labels = np.full(len(points), -1, dtype=np.int32)
    cluster_id = 0
    for seed in range(len(points)):
        if not core[seed] or labels[seed] >= 0:
            continue
        labels[seed] = cluster_id
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            if not core[current]:
                continue
            for neighbor in neighbors[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = cluster_id
                    if core[neighbor]:
                        queue.append(neighbor)
        cluster_id += 1
    if cluster_id == 0:
        raise RuntimeError("No dense metric sparse cluster was found for automatic ROI")
    counts = np.bincount(labels[labels >= 0], minlength=cluster_id)
    selected_id = int(np.argmax(counts))
    selected = points[labels == selected_id]
    if len(selected) < 1000:
        raise RuntimeError(f"Largest metric sparse cluster is too small: {len(selected)}")
    details = {
        "algorithm": "deterministic 3D DBSCAN implemented with cKDTree",
        "epsilon_m": SPARSE_CLUSTER_EPS_M,
        "minimum_neighbors": SPARSE_CLUSTER_MIN_NEIGHBORS,
        "candidate_count": len(points),
        "cluster_count": cluster_id,
        "noise_count": int(np.count_nonzero(labels < 0)),
        "selected_cluster_id": selected_id,
        "selected_cluster_count": len(selected),
    }
    return selected, details


def derive_roi(
    train_cameras,
    sparse_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    poses = [camera_center_and_forward(view) for view in train_cameras]
    centers = np.stack([value[0] for value in poses])
    directions = np.stack([value[1] for value in poses])
    focus, residuals, iterations = solve_viewing_axis_focus(centers, directions)
    forward_depths = np.sum((focus[None, :] - centers) * directions, axis=1)
    if float(np.median(forward_depths)) <= 0:
        directions = -directions
        focus, residuals, iterations = solve_viewing_axis_focus(centers, directions)
        forward_depths = np.sum((focus[None, :] - centers) * directions, axis=1)
    if np.count_nonzero(forward_depths > 0) < int(np.ceil(0.95 * len(forward_depths))):
        raise RuntimeError("Automatic focus is not in front of at least 95 percent of cameras")

    median_depth = float(np.median(forward_depths))
    median_camera_z = float(np.median(centers[:, 2]))
    sparse_floor = estimate_sparse_floor(
        sparse_points, focus[:2], median_camera_z, 2.5
    )
    ground_z = float(sparse_floor["ground_z_at_focus_m"])
    candidates = sparse_points[
        (sparse_points[:, 2] > ground_z + 0.08)
        & (sparse_points[:, 2] < ground_z + 2.20)
        & (np.linalg.norm(sparse_points[:, :2] - focus[None, :2], axis=1) < 2.5)
    ]
    cluster, cluster_details = largest_density_cluster(candidates)
    lower_xy = np.quantile(cluster[:, :2], 0.005, axis=0) - SPARSE_CLUSTER_HORIZONTAL_PADDING_M
    upper_xy = np.quantile(cluster[:, :2], 0.995, axis=0) + SPARSE_CLUSTER_HORIZONTAL_PADDING_M
    horizontal_extent = upper_xy - lower_xy
    if np.any(horizontal_extent < 1.0) or np.any(horizontal_extent > 2.5):
        raise RuntimeError(f"Sparse-cluster ROI extent is implausible: {horizontal_extent.tolist()}")
    cluster_top = float(np.quantile(cluster[:, 2], 0.995))
    corners = np.asarray([
        [lower_xy[0], lower_xy[1]], [lower_xy[0], upper_xy[1]],
        [upper_xy[0], lower_xy[1]], [upper_xy[0], upper_xy[1]],
    ])
    floor_normal = np.asarray(sparse_floor["normal"])
    floor_corners = -(
        float(sparse_floor["offset_m"]) + corners @ floor_normal[:2]
    ) / floor_normal[2]
    minimum = np.array(
        [lower_xy[0], lower_xy[1], float(np.min(floor_corners) - 0.15)],
        dtype=np.float64,
    )
    maximum = np.array(
        [
            upper_xy[0],
            upper_xy[1],
            min(
                ground_z + 2.20,
                max(float(focus[2]) + 0.35, cluster_top + SPARSE_CLUSTER_TOP_PADDING_M),
            ),
        ],
        dtype=np.float64,
    )
    if not np.all(maximum > minimum) or not np.all((focus >= minimum) & (focus <= maximum)):
        raise RuntimeError("Automatically derived ROI does not contain the robust viewing focus")
    details = {
        "method": "IRLS Huber fit to metric camera optical axes",
        "camera_count": len(train_cameras),
        "huber_delta_m": FOCUS_HUBER_DELTA_M,
        "irls_iterations": iterations,
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
        "spatial_sparse_floor_plane": sparse_floor,
        "metric_sparse_vehicle_support": {
            "ground_z_m": ground_z,
            "ground_support_count": sparse_floor["support_point_count_within_5cm"],
        },
        "metric_sparse_cluster": {
            **cluster_details,
            "selection_rule": (
                "spatial_floor_z + 0.08 < z < spatial_floor_z + 2.20 and "
                "XY distance to optical-axis focus < 2.5 m"
            ),
            "quantile_interval": [0.005, 0.995],
            "horizontal_padding_m": SPARSE_CLUSTER_HORIZONTAL_PADDING_M,
            "top_padding_m": SPARSE_CLUSTER_TOP_PADDING_M,
            "quantile_minimum_xyz_m": np.quantile(cluster, 0.005, axis=0).tolist(),
            "quantile_maximum_xyz_m": np.quantile(cluster, 0.995, axis=0).tolist(),
        },
        "fixed_rule": {
            "horizontal_bounds": "largest sparse 3D density cluster q0.005..q0.995 plus 0.15 m",
            "minimum_z": "minimum spatial floor height over ROI corners - 0.15",
            "maximum_z": (
                "min(spatial_floor_z + 2.20, "
                "max(focus_z + 0.35, sparse_cluster_q0.995_z + 0.12))"
            ),
        },
    }
    return minimum, maximum, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stage-name", default="06_pgsr_inlier")
    parser.add_argument("--output-stage", default="07_roi_tsdf")
    parser.add_argument("--iteration", type=int, default=15000)
    args = parser.parse_args()

    run = args.run_root.resolve(strict=True)
    stage = run / args.stage_name
    source = args.source.resolve(strict=True)
    model = stage / f"model_pgsr_{args.iteration}"
    output = run / args.output_stage
    if output.exists():
        raise FileExistsError(f"TSDF output already exists: {output}")
    training_receipt = read_json(stage / "training_receipt.json")
    if training_receipt.get("status") != "trained_15000_checkpoint_verified":
        raise RuntimeError("PGSR training receipt is not complete and verified")
    if training_receipt.get("language_model_used") is not False:
        raise RuntimeError("No-language-model contract is not satisfied")
    if training_receipt.get("object_mask_used") is not False:
        raise RuntimeError("No-object-mask contract is not satisfied")
    output.mkdir(parents=True)

    sys.path.insert(0, str(source))
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    pgsr_parser = ArgumentParser()
    model_params = ModelParams(pgsr_parser, sentinel=True)
    pipeline_params = PipelineParams(pgsr_parser)
    pgsr_parser.add_argument("--iteration", default=args.iteration, type=int)
    original_argv = sys.argv
    sys.argv = ["auto_roi_tsdf", "-m", str(model), "--iteration", str(args.iteration)]
    try:
        combined = get_combined_args(pgsr_parser)
    finally:
        sys.argv = original_argv
    dataset = model_params.extract(combined)
    pipeline = pipeline_params.extract(combined)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    expected_cameras = int(training_receipt["registered_camera_count"])
    if len(train_cameras) != expected_cameras:
        raise RuntimeError(
            f"Loaded {len(train_cameras)} cameras, expected {expected_cameras} from training receipt"
        )

    sparse_points = load_sparse_points(stage / "input" / "sparse" / "points3D.txt")
    roi_minimum, roi_maximum, roi_derivation = derive_roi(train_cameras, sparse_points)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE_M,
        sdf_trunc=SDF_TRUNCATION_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    started = time.monotonic()
    total_valid = 0
    total_roi = 0
    frame_rows: list[dict] = []
    with torch.no_grad():
        for index, view in enumerate(train_cameras):
            rendered = render(view, gaussians, pipeline, background)
            depth = rendered["plane_depth"].squeeze()
            view_direction = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = torch.nn.functional.normalize(
                rendered["depth_normal"].permute(1, 2, 0), p=2, dim=-1
            )
            angle = torch.acos(
                torch.sum(view_direction * depth_normal, dim=-1).abs().clamp(0.0, 1.0)
            )
            depth = depth.clone()
            depth[angle > np.deg2rad(MAX_NORMAL_VIEW_ANGLE_DEGREES)] = 0
            depth[(depth <= 0) | (depth > MAX_DEPTH_M)] = 0
            depth_array = depth.cpu().numpy().astype(np.float32)
            height, width = depth_array.shape
            rows, columns = np.indices((height, width), dtype=np.float64)
            camera_points = np.stack(
                (
                    (columns - float(view.Cx)) / float(view.Fx) * depth_array,
                    (rows - float(view.Cy)) / float(view.Fy) * depth_array,
                    depth_array,
                ),
                axis=-1,
            )
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = np.asarray(view.R, dtype=np.float64).T
            world_to_camera[:3, 3] = np.asarray(view.T, dtype=np.float64)
            camera_to_world = np.linalg.inv(world_to_camera)
            world_points = (
                camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
            )
            valid = depth_array > 0
            inside = valid & np.all(world_points >= roi_minimum, axis=-1) & np.all(
                world_points <= roi_maximum, axis=-1
            )
            roi_depth = depth_array.copy()
            roi_depth[~inside] = 0
            color_array = np.ascontiguousarray(
                (
                    rendered["render"]
                    .permute(1, 2, 0)
                    .clamp(0.0, 1.0)
                    .cpu()
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
            )
            depth_u16 = np.ascontiguousarray((roi_depth * 1000.0).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_array),
                o3d.geometry.Image(depth_u16),
                depth_scale=1000.0,
                depth_trunc=MAX_DEPTH_M,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width,
                    height,
                    float(view.Fx),
                    float(view.Fy),
                    float(view.Cx),
                    float(view.Cy),
                ),
                world_to_camera,
            )
            valid_count = int(np.count_nonzero(valid))
            roi_count = int(np.count_nonzero(inside))
            total_valid += valid_count
            total_roi += roi_count
            frame_rows.append(
                {
                    "index": index,
                    "image_name": view.image_name,
                    "valid_depth_pixels": valid_count,
                    "roi_depth_pixels": roi_count,
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(train_cameras):
                heartbeat = {
                    "status": "running",
                    "integrated_frames": index + 1,
                    "frame_count": len(train_cameras),
                    "runtime_seconds": time.monotonic() - started,
                    "total_valid_depth_pixels": total_valid,
                    "total_roi_depth_pixels": total_roi,
                }
                (output / "heartbeat.json").write_text(
                    json.dumps(heartbeat, indent=2) + "\n", encoding="utf-8"
                )
                print(f"integrated {index + 1}/{len(train_cameras)}", flush=True)
            del rendered, depth, view_direction, depth_normal, angle

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(roi_minimum, roi_maximum))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("TSDF extraction returned an empty mesh")
    raw_mesh = output / "paperbox_auto_roi_pgsr_5mm_raw_mesh.ply"
    if not o3d.io.write_triangle_mesh(str(raw_mesh), mesh, write_ascii=False):
        raise RuntimeError("Failed to write TSDF mesh")
    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": "e26_c1_auto_roi_pgsr_tsdf_v5",
        "status": "raw_s08_mesh_generated",
        "source_model": str(model),
        "source_iteration": args.iteration,
        "source_training_receipt": str(stage / "training_receipt.json"),
        "language_model_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "known_box_dimension_used": False,
        "roi_derivation": roi_derivation,
        "roi_world_lio_m": {
            "minimum": roi_minimum.tolist(),
            "maximum": roi_maximum.tolist(),
        },
        "voxel_size_m": VOXEL_SIZE_M,
        "sdf_truncation_m": SDF_TRUNCATION_M,
        "maximum_depth_m": MAX_DEPTH_M,
        "maximum_normal_view_angle_degrees": MAX_NORMAL_VIEW_ANGLE_DEGREES,
        "streaming_integration": True,
        "frame_count": len(frame_rows),
        "total_valid_depth_pixels": total_valid,
        "total_roi_depth_pixels": total_roi,
        "roi_pixel_fraction": total_roi / max(total_valid, 1),
        "mesh": {
            "path": str(raw_mesh),
            "sha256": sha256_file(raw_mesh),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        },
        "connected_component_filter_applied": False,
        "runtime_seconds": elapsed,
        "peak_rss_kib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        ),
        "per_frame": frame_rows,
        "scientific_boundary": (
            "The ROI is derived mechanically from the metric camera optical axes and fixed numeric rules. "
            "It supplies no semantic label, object mask, box dimensions, or accepted 3D reference."
        ),
    }
    write_json_new(output / "roi_tsdf_receipt.json", receipt)
    (output / "heartbeat.json").write_text(
        json.dumps({"status": "complete", **{k: v for k, v in receipt.items() if k != "per_frame"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
