#!/usr/bin/env python3
"""Fuse multi-view-verified PGSR depths with stable registered LiDAR anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree


VOXEL_SIZE_M = 0.005
SDF_TRUNCATION_M = 0.020
MAXIMUM_DEPTH_M = 10.0
MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES = 60.0
MINIMUM_RENDERED_ALPHA = 0.60
BROAD_MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES = 80.0
MINIMUM_NEIGHBOR_SUPPORT = 3
NEIGHBORS_PER_VIEW = 8
MINIMUM_NEIGHBOR_AZIMUTH_DEGREES = 3.0
MAXIMUM_NEIGHBOR_AZIMUTH_DEGREES = 40.0
MINIMUM_BASELINE_M = 0.15
MAXIMUM_BASELINE_M = 2.50
DEPTH_ABSOLUTE_TOLERANCE_M = 0.030
DEPTH_RELATIVE_TOLERANCE = 0.012
LIDAR_ANCHOR_DILATION_PIXELS = 1
LIDAR_EXTRA_INTEGRATIONS = 2
MINIMUM_COHERENT_DOMINANT_AREA_FRACTION = 0.98
MAXIMUM_COHERENT_COMPONENT_COUNT = 1000
MAXIMUM_FRAGMENTED_DOMINANT_AREA_FRACTION = 0.95
MINIMUM_FRAGMENTED_COMPONENT_COUNT = 1000
MINIMUM_VISUAL_SPARSE_SUPPORT_FRACTION = 0.95
MAXIMUM_VISUAL_SUPPORT_DROP = 0.03
MINIMUM_BROAD_TRIANGLE_GAIN_FRACTION = 0.05
MAXIMUM_BROAD_TRIANGLE_GAIN_FRACTION = 0.25
SPARSE_SUPPORT_DISTANCE_M = 0.05
ROI_TRIM_PADDING_CANDIDATES_M = (0.03, 0.06, 0.10, 0.15)
MAXIMUM_ROI_TRIM_SUPPORT_DROP = 0.002


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_sparse_support(stage: Path, roi_receipt: dict) -> np.ndarray:
    floor = roi_receipt.get("roi_derivation", {}).get("spatial_sparse_floor_plane")
    if not isinstance(floor, dict):
        raise RuntimeError("Spatial sparse floor evidence is missing for surface selection")
    points = []
    with (stage / "input" / "sparse" / "points3D.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if line and not line.startswith("#"):
                points.append([float(value) for value in line.split()[1:4]])
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("Metric sparse support is invalid")
    minimum = np.asarray(roi_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    maximum = np.asarray(roi_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    normal = np.asarray(floor["normal"], dtype=np.float64)
    height = points @ normal + float(floor["offset_m"])
    inside = np.all((points >= minimum) & (points <= maximum), axis=1)
    support = points[inside & (height > 0.10) & (height < 2.20)]
    if len(support) < 1000:
        raise RuntimeError(f"Too few elevated sparse points for surface selection: {len(support)}")
    return support


def surface_candidate_metrics(mesh: o3d.geometry.TriangleMesh, support: np.ndarray) -> dict:
    _, counts_raw, areas_raw = mesh.cluster_connected_triangles()
    counts = np.asarray(counts_raw, dtype=np.int64)
    areas = np.asarray(areas_raw, dtype=np.float64)
    if len(counts) == 0 or not np.isfinite(areas).all() or float(np.sum(areas)) <= 0:
        raise RuntimeError("Surface candidate has no valid connected components")
    distance = cKDTree(np.asarray(mesh.vertices)).query(support, k=1, workers=1)[0]
    return {
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "component_count": len(counts),
        "dominant_component_area_fraction": float(np.max(areas) / np.sum(areas)),
        "tiny_component_count_at_most_8_triangles": int(np.count_nonzero(counts <= 8)),
        "sparse_support_point_count": len(support),
        "sparse_support_fraction_within_5cm": float(np.mean(distance <= SPARSE_SUPPORT_DISTANCE_M)),
    }


def trim_surface_roi_by_sparse_support(
    mesh: o3d.geometry.TriangleMesh, support: np.ndarray, roi_receipt: dict
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Trim unsupported ROI padding while preserving the selected surface's sparse anchors."""
    sparse_bounds = roi_receipt["roi_derivation"]["metric_sparse_vehicle_support"]
    quantile_minimum = np.asarray(sparse_bounds["quantile_minimum_xyz_m"], dtype=np.float64)
    quantile_maximum = np.asarray(sparse_bounds["quantile_maximum_xyz_m"], dtype=np.float64)
    source_minimum = np.asarray(roi_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    source_maximum = np.asarray(roi_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    if not np.all(np.isfinite(quantile_minimum)) or not np.all(np.isfinite(quantile_maximum)):
        raise RuntimeError("Sparse target ROI quantile bounds are invalid")
    if not np.all(quantile_maximum[:2] > quantile_minimum[:2]):
        raise RuntimeError("Sparse target ROI quantile bounds are reversed")
    source_vertices = np.asarray(mesh.vertices)
    source_triangles = np.asarray(mesh.triangles)
    if len(source_vertices) == 0 or len(source_triangles) == 0:
        raise RuntimeError("Selected surface is empty before ROI trim")
    baseline_distances = cKDTree(source_vertices).query(support, k=1, workers=1)[0]
    baseline_supported = baseline_distances <= SPARSE_SUPPORT_DISTANCE_M
    baseline_fraction = float(np.mean(baseline_supported))
    rows = []
    chosen = mesh
    chosen_padding = ROI_TRIM_PADDING_CANDIDATES_M[-1]
    chosen_minimum = source_minimum
    chosen_maximum = source_maximum
    for padding in ROI_TRIM_PADDING_CANDIDATES_M:
        minimum = source_minimum.copy()
        maximum = source_maximum.copy()
        minimum[:2] = quantile_minimum[:2] - padding
        maximum[:2] = quantile_maximum[:2] + padding
        if np.any(minimum[:2] < source_minimum[:2] - 1e-6) or np.any(maximum[:2] > source_maximum[:2] + 1e-6):
            raise RuntimeError("Adaptive ROI trim would expand beyond the validated source ROI")
        candidate = mesh if padding == ROI_TRIM_PADDING_CANDIDATES_M[-1] else mesh.crop(
            o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
        )
        if not candidate.has_triangles():
            rows.append({"padding_m": padding, "status": "empty"})
            continue
        distances = cKDTree(np.asarray(candidate.vertices)).query(support, k=1, workers=1)[0]
        supported_fraction = float(np.mean(distances <= SPARSE_SUPPORT_DISTANCE_M))
        support_drop = max(0.0, baseline_fraction - supported_fraction)
        accepted = support_drop <= MAXIMUM_ROI_TRIM_SUPPORT_DROP
        rows.append({
            "padding_m": padding,
            "status": "accepted" if accepted else "support_loss",
            "triangle_count": len(candidate.triangles),
            "supported_fraction_within_5cm": supported_fraction,
            "absolute_support_drop": support_drop,
        })
        if accepted:
            chosen = candidate
            chosen_padding = padding
            chosen_minimum = minimum
            chosen_maximum = maximum
            break
    if not chosen.has_triangles():
        raise RuntimeError("Adaptive ROI trim produced an empty selected surface")
    if not rows or rows[-1]["status"] != "accepted":
        raise RuntimeError("No ROI trim candidate preserved metric sparse surface support")
    return chosen, {
        "method": "smallest sparse-supported XY padding around target quantile bounds",
        "candidate_padding_m": list(ROI_TRIM_PADDING_CANDIDATES_M),
        "maximum_absolute_sparse_support_drop": MAXIMUM_ROI_TRIM_SUPPORT_DROP,
        "reference_support_point_count": len(support),
        "source_supported_fraction_within_5cm": baseline_fraction,
        "selected_padding_m": chosen_padding,
        "source_triangle_count": len(source_triangles),
        "selected_triangle_count": len(chosen.triangles),
        "effective_roi_world_lio_m": {
            "minimum": chosen_minimum.tolist(),
            "maximum": chosen_maximum.tolist(),
        },
        "candidates": rows,
    }


def select_surface_candidate(fused: dict, visual: dict, broad: dict) -> tuple[str, dict]:
    fused_fragmented = (
        fused["dominant_component_area_fraction"] < MAXIMUM_FRAGMENTED_DOMINANT_AREA_FRACTION
        and fused["component_count"] > MINIMUM_FRAGMENTED_COMPONENT_COUNT
    )
    visual_coherent = (
        visual["dominant_component_area_fraction"] >= MINIMUM_COHERENT_DOMINANT_AREA_FRACTION
        and visual["component_count"] <= MAXIMUM_COHERENT_COMPONENT_COUNT
    )
    visual_supported = (
        visual["sparse_support_fraction_within_5cm"] >= MINIMUM_VISUAL_SPARSE_SUPPORT_FRACTION
        and visual["sparse_support_fraction_within_5cm"]
        >= fused["sparse_support_fraction_within_5cm"] - MAXIMUM_VISUAL_SUPPORT_DROP
    )
    broad_coherent = (
        broad["dominant_component_area_fraction"] >= MINIMUM_COHERENT_DOMINANT_AREA_FRACTION
        and broad["component_count"] <= MAXIMUM_COHERENT_COMPONENT_COUNT
    )
    broad_supported = (
        broad["sparse_support_fraction_within_5cm"] >= MINIMUM_VISUAL_SPARSE_SUPPORT_FRACTION
        and broad["sparse_support_fraction_within_5cm"]
        >= fused["sparse_support_fraction_within_5cm"] - MAXIMUM_VISUAL_SUPPORT_DROP
    )
    broad_triangle_gain = (
        broad["triangle_count"] / max(visual["triangle_count"], 1) - 1.0
    )
    broad_adds_supported_surface = (
        MINIMUM_BROAD_TRIANGLE_GAIN_FRACTION <= broad_triangle_gain
        <= MAXIMUM_BROAD_TRIANGLE_GAIN_FRACTION
        and broad["sparse_support_fraction_within_5cm"]
        >= visual["sparse_support_fraction_within_5cm"]
    )
    selected = "registered_lidar_fused"
    if fused_fragmented and visual_coherent and visual_supported:
        if broad_coherent and broad_supported and broad_adds_supported_surface:
            selected = "pgsr_depth_broad"
        else:
            selected = "pgsr_depth_only"
    return selected, {
        "method": "deterministic connected-surface and metric-sparse-support selection",
        "selected_mode": selected,
        "fused_fragmented": fused_fragmented,
        "visual_coherent": visual_coherent,
        "visual_supported": visual_supported,
        "broad_coherent": broad_coherent,
        "broad_supported": broad_supported,
        "broad_triangle_gain_fraction": broad_triangle_gain,
        "broad_adds_supported_surface": broad_adds_supported_surface,
        "thresholds": {
            "maximum_fragmented_dominant_area_fraction": MAXIMUM_FRAGMENTED_DOMINANT_AREA_FRACTION,
            "minimum_fragmented_component_count": MINIMUM_FRAGMENTED_COMPONENT_COUNT,
            "minimum_coherent_dominant_area_fraction": MINIMUM_COHERENT_DOMINANT_AREA_FRACTION,
            "maximum_coherent_component_count": MAXIMUM_COHERENT_COMPONENT_COUNT,
            "minimum_visual_sparse_support_fraction": MINIMUM_VISUAL_SPARSE_SUPPORT_FRACTION,
            "maximum_visual_support_drop": MAXIMUM_VISUAL_SUPPORT_DROP,
            "minimum_broad_triangle_gain_fraction": MINIMUM_BROAD_TRIANGLE_GAIN_FRACTION,
            "maximum_broad_triangle_gain_fraction": MAXIMUM_BROAD_TRIANGLE_GAIN_FRACTION,
            "sparse_support_distance_m": SPARSE_SUPPORT_DISTANCE_M,
        },
        "candidates": {
            "registered_lidar_fused": fused,
            "pgsr_depth_only": visual,
            "pgsr_depth_broad": broad,
        },
    }


def unique_image_names(names, label: str) -> tuple[set[str], set[str]]:
    full_names = list(names)
    stems = [Path(name).stem for name in full_names]
    if (
        not full_names
        or any(not name or not stem for name, stem in zip(full_names, stems))
        or len(set(full_names)) != len(full_names)
        or len(set(stems)) != len(stems)
    ):
        raise RuntimeError(f"{label} image names are empty or ambiguous")
    return set(full_names), set(stems)


def assess_lidar_view_alignment(
    lidar_receipt: dict, metric_input: dict, view_names, manifest_path: Path
) -> dict:
    manifest_path = manifest_path.resolve(strict=True)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != lidar_receipt["image_manifest_sha256"] or manifest_hash != metric_input[
        "source_registered_manifest_sha256"
    ]:
        raise RuntimeError("LiDAR and PGSR inputs do not share the selected image manifest")
    selected, _ = unique_image_names(
        (row["image_name"] for row in read_json(manifest_path)), "Selected manifest"
    )
    lidar, lidar_stems = unique_image_names(
        (row["image_name"] for row in lidar_receipt["per_frame"]), "LiDAR receipt"
    )
    trained, trained_stems = unique_image_names(
        (row["name"] for row in metric_input["images"]), "Metric PGSR input"
    )
    _, view_stems = unique_image_names(view_names, "Loaded PGSR views")
    if lidar != selected or len(lidar) != int(lidar_receipt["image_count"]):
        raise RuntimeError("LiDAR anchors do not exactly cover selected images")
    if not trained.issubset(selected) or not trained_stems.issubset(lidar_stems):
        raise RuntimeError("PGSR training includes an image without selected LiDAR anchors")
    if view_stems != trained_stems or len(view_stems) != int(metric_input["registered_camera_count"]):
        raise RuntimeError("Loaded PGSR views disagree with admitted metric training images")
    unused = sorted(selected - trained)
    return {
        "status": "complete_training_subset",
        "selected_manifest_sha256": manifest_hash,
        "selected_frame_count": len(selected),
        "trained_view_count": len(trained),
        "unused_anchor_frame_count": len(unused),
        "unused_anchor_image_names": unused,
    }


def camera_geometry(view):
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.asarray(view.R, dtype=np.float64).T
    world_to_camera[:3, 3] = np.asarray(view.T, dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    return world_to_camera, camera_to_world, camera_to_world[:3, 3]


def camera_points_from_depth(view, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    return np.stack(
        (
            (columns - float(view.Cx)) / float(view.Fx) * depth,
            (rows - float(view.Cy)) / float(view.Fy) * depth,
            depth,
        ),
        axis=-1,
    )


def choose_neighbors(centers: np.ndarray, focus: np.ndarray) -> list[list[int]]:
    offsets = centers[:, :2] - focus[None, :2]
    azimuths = np.arctan2(offsets[:, 1], offsets[:, 0])
    result = []
    for index in range(len(centers)):
        delta = np.arctan2(
            np.sin(azimuths - azimuths[index]), np.cos(azimuths - azimuths[index])
        )
        absolute_degrees = np.abs(np.degrees(delta))
        baseline = np.linalg.norm(centers - centers[index], axis=1)
        valid = (
            (absolute_degrees >= MINIMUM_NEIGHBOR_AZIMUTH_DEGREES)
            & (absolute_degrees <= MAXIMUM_NEIGHBOR_AZIMUTH_DEGREES)
            & (baseline >= MINIMUM_BASELINE_M)
            & (baseline <= MAXIMUM_BASELINE_M)
        )
        negative = np.flatnonzero(valid & (delta < 0))
        positive = np.flatnonzero(valid & (delta > 0))
        negative = negative[np.argsort(absolute_degrees[negative])]
        positive = positive[np.argsort(absolute_degrees[positive])]
        half = NEIGHBORS_PER_VIEW // 2
        chosen = list(negative[:half]) + list(positive[:half])
        if len(chosen) < NEIGHBORS_PER_VIEW:
            remaining = np.flatnonzero(valid & ~np.isin(np.arange(len(centers)), chosen))
            remaining = remaining[np.argsort(absolute_degrees[remaining])]
            chosen.extend(remaining[: NEIGHBORS_PER_VIEW - len(chosen)].tolist())
        if len(chosen) < MINIMUM_NEIGHBOR_SUPPORT:
            raise RuntimeError(f"Camera {index} has only {len(chosen)} usable multi-view neighbors")
        result.append([int(value) for value in chosen])
    return result


def project_lidar_depth(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    view,
    height: int,
    width: int,
) -> np.ndarray:
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.05) & (camera[:, 2] < MAXIMUM_DEPTH_M)
    camera = camera[valid]
    columns = np.rint(camera[:, 0] / camera[:, 2] * float(view.Fx) + float(view.Cx)).astype(np.int64)
    rows = np.rint(camera[:, 1] / camera[:, 2] * float(view.Fy) + float(view.Cy)).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    rows = rows[inside]
    columns = columns[inside]
    depths = camera[inside, 2].astype(np.float32)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flat, rows * width + columns, depths)
    result = flat.reshape(height, width)
    if LIDAR_ANCHOR_DILATION_PIXELS > 0:
        size = 2 * LIDAR_ANCHOR_DILATION_PIXELS + 1
        result = minimum_filter(result, size=size, mode="constant", cval=np.inf)
    result[~np.isfinite(result)] = 0.0
    return result


def integrate_depth(volume, color: np.ndarray, depth: np.ndarray, view, world_to_camera):
    height, width = depth.shape
    depth_u16 = np.ascontiguousarray(np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(depth_u16),
        depth_scale=1000.0,
        depth_trunc=MAXIMUM_DEPTH_M,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stage-name", default="06_pgsr_inlier")
    parser.add_argument("--roi-receipt", required=True, type=Path)
    parser.add_argument("--lidar-receipt", required=True, type=Path)
    parser.add_argument("--output-stage", default="07_reliable_lidar_tsdf")
    parser.add_argument("--iteration", type=int, default=15000)
    args = parser.parse_args()

    run = args.run_root.resolve(strict=True)
    stage = run / args.stage_name
    source = args.source.resolve(strict=True)
    output = run / args.output_stage
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    training_receipt = read_json(stage / "training_receipt.json")
    metric_input = read_json(stage / "metric_input_receipt.json")
    roi_receipt_path = args.roi_receipt.resolve(strict=True)
    lidar_receipt_path = args.lidar_receipt.resolve(strict=True)
    roi_receipt = read_json(roi_receipt_path)
    lidar_receipt = read_json(lidar_receipt_path)
    if training_receipt.get("language_model_used") is not False:
        raise RuntimeError("PGSR training does not satisfy the no-language-model contract")
    if lidar_receipt.get("status") != "complete" or roi_receipt.get("status") != "complete":
        raise RuntimeError("ROI or LiDAR anchor receipt is incomplete")
    if (
        lidar_receipt.get("schema_version") != "e26_stable_registered_lidar_anchors_v2"
        or lidar_receipt.get("anchor_coverage", {}).get("status")
        not in ("strong_all_frames", "isolated_sparse_frames")
    ):
        raise RuntimeError("LiDAR anchor coverage contract is missing or invalid")
    roi_minimum = np.asarray(roi_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    roi_maximum = np.asarray(roi_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    focus = np.asarray(roi_receipt["roi_derivation"]["focus_world_lio_m"], dtype=np.float64)
    lidar_rows = {Path(row["image_name"]).stem: row for row in lidar_receipt["per_frame"]}

    sys.path.insert(0, str(source))
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    model = stage / f"model_pgsr_{args.iteration}"
    pgsr_parser = ArgumentParser()
    model_params = ModelParams(pgsr_parser, sentinel=True)
    pipeline_params = PipelineParams(pgsr_parser)
    pgsr_parser.add_argument("--iteration", default=args.iteration, type=int)
    original_argv = sys.argv
    sys.argv = ["reliable_lidar_tsdf", "-m", str(model), "--iteration", str(args.iteration)]
    try:
        combined = get_combined_args(pgsr_parser)
    finally:
        sys.argv = original_argv
    dataset = model_params.extract(combined)
    pipeline = pipeline_params.extract(combined)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    if len(views) != int(training_receipt["registered_camera_count"]):
        raise RuntimeError("Loaded camera count disagrees with the training receipt")
    image_alignment = assess_lidar_view_alignment(
        lidar_receipt,
        metric_input,
        (view.image_name for view in views),
        run / "01_visual_workspace" / "registered_image_manifest.json",
    )

    geometry = [camera_geometry(view) for view in views]
    centers = np.stack([item[2] for item in geometry])
    neighbors = choose_neighbors(centers, focus)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    depths = []
    base_masks = []
    broad_masks = []
    colors = []
    render_rows = []
    started = time.monotonic()
    with torch.no_grad():
        for index, view in enumerate(views):
            rendered = render(view, gaussians, pipeline, background)
            depth_tensor = rendered["plane_depth"].squeeze()
            depth_normal = torch.nn.functional.normalize(
                rendered["depth_normal"].permute(1, 2, 0), p=2, dim=-1
            )
            view_direction = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            angle = torch.acos(
                torch.sum(view_direction * depth_normal, dim=-1).abs().clamp(0.0, 1.0)
            )
            depth = depth_tensor.cpu().numpy().astype(np.float32)
            alpha = rendered["rendered_alpha"].squeeze().cpu().numpy().astype(np.float32)
            camera_points = camera_points_from_depth(view, depth)
            camera_to_world = geometry[index][1]
            world_points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
            angle_array = angle.cpu().numpy()
            roi_depth_valid = (
                np.isfinite(depth)
                & (depth > 0.0)
                & (depth < MAXIMUM_DEPTH_M)
                & np.all(world_points >= roi_minimum, axis=-1)
                & np.all(world_points <= roi_maximum, axis=-1)
            )
            mask = (
                roi_depth_valid
                & (alpha >= MINIMUM_RENDERED_ALPHA)
                & (angle_array <= np.deg2rad(MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES))
            )
            broad_mask = (
                roi_depth_valid
                & (angle_array <= np.deg2rad(BROAD_MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES))
            )
            color = (
                rendered["render"].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            depths.append(depth)
            base_masks.append(mask)
            broad_masks.append(broad_mask)
            colors.append(np.ascontiguousarray(color))
            render_rows.append(
                {
                    "image_name": view.image_name,
                    "base_valid_pixel_count": int(np.count_nonzero(mask)),
                    "broad_valid_pixel_count": int(np.count_nonzero(broad_mask)),
                }
            )
            if (index + 1) % 20 == 0 or index + 1 == len(views):
                print(f"rendered {index + 1}/{len(views)}", flush=True)
            del rendered, depth_tensor, depth_normal, view_direction, angle

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE_M,
        sdf_trunc=SDF_TRUNCATION_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    total_base = 0
    total_trusted = 0
    total_lidar_pixels = 0
    total_lidar_overrides = 0
    for index, view in enumerate(views):
        depth = depths[index]
        base = base_masks[index]
        flat_indices = np.flatnonzero(base.reshape(-1))
        camera_points = camera_points_from_depth(view, depth).reshape(-1, 3)[flat_indices]
        camera_to_world = geometry[index][1]
        world_points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        support = np.zeros(len(flat_indices), dtype=np.uint8)
        residual_samples = []
        for neighbor_index in neighbors[index]:
            neighbor_view = views[neighbor_index]
            world_to_neighbor = geometry[neighbor_index][0]
            camera = world_points @ world_to_neighbor[:3, :3].T + world_to_neighbor[:3, 3]
            positive = camera[:, 2] > 0.05
            columns = np.rint(camera[:, 0] / np.maximum(camera[:, 2], 1e-9) * float(neighbor_view.Fx) + float(neighbor_view.Cx)).astype(np.int64)
            rows = np.rint(camera[:, 1] / np.maximum(camera[:, 2], 1e-9) * float(neighbor_view.Fy) + float(neighbor_view.Cy)).astype(np.int64)
            height, width = depths[neighbor_index].shape
            inside = positive & (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
            candidate = np.flatnonzero(inside)
            if len(candidate) == 0:
                continue
            sampled_depth = depths[neighbor_index][rows[candidate], columns[candidate]]
            sampled_valid = base_masks[neighbor_index][rows[candidate], columns[candidate]]
            residual = np.abs(sampled_depth - camera[candidate, 2])
            tolerance = np.maximum(
                DEPTH_ABSOLUTE_TOLERANCE_M,
                DEPTH_RELATIVE_TOLERANCE * camera[candidate, 2],
            )
            consistent = sampled_valid & (residual <= tolerance)
            support[candidate[consistent]] += 1
            if np.any(consistent):
                residual_samples.append(residual[consistent])
        trusted_flat = np.zeros(depth.size, dtype=bool)
        trusted_flat[flat_indices] = support >= MINIMUM_NEIGHBOR_SUPPORT
        trusted = trusted_flat.reshape(depth.shape)
        lidar_row = lidar_rows[Path(view.image_name).stem]
        lidar_path = Path(lidar_row["path"])
        if not lidar_path.is_absolute():
            lidar_path = lidar_receipt_path.parent / lidar_path
        lidar_points = np.load(lidar_path)["points_world_lio_m"]
        lidar_depth = project_lidar_depth(
            lidar_points,
            geometry[index][0],
            view,
            depth.shape[0],
            depth.shape[1],
        )
        lidar_valid = lidar_depth > 0
        fused_depth = depth.copy()
        fused_depth[~trusted] = 0.0
        disagreements = lidar_valid & (fused_depth > 0) & (
            np.abs(fused_depth - lidar_depth) > 0.05
        )
        fused_depth[lidar_valid] = lidar_depth[lidar_valid]
        integrate_depth(volume, colors[index], fused_depth, view, geometry[index][0])
        lidar_only = np.where(lidar_valid, lidar_depth, 0.0).astype(np.float32)
        for _ in range(LIDAR_EXTRA_INTEGRATIONS):
            integrate_depth(volume, colors[index], lidar_only, view, geometry[index][0])
        base_count = int(np.count_nonzero(base))
        trusted_count = int(np.count_nonzero(trusted))
        lidar_count = int(np.count_nonzero(lidar_valid))
        total_base += base_count
        total_trusted += trusted_count
        total_lidar_pixels += lidar_count
        total_lidar_overrides += int(np.count_nonzero(disagreements))
        render_rows[index].update(
            {
                "neighbor_indices": neighbors[index],
                "multi_view_trusted_pixel_count": trusted_count,
                "multi_view_trusted_fraction_of_base": trusted_count / max(base_count, 1),
                "lidar_anchor_pixel_count": lidar_count,
                "lidar_override_disagreement_count": int(np.count_nonzero(disagreements)),
                "consistent_depth_residual_p50_m": (
                    float(np.quantile(np.concatenate(residual_samples), 0.50)) if residual_samples else None
                ),
                "consistent_depth_residual_p95_m": (
                    float(np.quantile(np.concatenate(residual_samples), 0.95)) if residual_samples else None
                ),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(views):
            heartbeat = {
                "status": "running",
                "integrated_frames": index + 1,
                "frame_count": len(views),
                "total_base_pixels": total_base,
                "total_multi_view_trusted_pixels": total_trusted,
                "total_lidar_anchor_pixels": total_lidar_pixels,
            }
            (output / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2) + "\n", encoding="utf-8")
            print(f"integrated {index + 1}/{len(views)}", flush=True)
    trusted_fraction = total_trusted / max(total_base, 1)
    if trusted_fraction < 0.05:
        raise RuntimeError(f"Multi-view trusted depth fraction is too low: {trusted_fraction:.6f}")
    if total_lidar_pixels < 10000:
        raise RuntimeError(f"Projected stable LiDAR anchor support is too low: {total_lidar_pixels}")
    fused_mesh = volume.extract_triangle_mesh()
    del volume
    fused_mesh.compute_vertex_normals()
    roi_box = o3d.geometry.AxisAlignedBoundingBox(roi_minimum, roi_maximum)
    fused_mesh = fused_mesh.crop(roi_box)
    if not fused_mesh.has_vertices() or not fused_mesh.has_triangles():
        raise RuntimeError("Reliable LiDAR TSDF extraction returned an empty mesh")
    visual_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE_M,
        sdf_trunc=SDF_TRUNCATION_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, view in enumerate(views):
        visual_depth = np.where(base_masks[index], depths[index], 0.0).astype(np.float32)
        integrate_depth(visual_volume, colors[index], visual_depth, view, geometry[index][0])
        if (index + 1) % 20 == 0 or index + 1 == len(views):
            print(f"integrated visual candidate {index + 1}/{len(views)}", flush=True)
    visual_mesh = visual_volume.extract_triangle_mesh()
    del visual_volume
    visual_mesh.compute_vertex_normals()
    visual_mesh = visual_mesh.crop(roi_box)
    if not visual_mesh.has_vertices() or not visual_mesh.has_triangles():
        raise RuntimeError("PGSR-only TSDF extraction returned an empty mesh")

    broad_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE_M,
        sdf_trunc=SDF_TRUNCATION_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, view in enumerate(views):
        broad_depth = np.where(broad_masks[index], depths[index], 0.0).astype(np.float32)
        integrate_depth(broad_volume, colors[index], broad_depth, view, geometry[index][0])
        if (index + 1) % 20 == 0 or index + 1 == len(views):
            print(f"integrated broad visual candidate {index + 1}/{len(views)}", flush=True)
    broad_mesh = broad_volume.extract_triangle_mesh()
    del broad_volume
    broad_mesh.compute_vertex_normals()
    broad_mesh = broad_mesh.crop(roi_box)
    if not broad_mesh.has_vertices() or not broad_mesh.has_triangles():
        raise RuntimeError("Broad PGSR TSDF extraction returned an empty mesh")

    support = metric_sparse_support(stage, roi_receipt)
    fused_metrics = surface_candidate_metrics(fused_mesh, support)
    visual_metrics = surface_candidate_metrics(visual_mesh, support)
    broad_metrics = surface_candidate_metrics(broad_mesh, support)
    selected_mode, selection = select_surface_candidate(fused_metrics, visual_metrics, broad_metrics)
    candidate_paths = {
        "registered_lidar_fused": output / "candidate_registered_lidar_fused_mesh.ply",
        "pgsr_depth_only": output / "candidate_pgsr_depth_only_mesh.ply",
        "pgsr_depth_broad": output / "candidate_pgsr_depth_broad_mesh.ply",
    }
    for mode, candidate in (
        ("registered_lidar_fused", fused_mesh),
        ("pgsr_depth_only", visual_mesh),
        ("pgsr_depth_broad", broad_mesh),
    ):
        if not o3d.io.write_triangle_mesh(str(candidate_paths[mode]), candidate, write_ascii=False):
            raise RuntimeError(f"Failed to write {mode} TSDF candidate")
    selection["candidate_meshes"] = {
        mode: {"path": str(path), "sha256": sha256_file(path)}
        for mode, path in candidate_paths.items()
    }
    meshes = {
        "registered_lidar_fused": fused_mesh,
        "pgsr_depth_only": visual_mesh,
        "pgsr_depth_broad": broad_mesh,
    }
    mesh, roi_trim = trim_surface_roi_by_sparse_support(
        meshes[selected_mode], support, roi_receipt
    )
    raw_mesh = output / "tesla_reliable_lidar_pgsr_5mm_raw_mesh.ply"
    if roi_trim["selected_padding_m"] == ROI_TRIM_PADDING_CANDIDATES_M[-1]:
        shutil.copyfile(candidate_paths[selected_mode], raw_mesh)
    elif not o3d.io.write_triangle_mesh(str(raw_mesh), mesh, write_ascii=False):
        raise RuntimeError("Failed to write sparse-supported trimmed TSDF surface")
    receipt = {
        "schema_version": "e26_reliable_lidar_pgsr_tsdf_v1",
        "status": "raw_s08_mesh_generated",
        "source_model": str(model),
        "source_iteration": args.iteration,
        "source_training_receipt": str(stage / "training_receipt.json"),
        "source_roi_receipt": str(roi_receipt_path),
        "source_lidar_receipt": str(lidar_receipt_path),
        "lidar_anchor_coverage": lidar_receipt["anchor_coverage"],
        "image_alignment": image_alignment,
        "language_model_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "known_object_dimensions_used": False,
        "roi_derivation": roi_receipt["roi_derivation"],
        "roi_world_lio_m": roi_receipt["roi_world_lio_m"],
        "multi_view_gate": {
            "neighbor_count": NEIGHBORS_PER_VIEW,
            "minimum_neighbor_support": MINIMUM_NEIGHBOR_SUPPORT,
            "neighbor_azimuth_degrees": [MINIMUM_NEIGHBOR_AZIMUTH_DEGREES, MAXIMUM_NEIGHBOR_AZIMUTH_DEGREES],
            "neighbor_baseline_m": [MINIMUM_BASELINE_M, MAXIMUM_BASELINE_M],
            "minimum_rendered_alpha": MINIMUM_RENDERED_ALPHA,
            "maximum_normal_view_angle_degrees": MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES,
            "absolute_depth_tolerance_m": DEPTH_ABSOLUTE_TOLERANCE_M,
            "relative_depth_tolerance": DEPTH_RELATIVE_TOLERANCE,
            "base_pixel_count": total_base,
            "trusted_pixel_count": total_trusted,
            "trusted_fraction": trusted_fraction,
        },
        "lidar_anchor_fusion": {
            "applied_to_selected_mesh": selected_mode == "registered_lidar_fused",
            "projected_anchor_pixel_count": total_lidar_pixels,
            "pgsr_disagreement_override_count": total_lidar_overrides,
            "dilation_pixels": LIDAR_ANCHOR_DILATION_PIXELS,
            "extra_tsdf_integrations": LIDAR_EXTRA_INTEGRATIONS,
        },
        "broad_visual_gate": {
            "maximum_normal_view_angle_degrees": BROAD_MAXIMUM_NORMAL_VIEW_ANGLE_DEGREES,
            "rendered_alpha_filter": "none",
            "valid_pixel_count": sum(row["broad_valid_pixel_count"] for row in render_rows),
        },
        "surface_candidate_selection": selection,
        "roi_boundary_trim": roi_trim,
        "voxel_size_m": VOXEL_SIZE_M,
        "sdf_truncation_m": SDF_TRUNCATION_M,
        "frame_count": len(views),
        "mesh": {
            "path": str(raw_mesh),
            "sha256": sha256_file(raw_mesh),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
        },
        "runtime_seconds": time.monotonic() - started,
        "per_frame": render_rows,
        "scientific_boundary": (
            "The selected surface uses multi-view checked PGSR depths and registered LiDAR "
            "depth overrides with two extra anchor integrations. Missing glass remains unknown."
            if selected_mode == "registered_lidar_fused" else
            "The selected surface uses PGSR depths after view-angle and metric ROI checks; "
            "the strict candidate also requires rendered opacity. Registered LiDAR supports "
            "the metric scene and candidate audit but is not directly fused into this selected "
            "mesh. Missing glass remains unknown."
        ),
    }
    (output / "reliable_lidar_tsdf_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    (output / "heartbeat.json").write_text(
        json.dumps({"status": "complete", "mesh": receipt["mesh"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
