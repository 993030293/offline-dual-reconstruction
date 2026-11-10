"""Inspect observed sparse support without treating a display ROI as a surface label."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil
import pycolmap

V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def quantiles(values):
    return np.quantile(values, [0, .05, .5, .95, 1]).tolist() if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "roi", "exclusions", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    started = time.monotonic()
    model_path = args.dataset/"sparse/0"
    inputs = [data_identity(model_path/name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    inputs += [data_identity(args.roi), data_identity(args.exclusions)]
    contract = {"maximum_seconds": 300, "maximum_RSS_bytes": 4*1024**3,
        "maximum_output_bytes": 64*1024**2, "source": data_identity(Path(__file__)), "inputs": inputs,
        "read_only": True, "upper_band": "highest20percent_of_nonfloor_ROI_height_range",
        "nonfloor_minimum_m": .08, "floor_and_ROI_source": str(args.roi),
        "counts_are_observed_sparse_tracks_not_visibility_or_GT": True,
        "diagnostic_reliability": {"minimum_track": 5, "minimum_max_parallax_degrees": 3,
            "maximum_mean_reprojection_px": 1}, "counted": False}
    atomic_json(args.output/"CONTRACT.json", contract)
    model = pycolmap.Reconstruction(model_path)
    receipt = json.loads(args.roi.read_text(encoding="utf-8-sig"))
    excluded = json.loads(args.exclusions.read_text(encoding="utf-8-sig"))
    excluded_ids = {int(key) for key in excluded["frames"]}
    images = {im.image_id: im for im in model.images.values() if im.has_pose}
    if len(images) != 1016:
        raise ValueError("The frozen1016-input map is required")
    if any(int(im.name.split("_")[1]) in excluded_ids for im in images.values()):
        raise ValueError("A development excluded image entered the map")
    centers = {key: im.projection_center() for key, im in images.items()}
    transforms = {key: im.cam_from_world() for key, im in images.items()}
    plane = receipt["roi_derivation"]["spatial_sparse_floor_plane"]
    normal, offset = np.array(plane["normal"]), plane["offset_m"]
    bounds = receipt["roi_world_lio_m"]
    lo, hi = np.array(bounds["minimum"]), np.array(bounds["maximum"])
    point_ids = sorted(model.points3D.keys())
    xyz = np.array([model.points3D[key].xyz for key in point_ids])
    height = xyz@normal+offset
    inside = ((xyz >= lo) & (xyz <= hi)).all(1) & (height > .08)
    eligible = np.flatnonzero(inside)
    if not len(eligible):
        raise ValueError("No nonfloor sparse support inside the existing ROI")
    lower, upper = np.min(height[eligible]), np.max(height[eligible])
    cutoff = lower+.8*(upper-lower)
    point_rows, observations = [], defaultdict(list)
    peak = 0
    for count, index in enumerate(eligible):
        key = point_ids[index]
        point = model.points3D[key]
        track = list(point.track.elements)
        rays = np.array([centers[item.image_id]-point.xyz for item in track])
        norms = np.linalg.norm(rays, axis=1)
        if not np.isfinite(rays).all() or (norms <= 0).any():
            raise ValueError("Invalid observed ray")
        rays /= norms[:, None]
        # Acute parallax treats opposite collinear rays as degenerate too.
        cosine = np.clip(np.abs(rays@rays.T), 0, 1)
        angle = float(np.degrees(np.arccos(cosine.min())))
        reliable = len(track) >= 5 and angle >= 3 and point.error <= 1
        high = bool(height[index] >= cutoff)
        point_rows.append({"point3D_id": int(key), "xyz": point.xyz.tolist(),
            "height_above_estimated_floor_m": float(height[index]), "upper_band": high,
            "track_length": len(track), "max_acute_parallax_degrees": angle,
            "mean_reprojection_px": float(point.error), "diagnostic_reliable": bool(reliable)})
        for item in track:
            im = images[item.image_id]
            xy = im.points2D[item.point2D_idx].xy
            camera = model.cameras[im.camera_id]
            cam_xyz = transforms[item.image_id]*point.xyz
            if cam_xyz[2] <= 0:
                raise ValueError("An observed target point has nonpositive depth")
            pred = camera.img_from_cam(cam_xyz)
            if not np.isfinite(pred).all():
                raise ValueError("An observed target point has invalid projection")
            if high:
                observations[item.image_id].append({"point3D_id": int(key), "xy": xy.tolist(),
                    "z_depth_m": float(cam_xyz[2]), "residual_px": float(np.linalg.norm(pred-xy)),
                    "diagnostic_reliable": bool(reliable)})
        if count % 1000 == 0:
            peak = max(peak, psutil.Process().memory_info().rss)
            if time.monotonic()-started > 300 or peak > 4*1024**3:
                raise RuntimeError("Target support diagnostic budget exceeded")
    camera_rows = []
    for key, im in images.items():
        obs = observations[key]
        camera = model.cameras[im.camera_id]
        cells = {tuple(np.minimum(np.maximum(np.array(ob["xy"])/[camera.width, camera.height]*8, 0), 7).astype(int)) for ob in obs}
        camera_rows.append({"image_name": im.name, "image_id": int(key),
            "center": centers[key].tolist(), "height_above_estimated_floor_m": float(centers[key]@normal+offset),
            "upper_observed_points": len(obs), "upper_reliable_points": sum(ob["diagnostic_reliable"] for ob in obs),
            "upper_occupied_8x8_cells": len(cells)})
    def stats(rows):
        return {"point_count": len(rows), "track_length_min_p05_p50_p95_max": quantiles([p["track_length"] for p in rows]),
            "max_acute_parallax_deg_min_p05_p50_p95_max": quantiles([p["max_acute_parallax_degrees"] for p in rows]),
            "mean_reprojection_px_min_p05_p50_p95_max": quantiles([p["mean_reprojection_px"] for p in rows]),
            "diagnostic_reliable_count": sum(p["diagnostic_reliable"] for p in rows)}
    high_rows = [p for p in point_rows if p["upper_band"]]
    # Choose by geometry only, before inspecting image appearance.
    ranked_support = sorted(camera_rows, key=lambda r: (-r["upper_observed_points"], r["image_name"]))
    ranked_height = sorted((r for r in camera_rows if r["upper_observed_points"]),
        key=lambda r: (-r["height_above_estimated_floor_m"], r["image_name"]))
    selected = []
    for order in (ranked_support, ranked_height):
        for row in order:
            if row["image_name"] not in {r["image_name"] for r in selected}:
                selected.append(row)
            if len(selected) >= (3 if order is ranked_support else 6):
                break
    tiles = []
    for row in selected:
        original = cv2.imread(str(args.dataset/"images"/row["image_name"]))
        if original is None:
            raise ValueError("A frozen source image cannot be decoded")
        camera = model.cameras[images[row["image_id"]].camera_id]
        if original.shape[:2] != (camera.height, camera.width):
            raise ValueError("Observed feature pixels and source image dimensions differ")
        image = cv2.resize(original, (406, 310), interpolation=cv2.INTER_AREA)
        annotated = image.copy()
        sy, sx = 310/original.shape[0], 406/original.shape[1]
        for ob in observations[row["image_id"]]:
            xy = tuple(np.rint(np.array(ob["xy"])*[sx, sy]).astype(int))
            cv2.circle(annotated, xy, 1, (0, 255, 0) if ob["diagnostic_reliable"] else (0, 128, 255), -1)
        tile = np.vstack((image, annotated))
        cv2.putText(tile, row["image_name"].split("_")[1]+" count="+str(row["upper_observed_points"]),
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if len(tiles) != 6:
        raise ValueError("Insufficient observed views for the registered six-view diagnostic")
    if not cv2.imwrite(str(args.output/"UPPER_SUPPORT_SIX_VIEWS.jpg"), np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))):
        raise OSError("Could not save the actual support image")
    atomic_json(args.output/"POINT_SUPPORT.json", point_rows)
    atomic_json(args.output/"CAMERA_SUPPORT.json", camera_rows)
    atomic_json(args.output/"SELECTED_VIEWS.json", selected)
    original_inputs = [data_identity(Path(row["path"])) for row in inputs]
    if inputs != original_inputs:
        raise ValueError("Source map changed during the read-only diagnostic")
    total_bytes = sum(path.stat().st_size for path in args.output.iterdir() if path.is_file())
    if total_bytes > 64*1024**2:
        raise RuntimeError("Output budget exceeded")
    result = {"status": "diagnostic_complete_visual_interpretation_pending", "map_images": len(images),
        "map_points": len(point_ids), "nonfloor_ROI": stats(point_rows), "upper_band": stats(high_rows),
        "upper_height_cutoff_m": float(cutoff), "height_range_m": [float(lower), float(upper)],
        "upper_observing_cameras": sum(r["upper_observed_points"] > 0 for r in camera_rows),
        "upper_observations": sum(r["upper_observed_points"] for r in camera_rows),
        "upper_cameras_at_least30_points": sum(r["upper_observed_points"] >= 30 for r in camera_rows),
        "elapsed_seconds": time.monotonic()-started, "peak_sampled_RSS_bytes": peak,
        "output_bytes": total_bytes, "source_inputs_unchanged": True, "heldout_used": False,
        "surface_classification": "unknown_until_image_review", "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
