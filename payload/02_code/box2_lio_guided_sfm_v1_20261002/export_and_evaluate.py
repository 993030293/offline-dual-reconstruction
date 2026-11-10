"""Export a consistent optimized model using the frozen geometry policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

from common import PACKAGE, Heartbeat, configure_stdout, read_json, write_json
from run_prior_ba import observation_metrics


def export(run, case, model_path, image_path):
    output = run / "04_downstream" / case
    output.mkdir(parents=True, exist_ok=False)
    heartbeat = Heartbeat(output / "EXPORT_STATE.json")
    start = time.monotonic()
    try:
        model = pycolmap.Reconstruction(model_path)
        model.update_point_3d_errors()
        policy_path = PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"
        policy = read_json(policy_path)
        sys.path.insert(0, str(PACKAGE / "code/validated_fresh_runtime"))
        from metric_admission import camera_geometry_metrics
        geometry = [camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                    for im in sorted(model.images.values(), key=lambda x: x.image_id) if im.has_pose]
        rejected_cameras = [r["image_name"] for r in geometry if not r["passed"]]
        write_json(output / "CAMERA_GEOMETRY.json", {"rows": geometry, "rejected": rejected_cameras})
        if rejected_cameras:
            raise RuntimeError("Camera admission rejected views; preserve the paired image set and diagnose")
        centers = np.asarray([im.projection_center() for im in model.images.values() if im.has_pose])
        point_policy = policy["point_filter"]
        if np.linalg.norm(centers, axis=1).max() > point_policy["maximum_camera_radius_m"]:
            raise RuntimeError("Camera extent violates the existing metric policy")
        lo, hi = centers.min(0) - point_policy["camera_aabb_margin_m"], centers.max(0) + point_policy["camera_aabb_margin_m"]
        attempts, keep = [], None
        for minimum_track in (point_policy["minimum_point_track_length"], point_policy["fallback_minimum_point_track_length"]):
            candidate = {i for i, p in model.points3D.items() if np.isfinite(p.xyz).all()
                         and np.all(p.xyz >= lo) and np.all(p.xyz <= hi)
                         and p.error <= point_policy["maximum_point_reprojection_error_px"]
                         and p.track.length() >= minimum_track}
            attempts.append({"minimum_track_length": minimum_track, "retained_points": len(candidate)})
            if len(candidate) >= point_policy["minimum_retained_sparse_points"]:
                keep = candidate
                break
        if keep is None:
            write_json(output / "POINT_FILTER_FAILURE.json", {"status": "failed_verification", "attempts": attempts})
            raise RuntimeError("Point filtering failed the frozen support policy")
        before = model.num_points3D()
        for point_id in list(model.points3D):
            if point_id not in keep:
                model.delete_point3D(point_id)
        filtered = output / "filtered_opencv_model"
        filtered.mkdir()
        model.write(filtered)
        model.export_PLY(output / "optimized_sparse_metric.ply")
        raw_metrics = observation_metrics(model)
        heartbeat.phase = "undistorting_images_and_model"
        dataset = output / "dataset"
        pycolmap.undistort_images(dataset, filtered, image_path, output_type="COLMAP",
                                 num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        undistorted = pycolmap.Reconstruction(dataset / "sparse")
        sparse = dataset / "sparse/0"
        sparse.mkdir()
        undistorted.write(sparse)
        undistorted.write_text(sparse)
        undistorted.export_PLY(sparse / "points3D.ply")
        if any(c.model_name != "PINHOLE" for c in undistorted.cameras.values()):
            raise RuntimeError("The downstream camera is not PINHOLE")
        names = {im.name for im in model.images.values() if im.has_pose}
        if names != {im.name for im in undistorted.images.values() if im.has_pose}:
            raise RuntimeError("Undistortion changed the image identities")
        for im in undistorted.images.values():
            with Image.open(dataset / "images" / im.name) as decoded:
                camera = undistorted.cameras[im.camera_id]
                if decoded.size != (camera.width, camera.height):
                    raise RuntimeError("Downstream image and intrinsics dimensions disagree")
        if set(model.points3D) != set(undistorted.points3D):
            raise RuntimeError("Undistortion changed the filtered point identities")
        center_error = max(np.linalg.norm(im.projection_center() - undistorted.images[i].projection_center())
                           for i, im in model.images.items())
        rotation_error = max(np.max(np.abs(im.cam_from_world().rotation.matrix()
                             - undistorted.images[i].cam_from_world().rotation.matrix())) for i, im in model.images.items())
        point_error = max(np.linalg.norm(p.xyz - undistorted.points3D[i].xyz) for i, p in model.points3D.items())
        if max(center_error, rotation_error, point_error) > 1e-9:
            raise RuntimeError("Undistortion changed camera or point world coordinates")
        write_json(output / "EXPORT_CONTRACT.json", {"status": "passed", "case": case,
            "model_source": model_path, "image_source": image_path, "policy_path": policy_path,
            "input_cameras": model.num_reg_images(), "points_before_filter": before,
            "points_after_filter": model.num_points3D(), "point_filter_attempts": attempts,
            "raw_filtered_metrics": raw_metrics, "undistorted_metrics": observation_metrics(undistorted),
            "PINHOLE_cameras": {i: c.todict() for i, c in undistorted.cameras.items()},
            "max_undistortion_camera_center_delta_m": center_error, "max_rotation_matrix_delta": rotation_error,
            "max_point_world_delta_m": point_error, "dataset": dataset,
            "lidar_map_policy": "recorded_odometry_map_fixed_body_trajectory_not_jointly_optimized",
            "elapsed_seconds": time.monotonic() - start})
        print(f"Exported {case}: {model.num_reg_images()} cameras, {model.num_points3D()} points")
    except Exception as error:
        write_json(output / "EXPORT_FAILURE.json", {"status": "failed_verification", "error": repr(error)})
        raise
    finally:
        heartbeat.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--images", type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.case,
           args.model or args.run / "02_same_180_ba" / args.case / "model",
           args.images or args.run / "00_contract/images")


if __name__ == "__main__":
    configure_stdout()
    main()
