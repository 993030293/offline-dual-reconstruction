"""Apply the existing metric sparse-support policy before joint bundle adjustment."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity
from coarse_prior_provenance import validate_alignment_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alignment-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "02_code/box2_lio_guided_sfm_v1_20261002"))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE / "code/validated_fresh_runtime"))
    from metric_admission import camera_geometry_metrics
    contract = json.loads((args.alignment_run / "CONTRACT.json").read_text())
    source_ids = [data_identity(args.model / name) for name in ["cameras.bin", "images.bin", "points3D.bin"]]
    if [row["sha256"] for row in source_ids] != [row["sha256"] for row in contract["source_model_identity"]]:
        raise ValueError("Metric initialization belongs to a different source map")
    alignment = json.loads((args.alignment_run / "ALIGNMENT.json").read_text())
    validate_alignment_source(alignment)
    policy_path = common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())
    model = pycolmap.Reconstruction(args.model)
    transform = alignment["new_Sim3"]
    model.transform(pycolmap.Sim3d(scale=transform["scale"],
        rotation=pycolmap.Rotation3d(np.asarray(transform["rotation"]["quat"])),
        translation=np.asarray(transform["translation"])))
    model.update_point_3d_errors()
    poses = {image.image_id: image.cam_from_world().matrix().copy() for image in model.images.values() if image.has_pose}
    calibration = {key: camera.params.copy() for key, camera in model.cameras.items()}
    registered_names = sorted(image.name for image in model.images.values() if image.has_pose)
    geometry_before = [camera_geometry_metrics(model, image, policy["camera_geometry_admission"])
        for image in model.images.values() if image.has_pose]
    centers = np.asarray([image.projection_center() for image in model.images.values() if image.has_pose])
    point_policy = policy["point_filter"]
    if np.linalg.norm(centers, axis=1).max() > point_policy["maximum_camera_radius_m"]:
        raise ValueError("The current metric camera extent violates the existing export policy")
    lo = centers.min(0)-point_policy["camera_aabb_margin_m"]
    hi = centers.max(0)+point_policy["camera_aabb_margin_m"]
    attempts, keep = [], None
    for minimum_track in [point_policy["minimum_point_track_length"], point_policy["fallback_minimum_point_track_length"]]:
        selected = {key for key, point in model.points3D.items() if np.isfinite(point.xyz).all()
            and np.all(point.xyz >= lo) and np.all(point.xyz <= hi)
            and point.error <= point_policy["maximum_point_reprojection_error_px"] and point.track.length() >= minimum_track}
        attempts.append({"minimum_track_length": minimum_track, "retained_points": len(selected)})
        if len(selected) >= point_policy["minimum_retained_sparse_points"]:
            keep = selected
            break
    if keep is None:
        atomic_json(args.output / "FAILURE.json", {"status": "insufficient_sparse_support", "attempts": attempts})
        raise ValueError("The unchanged minimum sparse point count was not met")
    before_points, before_observations = model.num_points3D(), model.compute_num_observations()
    removed = []
    for point_id in list(model.points3D):
        if point_id in keep:
            continue
        point = model.points3D[point_id]
        removed.append({"point3D_id": int(point_id), "track_length": point.track.length(),
            "reprojection_error_px": float(point.error) if np.isfinite(point.error) else None,
            "finite_xyz": bool(np.isfinite(point.xyz).all()),
            "inside_camera_aabb": bool(np.all(point.xyz >= lo) and np.all(point.xyz <= hi)),
            "observations": [{"image_id": item.image_id, "point2D_index": item.point2D_idx} for item in point.track.elements]})
        model.delete_point3D(point_id)
    geometry_after = [camera_geometry_metrics(model, image, policy["camera_geometry_admission"])
        for image in model.images.values() if image.has_pose]
    checks = {"all_registered_images_retained": registered_names == sorted(image.name for image in model.images.values() if image.has_pose),
        "camera_poses_unchanged": all(np.array_equal(image.cam_from_world().matrix(), poses[image.image_id])
            for image in model.images.values() if image.has_pose),
        "calibration_unchanged": all(np.array_equal(camera.params, calibration[key]) for key, camera in model.cameras.items()),
        "all_cameras_meet_original_geometry_policy_before": all(row["passed"] for row in geometry_before),
        "all_cameras_meet_original_geometry_policy_after": all(row["passed"] for row in geometry_after)}
    output_model = args.output / "model"
    output_model.mkdir()
    model.write(output_model)
    atomic_json(args.output / "REMOVED_POINTS.json", removed)
    atomic_json(args.output / "CAMERA_SUPPORT.json", {"before": geometry_before, "after": geometry_after})
    result = {"status": "prepared_pending_BA" if all(checks.values()) else "failed_camera_support_policy",
        "checks": checks, "source_model": str(args.model), "source_identity": source_ids,
        "fresh_alignment": data_identity(args.alignment_run / "ALIGNMENT.json"),
        "unchanged_filter_policy": policy["point_filter"], "policy_source": data_identity(policy_path),
        "filter_attempts": attempts, "registered_images": model.num_reg_images(),
        "points_before": before_points, "points_after": model.num_points3D(),
        "observations_before": before_observations, "observations_after": model.compute_num_observations(),
        "removed_points_recorded": len(removed), "failed_before": [row["image_name"] for row in geometry_before if not row["passed"]],
        "failed_after": [row["image_name"] for row in geometry_after if not row["passed"]],
        "model": str(output_model), "elapsed_seconds": time.monotonic()-started, "counted": False,
        "source": data_identity(Path(__file__)), "previous_failed_runs_relabelled": False}
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ["source_identity", "unchanged_filter_policy", "source"]}), flush=True)
    if not all(checks.values()):
        raise ValueError("Do not delete cameras or relax support to accept the prepared geometry")


if __name__ == "__main__":
    main()
