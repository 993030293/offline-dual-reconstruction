"""Prepare a separate all-camera diagnostic without relaxing final geometry gates."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import pycolmap

from runtime import atomic_json, data_identity
from run_metric_refinement import bind_supported_priors, verified_pose_gap_names, projection_validity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "initial", "pose-gap-decisions", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = json.loads((args.run/"RESULT.json").read_text())
    audit = json.loads((args.run/"common_audit/RESULT.json").read_text())
    best = max(source["models"], key=lambda row: row["registered_count"])
    if best["registered_count"] != 1021 or best["original_registered_frames_lost"] or not audit["guard_passed"]:
        raise ValueError("This diagnostic requires the actual complete P20 registration and common-observation audit")
    declared_weak = {r["image_name"] for r in best["weak_cameras"]}
    if not 1 <= len(declared_weak) <= 3:
        raise ValueError("The preregistered weak-camera scope changed")
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_lio_guided_sfm_v1_20261002"))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE/"code/validated_fresh_runtime"))
    from metric_admission import camera_geometry_metrics
    policy_path = common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())
    model_path = Path(best["path"])
    source_identity = [data_identity(model_path/name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    model = pycolmap.Reconstruction(model_path)
    raw = [json.loads(line) for line in (args.initial/"frame_decisions.jsonl").read_text().splitlines()]
    lookup = {r["image_name"]: r for r in raw if r["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
              and r["coarse_pose_valid"] and not r["excluded_role"]}
    allowed_gaps = verified_pose_gap_names(args.initial, args.pose_gap_decisions)
    priors, gaps = bind_supported_priors(model, lookup, allowed_gaps)
    common.validate_prior_rows(model, priors)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/"CONTRACT.json", {"candidate": "P21_WEAK_POSE_RETRIANGULATION",
        "source_model_identity": source_identity, "source_run": data_identity(args.run/"RESULT.json"),
        "source": data_identity(Path(__file__)), "geometry_policy": data_identity(policy_path),
        "raw_input": data_identity(args.initial/"frame_decisions.jsonl"), "registered_count": 1021,
        "declared_weak_cameras": sorted(declared_weak), "supported_prior_count": len(priors), "no_prior_images": gaps,
        "all_cameras_retained": True, "final_geometry_thresholds_unchanged": True,
        "input_weak_cameras_allowed_only_for_diagnostic": True, "maximum_seconds": 300, "maximum_RSS_bytes": 4*1024**3})
    transform = pycolmap.align_reconstruction_to_locations(model, [r["image_name"] for r in priors],
        np.asarray([r["C_W_m"] for r in priors]), 3,
        pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0))
    if transform is None or transform.scale <= 0 or not np.isfinite(transform.scale):
        raise ValueError("Fresh raw metric alignment failed")
    model.transform(transform)
    model.update_point_3d_errors()
    atomic_json(args.output/"ALIGNMENT.json", common.serializable({"new_Sim3": transform.todict(),
        "source": "recorded_raw_camera_positions_not_GT", "old_alignment_reused": False}))
    poses = {im.image_id: im.cam_from_world().matrix().copy() for im in model.images.values() if im.has_pose}
    cameras = {cid: cam.params.copy() for cid,cam in model.cameras.items()}
    before = [camera_geometry_metrics(model, im, policy["camera_geometry_admission"]) for im in model.images.values() if im.has_pose]
    if {r["image_name"] for r in before if not r["passed"]} != declared_weak:
        raise ValueError("Actual weak-camera support does not reproduce the frozen source receipt")
    centers = np.asarray([im.projection_center() for im in model.images.values() if im.has_pose])
    point_policy = policy["point_filter"]
    if np.linalg.norm(centers, axis=1).max() > point_policy["maximum_camera_radius_m"]:
        raise ValueError("The aligned camera extent violates the existing metric policy")
    lo, hi = centers.min(0)-point_policy["camera_aabb_margin_m"], centers.max(0)+point_policy["camera_aabb_margin_m"]
    removed = []
    for pid in list(model.points3D):
        point = model.points3D[pid]
        reasons = []
        if not np.isfinite(point.xyz).all():
            reasons.append("nonfinite_xyz")
        if not np.all(point.xyz >= lo) or not np.all(point.xyz <= hi):
            reasons.append("outside_metric_camera_extent")
        if not np.isfinite(point.error) or point.error > point_policy["maximum_point_reprojection_error_px"]:
            reasons.append("reprojection_error")
        if point.track.length() < point_policy["minimum_point_track_length"]:
            reasons.append("track_length")
        if reasons:
            removed.append({"point3D_id": int(pid), "reasons": reasons,
                "observations": [[int(el.image_id), int(el.point2D_idx)] for el in point.track.elements]})
            model.delete_point3D(pid)
    after = [camera_geometry_metrics(model, im, policy["camera_geometry_admission"]) for im in model.images.values() if im.has_pose]
    new_weak = {r["image_name"] for r in after if not r["passed"]}-declared_weak
    validity = projection_validity(model)
    checks = {"no_previously_supported_camera_became_weak": not new_weak,
        "all1021_cameras_retained": model.num_reg_images() == 1021,
        "fixed_camera_pose_during_point_filter": all(np.array_equal(im.cam_from_world().matrix(), poses[im.image_id]) for im in model.images.values() if im.has_pose),
        "fixed_intrinsics": all(np.array_equal(cam.params, cameras[cid]) for cid,cam in model.cameras.items()),
        "minimum_point_count": model.num_points3D() >= point_policy["minimum_retained_sparse_points"],
        "valid_observed_projections": validity["valid"]}
    destination = args.output/"diagnostic_model"
    destination.mkdir()
    model.write(destination)
    atomic_json(args.output/"REMOVED_POINTS.json", removed)
    atomic_json(args.output/"CAMERA_SUPPORT.json", {"before": before, "after": after})
    atomic_json(args.output/"PROJECTION_VALIDITY.json", validity)
    unchanged = source_identity == [data_identity(model_path/name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    checks["source_model_unchanged"] = unchanged
    checks["resource_budget"] = time.monotonic()-started <= 300 and psutil.Process().memory_info().rss <= 4*1024**3
    result = {"status": "diagnostic_preparation_passed_BA_pending" if all(checks.values()) else "failed_preparation_guard",
        "checks": checks, "new_weak_cameras": sorted(new_weak), "weak_cameras": [r for r in after if not r["passed"]],
        "retained_points": model.num_points3D(), "removed_points": len(removed), "retained_observations": model.compute_num_observations(),
        "model": str(destination), "model_role": "diagnostic_only_no_export_or_training", "counted": False,
        "elapsed_seconds": time.monotonic()-started, "final_camera_admission_passed": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
