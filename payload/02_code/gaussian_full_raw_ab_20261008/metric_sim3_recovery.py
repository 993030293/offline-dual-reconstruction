"""Recover from a failed BA using only a verified similarity of the valid source map."""

import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from run_metric_refinement import projection_validity, observation_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-metric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefilter-original-track-policy", action="store_true")
    parser.add_argument("--prefilter-renderer-near-plane", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    failure = json.loads((args.failed_metric / "RESULT.json").read_text())
    contract = json.loads((args.failed_metric / "CONTRACT.json").read_text())
    alignment = json.loads((args.failed_metric / "ALIGNMENT.json").read_text())
    if failure["status"] != "failed_projection_validation" or failure["selected"] is not None:
        raise ValueError("This explicit branch only handles the preserved BA projection failure")
    source = Path(contract["source_model"])
    identities = [data_identity(source / name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if identities != contract["source_model_identity"] or alignment["old_alignment_reused"]:
        raise ValueError("The source or newly estimated alignment identity changed")
    model = pycolmap.Reconstruction(source)
    before = projection_validity(model)
    if not before["valid"]:
        raise ValueError("The original visual map is invalid; similarity cannot repair bad observations")
    original_observation_ids = observation_identity(model)
    renderer_removed = []
    early_removed = []
    if args.prefilter_original_track_policy:
        package = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
        policy_path = package / "config/METRIC_ALIGNMENT_POLICY.json"
        minimum = json.loads(policy_path.read_text())["point_filter"]["minimum_point_track_length"]
        for identity in list(model.points3D):
            point = model.points3D[identity]
            if point.track.length() < minimum:
                early_removed.append({"point3D_id": int(identity), "track_length": point.track.length(),
                    "reason": "existing_training_minimum_point_track_length",
                    "observations": [[int(item.image_id), int(item.point2D_idx)] for item in point.track.elements]})
                model.delete_point3D(identity)
        atomic_json(args.output / "EARLY_EXISTING_TRACK_POLICY.json", {"policy": data_identity(policy_path),
            "unchanged_minimum_track_length": minimum, "removed": early_removed,
            "full_original_source_map_preserved": True, "all_registered_cameras_retained": True,
            "original_observations": len(original_observation_ids), "retained_observations": model.compute_num_observations(),
            "protocol_change": "apply the existing training point support gate before similarity; not an original BA pass"})
        private_source = args.output / "prefiltered_visual_training_candidate"
        private_source.mkdir()
        model.write(private_source)
    if args.prefilter_renderer_near_plane:
        if not args.prefilter_original_track_policy:
            raise ValueError("The renderer interface filter requires the original training support policy")
        camera_source = ROOT / "04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/scene/cameras.py"
        tree = ast.parse(camera_source.read_text())
        camera_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Camera"]
        if len(camera_classes) != 1:
            raise ValueError("The native renderer Camera class changed")
        declared = [node.value.value for node in ast.walk(camera_classes[0]) if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant) and any(isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr == "znear" for target in node.targets)]
        if len(declared) != 1 or type(declared[0]) not in (int, float) or declared[0] <= 0:
            raise ValueError("The native renderer near plane is not one positive literal")
        near = float(declared[0])
        scale = float(alignment["new_Sim3"]["params"][7])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("The current new metric scale is invalid")
        outside = {}
        for image in model.images.values():
            if not image.has_pose:
                continue
            points = [(index, point) for index, point in enumerate(image.points2D) if point.has_point3D()]
            xyz = np.asarray([model.points3D[point.point3D_id].xyz for _, point in points]).reshape(-1, 3)
            pose = image.cam_from_world()
            depths = (xyz @ pose.rotation.matrix().T + pose.translation)[:, 2] * scale
            if not np.isfinite(depths).all() or np.any(depths <= 0):
                raise ValueError("The original supported visual map has invalid depths")
            for slot in np.flatnonzero(depths <= near):
                index, point = points[slot]
                outside.setdefault(int(point.point3D_id), []).append({"image_id": int(image.image_id),
                    "image_name": image.name, "point2D_index": index, "source_depth_scaled_to_m": float(depths[slot])})
        before_points, before_observations = model.num_points3D(), model.compute_num_observations()
        for identity in sorted(outside):
            point = model.points3D[identity]
            renderer_removed.append({"point3D_id": identity, "track_length": point.track.length(),
                "reason": "at_least_one_recorded_track_observation_inside_native_Camera_near_plane",
                "outside_renderer_observations": outside[identity],
                "all_removed_observations": [[int(item.image_id), int(item.point2D_idx)] for item in point.track.elements]})
            model.delete_point3D(identity)
        point_policy = json.loads((ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json").read_text())["point_filter"]
        if model.num_points3D() < point_policy["minimum_retained_sparse_points"]:
            raise ValueError("Renderer support filtering left fewer than the unchanged minimum sparse points")
        atomic_json(args.output / "RENDERER_NEAR_PLANE_POINT_POLICY.json", {"native_renderer_source": data_identity(camera_source),
            "near_m": near, "parameter_source": "native Camera.__init__.self.znear; no tuned constant",
            "scale": scale, "before_points": before_points, "retained_points": model.num_points3D(),
            "before_observations": before_observations, "retained_observations": model.compute_num_observations(),
            "removed": renderer_removed, "all_original_registered_cameras_retained": True,
            "full_original_visual_map_preserved": True, "quality_and_projection_tolerance_changed": False,
            "new_engineering_interface_repair_not_original75_pass": True})
    observation_ids = observation_identity(model)
    calibrated = {identity: camera.params.copy() for identity, camera in model.cameras.items()}
    old_centers = {identity: image.projection_center().copy() for identity, image in model.images.items() if image.has_pose}
    projected = {}
    for identity, image in model.images.items():
        if image.has_pose:
            points = [point for point in image.points2D if point.has_point3D()]
            xyz = np.asarray([model.points3D[point.point3D_id].xyz for point in points]).reshape(-1, 3)
            pose = image.cam_from_world()
            projected[identity] = model.cameras[image.camera_id].img_from_cam(xyz @ pose.rotation.matrix().T + pose.translation)
    values = alignment["new_Sim3"]["params"]
    transform = pycolmap.Sim3d(values[7], pycolmap.Rotation3d(np.asarray(values[:4])), np.asarray(values[4:7]))
    atomic_json(args.output / "CONTRACT.json", {"status": "frozen_before_pure_sim3_recovery",
        "failed_metric": data_identity(args.failed_metric / "RESULT.json"), "source_map": str(source),
        "source_identity": identities, "source": data_identity(Path(__file__)), "new_alignment": alignment,
        "BA_failure_preserved": True, "failed_BA_models_used": False, "SfM_or_GT_for_initial_selection": False,
        "active_position_prior_BA": False, "no_point_observation_or_camera_removed": not bool(early_removed or renderer_removed),
        "full_original_source_map_preserved": True, "all_registered_cameras_retained": True,
        "existing_track_policy_applied_before_similarity": args.prefilter_original_track_policy,
        "native_renderer_near_plane_interface_applied": args.prefilter_renderer_near_plane,
        "original_observation_count": len(original_observation_ids), "retained_observation_count": len(observation_ids),
        "projection_max_absolute_delta_px": 1e-6, "scale_truth": "coarse motion only, not GT"})
    model.transform(transform)
    after = projection_validity(model)
    if not after["valid"] or observation_ids != observation_identity(model):
        raise ValueError("Similarity changed observation membership or produced invalid projections")
    if any(not np.array_equal(camera.params, calibrated[identity]) for identity, camera in model.cameras.items()):
        raise ValueError("Similarity modified frozen calibration")
    maximum = 0.
    center_maximum = 0.
    for identity, image in model.images.items():
        if not image.has_pose:
            continue
        center_expected = values[7] * transform.rotation.matrix() @ old_centers[identity] + transform.translation
        center_maximum = max(center_maximum, float(np.max(np.abs(image.projection_center()-center_expected))))
        points = [point for point in image.points2D if point.has_point3D()]
        xyz = np.asarray([model.points3D[point.point3D_id].xyz for point in points]).reshape(-1, 3)
        pose = image.cam_from_world()
        actual = model.cameras[image.camera_id].img_from_cam(xyz @ pose.rotation.matrix().T + pose.translation)
        if len(actual):
            maximum = max(maximum, float(np.max(np.abs(actual-projected[identity]))))
    atomic_json(args.output / "ORACLE.json", {"maximum_projection_delta_px": maximum,
        "maximum_metric_center_delta_m": center_maximum, "retained_observations": len(observation_ids),
        "original_observations": len(original_observation_ids), "earlier_point_gate_removals": len(early_removed),
        "renderer_interface_point_removals": len(renderer_removed)})
    if maximum > 1e-6 or center_maximum > 1e-8:
        raise ValueError("Similarity failed the complete projection or metric-center oracle")
    target = args.output / "pure_sim3_alignment/model"
    target.mkdir(parents=True)
    model.write(target)
    result = {"status": "pure_sim3_metric_interface_verified_BA_not_accepted", "selected": "pure_sim3_alignment",
        "results": {"pure_sim3_alignment": {"model": str(target), "active_priors": 0,
            "registered_images": model.num_reg_images(), "projection": after,
            "common_observations_preserved": True, "reprojection_max_absolute_delta_px": maximum,
            "metric_center_maximum_delta_m": center_maximum}}, "BA_failure_preserved": True,
        "original_BA_guard_passed": False, "independent_accuracy_claim": False,
        "original_observations_preserved": not bool(early_removed or renderer_removed),
        "original_source_map_preserved": True, "early_existing_track_policy": bool(early_removed),
        "native_renderer_interface_repair": bool(args.prefilter_renderer_near_plane),
        "retained_observations_preserved_through_similarity": True,
        "common_observation_p95_delta_px": None, "development_guard_passed": False,
        "elapsed_seconds": time.monotonic()-started}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "selected", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
