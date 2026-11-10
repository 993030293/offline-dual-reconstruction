"""Inspect every similarity projection against existing point admission rules."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.metric / "CONTRACT.json").read_text())
    alignment = json.loads((args.metric / "ALIGNMENT.json").read_text())
    policy_path = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())["point_filter"]
    original = pycolmap.Reconstruction(contract["source_model"])
    original.update_point_3d_errors()
    changed = pycolmap.Reconstruction(original)
    values = alignment["new_Sim3"]["params"]
    changed.transform(pycolmap.Sim3d(values[7], pycolmap.Rotation3d(np.asarray(values[:4])), np.asarray(values[4:7])))
    maxima = {"all": 0., "existing_track_gate": 0., "existing_track_and_error_gate": 0.}
    worst = []
    for identity, image in original.images.items():
        if not image.has_pose:
            continue
        entries = [(index, point) for index, point in enumerate(image.points2D) if point.has_point3D()]
        xyz = np.asarray([original.points3D[point.point3D_id].xyz for _, point in entries]).reshape(-1, 3)
        mapped = np.asarray([changed.points3D[point.point3D_id].xyz for _, point in entries]).reshape(-1, 3)
        pose, updated_pose = image.cam_from_world(), changed.images[identity].cam_from_world()
        coordinates = xyz @ pose.rotation.matrix().T + pose.translation
        new_coordinates = mapped @ updated_pose.rotation.matrix().T + updated_pose.translation
        camera = original.cameras[image.camera_id]
        old_projection = camera.img_from_cam(coordinates)
        new_projection = camera.img_from_cam(new_coordinates)
        deltas = np.max(np.abs(new_projection-old_projection), axis=1) if len(entries) else np.empty(0)
        for slot, delta in enumerate(deltas):
            index, point2d = entries[slot]
            point = original.points3D[point2d.point3D_id]
            track = point.track.length() >= policy["minimum_point_track_length"]
            error = np.isfinite(point.error) and point.error <= policy["maximum_point_reprojection_error_px"]
            maxima["all"] = max(maxima["all"], float(delta))
            if track:
                maxima["existing_track_gate"] = max(maxima["existing_track_gate"], float(delta))
            if track and error:
                maxima["existing_track_and_error_gate"] = max(maxima["existing_track_and_error_gate"], float(delta))
            if delta > 1e-6:
                worst.append({"image_name": image.name, "point2D_index": index,
                    "point3D_id": int(point2d.point3D_id), "projection_delta_px": float(delta),
                    "original_depth_visual_units": float(coordinates[slot, 2]),
                    "transformed_depth_m": float(new_coordinates[slot, 2]),
                    "track_length": point.track.length(), "native_updated_error_px": float(point.error),
                    "existing_track_gate_passed": bool(track), "existing_error_gate_passed": bool(error),
                    "original_projection": old_projection[slot].tolist(), "observed_xy": point2d.xy.tolist()})
    worst.sort(key=lambda row: -row["projection_delta_px"])
    atomic_json(args.output / "RESULT.json", {"source": data_identity(Path(__file__)),
        "model_source": contract["source_model_identity"], "policy_source": data_identity(policy_path),
        "maximum_absolute_delta_px": maxima, "all_exceeding_observations": worst,
        "all_observations_inspected": original.compute_num_observations(),
        "no_model_or_admission_modified": True, "thresholds_changed": False})
    print(json.dumps({"maximum_absolute_delta_px": maxima, "exceeding_observations": len(worst),
        "first": worst[:3]}), flush=True)


if __name__ == "__main__":
    main()
