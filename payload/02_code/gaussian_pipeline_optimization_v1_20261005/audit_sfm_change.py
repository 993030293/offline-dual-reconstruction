"""Compare common image observations without confusing point merges with deletion."""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity


def observation_errors(model):
    errors, point_ids = {}, {}
    for image in model.images.values():
        if not image.has_pose:
            continue
        selected = [(index, point) for index, point in enumerate(image.points2D) if point.has_point3D()]
        if not selected:
            continue
        xyz = np.asarray([model.points3D[point.point3D_id].xyz for _, point in selected])
        transform = image.cam_from_world()
        camera_xyz = xyz @ transform.rotation.matrix().T + transform.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        observed = np.asarray([point.xy for _, point in selected])
        residuals = np.linalg.norm(projected-observed, axis=1)
        if not np.isfinite(residuals).all() or np.any(camera_xyz[:, 2] <= 0):
            raise ValueError("Non-finite or behind-camera observed 3D point")
        for (index, point), residual in zip(selected, residuals):
            key = (image.name, index)
            errors[key] = float(residual)
            point_ids[key] = int(point.point3D_id)
    return errors, point_ids


def summary(values):
    array = np.asarray(values)
    if len(array) == 0:
        return {"count": 0}
    return {"count": len(array), "mean": float(array.mean()),
            "p50": float(np.quantile(array, .5)), "p95": float(np.quantile(array, .95)), "max": float(array.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before, after = pycolmap.Reconstruction(args.before), pycolmap.Reconstruction(args.after)
    old, old_ids = observation_errors(before)
    new, new_ids = observation_errors(after)
    common = sorted(old.keys() & new.keys())
    lost = sorted(old.keys()-new.keys())
    added = sorted(new.keys()-old.keys())
    old_common, new_common = summary([old[key] for key in common]), summary([new[key] for key in common])
    old_images = {image.name: image for image in before.images.values() if image.has_pose}
    new_images = {image.name: image for image in after.images.values() if image.has_pose}
    pose_changes = {name: float(np.max(np.abs(image.cam_from_world().matrix()-new_images[name].cam_from_world().matrix())))
                    for name, image in old_images.items() if name in new_images}
    calibration_equal = all(camera_id in after.cameras and np.array_equal(camera.params, after.cameras[camera_id].params)
                            for camera_id, camera in before.cameras.items())
    delta = new_common["p95"]-old_common["p95"]
    checks = {"no_old_registered_image_lost": old_images.keys() <= new_images.keys(),
        "fixed_existing_pose_max_abs_below_1e_10": max(pose_changes.values(), default=0) <= 1e-10,
        "calibration_exact": calibration_equal, "common_observation_p95_increase_at_most_0_05_px": delta <= .05}
    result = {"status": "common_observations_audited", "before": str(args.before), "after": str(args.after),
        "old_observation_count": len(old), "new_observation_count": len(new), "common_before_px": old_common,
        "common_after_px": new_common, "p95_delta_px": delta, "lost_count": len(lost),
        "lost_original_residuals_px": summary([old[key] for key in lost]), "new_count": len(added),
        "new_observation_residuals_px": summary([new[key] for key in added]),
        "common_point_id_changed_count": sum(old_ids[key] != new_ids[key] for key in common),
        "maximum_existing_pose_element_change": max(pose_changes.values(), default=0),
        "checks": checks, "guard_passed": all(checks.values()),
        "no_geometry_accuracy_claim": True, "source": data_identity(Path(__file__))}
    atomic_json(args.output/"RESULT.json", result)
    atomic_json(args.output/"LOST_OBSERVATIONS.json", [{"image": key[0], "index": key[1], "old_px": old[key]} for key in lost])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
