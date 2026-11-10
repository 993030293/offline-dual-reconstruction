"""Bind optical-camera priors to actual model and database identities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap

from common import BASE, PoseCurve, configure_stdout, make_priors, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "01_priors"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("The prior output directory already contains artifacts")
    output.mkdir(exist_ok=True)
    curve = PoseCurve(BASE / "03_metric_context/metric_odometry.tum",
                      BASE / "03_metric_context/T_base_link_camera_optical.json")
    model = pycolmap.Reconstruction(run / "00_contract/initial_model")
    manifest = read_json(run / "00_contract/BASELINE_MANIFEST.json")["source_images"]
    source = read_json(BASE / "04_trajectory_binding/interpolated_camera_poses.json")
    existing = {item["image_name"]: item for item in source["images"]}
    images = {image.name: image for image in model.images.values()}
    groups, rows = {}, []
    for item in manifest:
        pose = curve.at(int(item["camera_header_ns"]))
        image = images[item["image_name"]]
        row = {**pose, "frame_uid": item["image_name"], "image_name": item["image_name"],
            "camera_frame": item["camera_frame"], "image_id": image.image_id,
            "camera_id": image.camera_id, "frame_id": image.frame_id,
            "corr_data_id": {"sensor_type": "CAMERA", "sensor_id": image.camera_id, "data_id": image.image_id},
            "quality_sharpness": item["sharpness"], "quality_clipping": item["clipped_fraction"],
            "position_covariance_m2": (np.eye(3) * .15**2).tolist(),
            "uncertainty_source": "predeclared_development_weight_raw_covariance_not_calibrated",
            "source_role": "recorded_current_mcap_odometry", "producer": "unconfirmed",
            "prior_active": False, "quality_flags": ["relative_quality_selection_not_absolute_certificate"]}
        key = (row["left_pose_timestamp_ns"], row["right_pose_timestamp_ns"])
        row["correlation_group"] = f"{key[0]}_{key[1]}"
        groups.setdefault(key, []).append(row)
        rows.append(row)
    for group in groups.values():
        selected = min(group, key=lambda r: (-r["quality_sharpness"], r["quality_clipping"], r["image_name"]))
        selected["prior_active"] = True
    deltas = [np.linalg.norm(np.asarray(row["C_W_m"]) -
        np.asarray(existing[row["image_name"]]["T_world_camera_interpolated"]["pose_xyz_xyzw"][:3])) for row in rows]
    if max(deltas) > 1e-8:
        raise ValueError("Recomputed camera centers disagree with the preserved binding")
    priors = make_priors(model, rows)
    database = pycolmap.Database.open(run / "00_contract/database.db")
    try:
        if database.num_pose_priors() != 0:
            raise ValueError("Experiment copy unexpectedly contains existing pose priors")
        for prior in priors:
            database.write_pose_prior(prior)
        restored = database.read_all_pose_priors()
        indexed = {int(p.corr_data_id.id): p for p in restored}
        equal = len(restored) == len(priors) and all(
            np.array_equal(p.position, indexed[int(p.corr_data_id.id)].position) and
            np.array_equal(p.position_covariance, indexed[int(p.corr_data_id.id)].position_covariance)
            and indexed[int(p.corr_data_id.id)].coordinate_system == pycolmap.PosePriorCoordinateSystem.CARTESIAN
            for p in priors)
        if not equal:
            raise ValueError("Pose-prior database readback mismatch")
    finally:
        database.close()
    write_json(output / "camera_prior_manifest.json", {"status": "validated_and_bound", "images": rows,
        "frame_count": len(rows), "active_prior_count": len(priors), "coordinate_system": "CARTESIAN",
        "maximum_recomputed_vs_existing_center_delta_m": float(max(deltas)),
        "claim_boundary": "Camera priors are estimates; no independent GT."})
    write_json(output / "PRIOR_WRITE_READBACK.json", {"status": "passed", "written": len(priors),
        "readback_equal": equal, "database_is_experiment_copy": True})
    print(f"P1 generated {len(rows)} camera poses and {len(priors)} active position priors")


if __name__ == "__main__":
    configure_stdout()
    main()
