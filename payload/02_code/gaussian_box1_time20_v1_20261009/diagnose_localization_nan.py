"""Reproduce one preserved query and separate pose validity from unused diagnostics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import cv2
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_full_raw_ab_20261008"))
import localize_evaluation as entry
from atomic_json_windows import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=7894)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    admission = entry.read(args.side / "export/TRAINING_ADMISSION.json")
    model = pycolmap.Reconstruction(Path(admission["training_opencv_model"]))
    camera = next(iter(model.cameras.values()))
    source = next(row for row in entry.read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]
                  if row["camera_frame"] == args.frame)
    cache = args.side / "localization/feature_cache" / (Path(source["image_name"]).stem+".npz")
    with np.load(cache) as data:
        keypoints, descriptors = data["keypoints"].copy(), data["descriptors"].copy()
    names = {image.name for image in model.images.values() if image.has_pose}
    store = entry.SiftFeatureStore(args.side / "pose_gap/database.db", names, "lazy", 32)
    bank = entry.FeatureBank.__new__(entry.FeatureBank)
    bank.original = store
    cv2.setNumThreads(4)
    native = pycolmap.estimate_and_refine_absolute_pose

    def estimate(xy, xyz, calibration, estimation, refinement):
        estimation.ransac.num_threads = 1
        return native(xy, xyz, calibration, estimation, refinement)

    pycolmap.estimate_and_refine_absolute_pose = estimate
    try:
        neighbors = entry.metadata_neighbors(store, model, source, source["camera_header_ns"])
        matches, _, _ = entry.split_correspondences(bank, model, keypoints, descriptors, neighbors)
        solution, info, used, unused = entry.estimate_pose(model, keypoints, matches, split=True)
        nonfinite = []

        def visit(value, prefix):
            if isinstance(value, dict):
                for key, item in value.items(): visit(item, prefix+"."+key)
            elif isinstance(value, list):
                for number, item in enumerate(value): visit(item, prefix+"."+str(number))
            elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                nonfinite.append(prefix)

        visit(info, "info")
        pose = solution["cam_from_world"] if solution else None
        xyz = np.asarray([model.points3D[row["point_id"]].xyz for row in unused])
        cam_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        pixels = camera.img_from_cam(cam_xyz)
        np.savez(args.output / "UNUSED_PROJECTION_VALUES.npz", camera_xyz=cam_xyz, projected_pixels=pixels)
        result = {"frame": args.frame, "image_name": source["image_name"], "status": info["status"],
            "inliers": info.get("inliers"), "nonfinite_fields": nonfinite,
            "finite_pose_matrix": bool(np.isfinite(pose.matrix()).all()),
            "finite_localization_inlier_error_quantiles": all(np.isfinite(value)
                for value in info["loc_inlier_reprojection_px"].values()),
            "unused_count": len(unused), "unused_nonprojectable_count": int((~np.isfinite(pixels).all(axis=1)).sum()),
            "unused_nonpositive_depth_count": int((cam_xyz[:,2] <= 0).sum()),
            "read_existing_query_features_only": True, "PnP_options_and_correspondences_unchanged": True}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps(result), flush=True)
    finally:
        pycolmap.estimate_and_refine_absolute_pose = native
        store.close()


if __name__ == "__main__":
    main()
