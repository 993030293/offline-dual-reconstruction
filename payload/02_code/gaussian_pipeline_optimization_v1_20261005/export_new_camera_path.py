"""Export the admitted map's calibrated trajectory in original header order."""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = pycolmap.Reconstruction(args.dataset / "sparse/0")
    raw = {}
    with args.raw_decisions.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["image_name"] in raw:
                raise ValueError("Duplicate raw image identity")
            raw[item["image_name"]] = item
    images = [image for image in model.images.values() if image.has_pose]
    names = {path.name for path in (args.dataset / "images").iterdir() if path.suffix.lower() in (".jpg", ".png", ".jpeg")}
    if len(images) != 1016 or {image.name for image in images} != names:
        raise ValueError("The camera path does not match the admitted 1016-image training input")
    images.sort(key=lambda image: (raw[image.name]["camera_header_ns"], image.name))
    views, projection_errors = [], []
    for image in images:
        source = raw[image.name]
        if source["excluded_role"] is not None:
            raise ValueError("An evaluation image entered the training path")
        camera = model.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise ValueError("The display trajectory requires the exported full PINHOLE model")
        fx, fy, cx, cy = map(float, camera.params)
        transform = image.cam_from_world()
        rotation = transform.rotation.matrix().T
        center = image.projection_center()
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10):
            raise ValueError("Invalid camera rotation")
        samples = []
        for point2d in image.points2D:
            if point2d.has_point3D():
                samples.append(model.points3D[point2d.point3D_id].xyz)
            if len(samples) >= 8:
                break
        if not samples:
            raise ValueError("An admitted camera has no geometry for its projection oracle")
        xyz = np.asarray(samples)
        local = (xyz - center) @ rotation
        direct = np.column_stack((fx * local[:, 0] / local[:, 2] + cx, fy * local[:, 1] / local[:, 2] + cy))
        oracle = camera.img_from_cam(transform * xyz)
        error = float(np.max(np.abs(direct - oracle)))
        if not np.isfinite(direct).all() or (local[:, 2] <= 0).any() or error > 1e-8:
            raise ValueError("World-to-camera convention or calibrated projection failed")
        projection_errors.append(error)
        views.append({"image_id": image.image_id, "image_name": image.name,
            "camera_frame": source["camera_frame"], "camera_header_ns": source["camera_header_ns"],
            "position": center.tolist(), "forward": rotation[:, 2].tolist(), "up": (-rotation[:, 1]).tolist(),
            "rotation_c2w": rotation.tolist(), "width": camera.width, "height": camera.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy})
    if not all(a["camera_header_ns"] < b["camera_header_ns"] for a, b in zip(views, views[1:])):
        raise ValueError("Raw header order is not strictly increasing")
    args.output.mkdir(parents=True)
    atomic_json(args.output / "camera_path.json", views)
    atomic_json(args.output / "RECEIPT.json", {"status": "all_admitted_cameras_verified", "camera_count": len(views),
        "input_version": "new_map_admission_v1", "world_units": "meters_from_recorded_odom_binding",
        "world_up": "positive_Z", "pixel_convention": "colmap_corner_first_center_0.5",
        "default_view": len(views) // 2, "max_native_projection_delta_px": max(projection_errors),
        "raw_decisions": data_identity(args.raw_decisions), "source": data_identity(Path(__file__)),
        "model_files": [data_identity(args.dataset / "sparse/0" / name) for name in ("cameras.bin", "images.bin", "points3D.bin")],
        "old_map_camera_poses_used": False, "independent_geometry_accuracy_claim": False})
    print(json.dumps({"camera_count": len(views), "max_native_projection_delta_px": max(projection_errors)}))


if __name__ == "__main__":
    main()
