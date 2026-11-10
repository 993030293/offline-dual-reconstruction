"""Preserve a common metric world while rectifying private branch camera inputs."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pycolmap
from ply_io import write_vertices

from common import COLMAP, atomic_json, read_json


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def write_initial_cloud(model, path):
    points = list(model.points3D.values())
    data = np.zeros(len(points), dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
                    + [(name, "u1") for name in ("red", "green", "blue")])
    xyz = np.asarray([point.xyz for point in points])
    colors = np.asarray([point.color for point in points])
    if not len(points) or not np.isfinite(xyz).all():
        raise ValueError("The common initial cloud is empty or nonfinite")
    for index, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        data[name] = colors[:, index]
    write_vertices(path, data)


def admitted_names(admission):
    if admission['status'] != 'export_and_admission_verified_training_pending' or admission['initial_not_registered']:
        raise ValueError('Shared input requires a verified export with every original camera registered')
    rows = admission['rows']
    mapped = [row['image_name'] for row in rows]
    included = {row['image_name'] for row in rows if row['training_role'] == 'admitted'}
    excluded = {row['image_name'] for row in rows if row['training_role'] == 'excluded_insufficient_support'}
    if (len(mapped) != len(set(mapped)) or len(mapped) != admission['full_registered_denominator']
            or len(included) != admission['training_count'] or not included
            or included & excluded or included | excluded != set(mapped)
            or excluded != set(admission['excluded_names'])):
        raise ValueError('The map and admitted-training camera partitions are inconsistent')
    return included


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = read_json(root / "CONFIG.json")
    side = root / config["case"] / "optimized"
    admission = read_json(side / "export/TRAINING_ADMISSION.json")
    admitted = admitted_names(admission)
    source_root = Path(admission["dataset"])
    source = pycolmap.Reconstruction(source_root / "sparse/0")
    if {image.name for image in source.images.values() if image.has_pose} != admitted:
        raise ValueError('The persisted training map differs from the verified admission partition')
    folder = root / "shared"
    folder.mkdir(exist_ok=False)
    target = folder / "rectified"
    command = [str(COLMAP), "image_undistorter", "--image_path", str(source_root / "images"),
        "--input_path", str(source_root / "sparse/0"), "--output_path", str(target),
        "--output_type", "COLMAP", "--copy_policy", "copy", "--max_image_size", "812", "--num_threads", "8"]
    atomic_json(folder / "COMMAND.json", {"command": command})
    with (folder / "undistortion.log").open("x", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    model = pycolmap.Reconstruction(target / "sparse")
    names = {image.name for image in model.images.values() if image.has_pose}
    if names != admitted or len(names) != admission["training_count"]:
        raise ValueError("Rectification changed the camera denominator")
    pose_error = max(float(np.max(np.abs(image.cam_from_world().matrix() - model.images[key].cam_from_world().matrix())))
                     for key, image in source.images.items())
    if set(source.points3D) != set(model.points3D):
        raise ValueError("Rectification changed the initial point denominator")
    point_error = max(float(np.max(np.abs(point.xyz - model.points3D[key].xyz)))
                      for key, point in source.points3D.items())
    if max(pose_error, point_error) > 1e-9:
        raise ValueError("Rectification moved the common world")
    cameras = []
    for camera in model.cameras.values():
        if camera.model_name != "PINHOLE" or len(camera.params) != 4:
            raise ValueError("Both branches require actual calibrated PINHOLE images")
        cameras.append({"id": camera.camera_id, "width": camera.width, "height": camera.height,
                        "model": camera.model_name, "corner_K": camera.params.tolist()})
    binary_identities = {name: file_identity(target / "sparse" / name)
                         for name in ("cameras.bin", "images.bin", "points3D.bin")}
    for branch, layout in (("gaussian", "sparse/0"), ("pgsr", "sparse")):
        dataset = folder / branch
        sparse = dataset / layout
        sparse.mkdir(parents=True)
        for name in binary_identities:
            shutil.copyfile(target / "sparse" / name, sparse / name)
        write_initial_cloud(model, sparse / "points3D.ply")
        images = dataset / "images"
        images.mkdir()
        for name in sorted(names):
            os.link(target / "images" / name, images / name)
        if any(file_identity(sparse / name) != value for name, value in binary_identities.items()):
            raise ValueError("A private sparse input changed bytes")
    result = {"status": "shared_calibrated_inputs_verified", "registered_frames": len(names),
        "registered_names": sorted(names), "points": model.num_points3D(), "cameras": cameras,
        "maximum_pose_component_error": pose_error, "maximum_point_component_error": point_error,
        "world_units": "coarse_sensor_motion_aligned_meters_independent_accuracy_unverified",
        "private_sparse_binaries": binary_identities, "image_sharing": "read_only_consumers_NTFS_hardlinks",
        "source_admission": str(side / "export/TRAINING_ADMISSION.json"), "source_dataset": str(source_root),
        "source_full_registered_frames": admission['full_registered_denominator'],
        "excluded_from_training_by_existing_geometry_policy": admission['excluded_names'],
        "every_admitted_frame_shared_by_both_branches": True,
        "GT_used": False, "old_models_used": False}
    atomic_json(folder / "RESULT.json", result)
    print(result["status"], len(names), flush=True)


if __name__ == "__main__":
    main()
