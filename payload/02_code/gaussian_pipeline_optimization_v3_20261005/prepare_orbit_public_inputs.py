"""Read the complete existing public SfM model without changing its inputs."""

import hashlib
import json
import sys
import time
from pathlib import Path

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001/public_inputs001"


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError("A fresh complete public pose readout is required")
    OUT.mkdir(parents=True)
    paths = [MODEL / name for name in ["cameras.bin", "images.bin", "points3D.bin"]]
    before = [identity(path) for path in paths]
    model = pycolmap.Reconstruction(str(MODEL))
    denominator = [model.num_images(), model.num_reg_images(), model.num_points3D(), model.compute_num_observations()]
    if denominator != [360, 360, 133020, 848219]:
        raise AssertionError("The original complete public model denominator changed")
    poses = []
    for image in sorted(model.images.values(), key=lambda value: value.image_id):
        matrix = image.cam_from_world().inverse().matrix()
        poses.append({"image_id": image.image_id, "name": image.name,
                      "position": matrix[:, 3].tolist(), "rotation_c2w": matrix[:, :3].tolist()})
    if len(poses) != 360 or [identity(path) for path in paths] != before:
        raise AssertionError("The full original public camera population or input identity changed")
    payload = {"status": "complete_public360_pose_component_input_read0", "denominator": denominator,
               "poses": poses, "inputs": before, "source": identity(Path(__file__)),
               "new_optimization_count": 0, "elapsed_seconds": time.perf_counter() - started}
    with (OUT / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({"denominator": denominator, "poses": len(poses), "elapsed_seconds": payload["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
