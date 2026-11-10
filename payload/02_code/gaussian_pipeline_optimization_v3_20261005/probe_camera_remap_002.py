"""Record native camera APIs and owned undistortion costs before implementation."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"))
from calibrated_reference_cache import camera_fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    database = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/p23_track_completion_001/database.db"
    reference = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330/00_contract"
    with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as connection:
        camera_id, model_id, width, height, params = connection.execute("SELECT camera_id,model,width,height,params FROM cameras").fetchone()
    camera = pycolmap.Camera(camera_id=camera_id, model=pycolmap.CameraModelId(model_id), width=width,
        height=height, params=np.frombuffer(params, np.float64).copy())
    split = json.loads((reference/"split_manifest.json").read_text(encoding="utf-8"))
    evaluation = [r for r in split["extra_frames"] if r["role"] == "H"]
    rows = []
    for row in evaluation[:3]:
        path = reference/"extra_images"/row["image_name"]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        bitmap = pycolmap.Bitmap.from_array(pixels)
        before = time.perf_counter()
        result, calibration = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(), bitmap, camera)
        elapsed = time.perf_counter()-before
        selected = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), camera)
        rays = selected.cam_from_img(np.asarray([[.5, .5], [selected.width-.5, selected.height-.5]], np.float64))
        source = camera.img_from_cam(np.column_stack([rays, np.ones(len(rays))]))
        rows.append({"name": path.name, "source": data_identity(path), "raw_shape": list(pixels.shape),
            "output_shape": list(result.to_array().shape), "undistort_seconds": elapsed,
            "calibration": camera_fields(calibration), "undistort_camera_exact": np.array_equal(selected.params, calibration.params)
                and selected.width == calibration.width and selected.height == calibration.height,
            "rays": rays.tolist(), "source_pixels": source.tolist()})
    atomic_json(args.output/"RESULT.json", {"status": "interface_probe_only", "pycolmap": pycolmap.__version__,
        "raw_camera": camera_fields(camera), "options": pycolmap.UndistortCameraOptions().todict(), "rows": rows,
        "Bitmap_rescale_available": hasattr(pycolmap.Bitmap, "rescale"),
        "camera_rescale_available": hasattr(pycolmap.Camera, "rescale"),
        "undistort_camera_doc": pycolmap.undistort_camera.__doc__, "whole_seconds": time.perf_counter()-started,
        "counted": False})
    print(json.dumps({"status": "interface_probe_only", "frames": len(rows), "counted": False}))


if __name__ == "__main__":
    main()
