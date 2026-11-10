"""Compare complete native pixels before any real-map performance campaign."""

import argparse
import io
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
from camera_pixel_remap_002 import CameraPixelRemap, camera_fields


def png(pixels):
    result = io.BytesIO()
    Image.fromarray(pixels).save(result, format="PNG")
    return result.getvalue()


def compare(operator, pixels, camera, options, label):
    bitmap = pycolmap.Bitmap.from_array(pixels)
    before = time.perf_counter()
    original, calibration = operator.original(options, bitmap, camera)
    original_seconds = time.perf_counter()-before
    before = time.perf_counter()
    current, returned = operator(options, bitmap, camera)
    current_seconds = time.perf_counter()-before
    a, b = original.to_array(), current.to_array()
    equal_shape = a.shape == b.shape
    differences = int(np.count_nonzero(a != b)) if equal_shape else None
    checks = {"camera": camera_fields(calibration) == camera_fields(returned),
        "shape_dtype": equal_shape and a.dtype == b.dtype,
        "all_pixels": differences == 0, "original_PNG_bytes": png(a) == png(b)}
    return {"label": label, "checks": checks, "differences": differences,
        "maximum_absolute_pixel_error": int(np.max(np.abs(a.astype(np.int16)-b.astype(np.int16)))) if equal_shape else None,
        "original_seconds": original_seconds, "candidate_seconds": current_seconds,
        "all_required": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original = pycolmap.undistort_image
    sources = [Path(__file__), Path(__file__).with_name("camera_pixel_remap_002.py")]
    identities = [data_identity(path) for path in sources]
    operator = CameraPixelRemap(original)
    rows, failure = [], None
    database = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/p23_track_completion_001/database.db"
    reference = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330/00_contract"
    try:
        with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as connection:
            cid, mid, width, height, params = connection.execute("SELECT camera_id,model,width,height,params FROM cameras").fetchone()
        actual_camera = pycolmap.Camera(camera_id=cid, model=pycolmap.CameraModelId(mid), width=width,
            height=height, params=np.frombuffer(params, np.float64).copy())
        split = json.loads((reference/"split_manifest.json").read_text(encoding="utf-8"))
        evaluation = [r for r in split["extra_frames"] if r["role"] == "H"]
        for row in evaluation[:3]:
            with Image.open(reference/"extra_images"/row["image_name"]) as image:
                pixels = np.asarray(image.convert("RGB"))
            result = compare(operator, pixels, actual_camera, pycolmap.UndistortCameraOptions(), row["image_name"])
            rows.append(result)
            if not result["all_required"]:
                failure = "actual_complete_pixels_or_camera_failed"
                break
        generator = np.random.default_rng(20261006)
        for model in ("PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"):
            if failure:
                break
            for width, height in ((73, 51), (121, 89)):
                for option_values in ({}, {"max_image_size": 48}, {"blank_pixels": .25},
                        {"roi_min_x": .1, "roi_max_y": .85}):
                    camera = pycolmap.Camera.create(1, model, 1.1*width, width, height)
                    camera.params[-1] = .02 if model != "PINHOLE" else camera.params[-1]
                    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
                    pixels[:, width//2] = 255
                    result = compare(operator, pixels, camera, pycolmap.UndistortCameraOptions(**option_values),
                        f"{model}_{width}_{height}_{option_values}")
                    rows.append(result)
                    if not result["all_required"]:
                        failure = "generic_complete_pixels_or_camera_failed"
                        break
                if failure:
                    break
    finally:
        operator.close()
    checks = {"all_actual_and_generic_pixels_cameras_PNG": failure is None and len(rows) == 43,
        "source_unchanged": identities == [data_identity(path) for path in sources],
        "owned_maps_closed": operator.closed and operator.maps is None,
        "original_global_callback_untouched": pycolmap.undistort_image is original}
    atomic_json(args.output/"RESULT.json", {"status": "pixel_preconditions_passed" if all(checks.values()) else "required_pixel_guard_failed",
        "checks": checks, "rows": rows, "failure": failure, "trace": operator.trace,
        "sources": identities, "whole_seconds": time.perf_counter()-started, "counted": False,
        "other_preconditions_and_full_cost": "not_run"})
    print(json.dumps({"status": "pixel_preconditions_passed" if all(checks.values()) else "required_pixel_guard_failed",
        "rows": len(rows), "failure": failure, "counted": False}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
