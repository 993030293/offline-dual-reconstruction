"""Check strict native RGB sampling against every current raw-derived JPEG."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V3)]
from runtime import atomic_json, data_identity
from vector_camera_sampling import VectorCameraPixelRemap
from camera_pixel_remap import camera_fields


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    library = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001/b0/native_vector_camera_sampling.dll"
    admitted = read(args.side / "export/TRAINING_ADMISSION.json")
    model = pycolmap.Reconstruction(admitted["training_opencv_model"])
    if len(model.cameras) != 1:
        raise ValueError("The complete current sampler expects one calibrated sensor")
    camera = next(iter(model.cameras.values()))
    paths = {}
    for folder in ("initial/images", "temporal/images", "pose_gap/images", "initial/evaluation_images"):
        for path in sorted((args.side / folder).glob("*.jpg")):
            paths[path.name] = path
    expected = {row["image_name"] for row in read(args.side / "initial/INITIAL_FRAME_MANIFEST.json")["selected_frames"]}
    expected |= {row["image_name"] for row in read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]}
    if not expected.issubset(paths):
        raise ValueError("Some frozen training or evaluation raw-derived images are absent")
    path_ids = [data_identity(path) for path in paths.values()]
    options = pycolmap.UndistortCameraOptions()
    options.min_scale = options.max_scale = 1.
    source_paths = [Path(__file__), V3 / "vector_camera_sampling.py", V3 / "native_camera_sampling.py",
        V3 / "camera_pixel_remap.py", library]
    source_ids = [data_identity(path) for path in source_paths]
    atomic_json(args.output / "CONTRACT.json", {"sources": source_ids, "inputs": path_ids,
        "complete_image_count": len(paths), "camera": camera_fields(camera), "options": options.todict(),
        "literal_complete_pixel_and_camera_oracle": True, "inputs_unchanged": True,
        "read_decode_is_outside_both_sampling_API_times": True})
    rows = []
    try:
        for trial in range(3):
            operator = VectorCameraPixelRemap(pycolmap.undistort_image, library)
            times = {"native": 0., "SIMD": 0.}
            checks = []
            try:
                if not operator.vector_admitted:
                    raise ValueError("This host did not admit the existing strict SIMD library")
                for name, path in paths.items():
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB")).copy()
                    bitmap = pycolmap.Bitmap.from_array(array)
                    result = {}
                    for label in (("native", "SIMD") if trial != 1 else ("SIMD", "native")):
                        before = time.perf_counter()
                        result[label] = pycolmap.undistort_image(options, bitmap, camera) if label == "native" else operator(options, bitmap, camera)
                        times[label] += time.perf_counter()-before
                    native, candidate = result["native"], result["SIMD"]
                    check = {"image_name": name, "all_RGB_pixels_literal_exact": np.array_equal(native[0].to_array(), candidate[0].to_array()),
                        "all_camera_fields_exact": camera_fields(native[1]) == camera_fields(candidate[1])}
                    checks.append(check)
                    if not all(check[key] for key in ("all_RGB_pixels_literal_exact", "all_camera_fields_exact")):
                        raise ValueError("Strict native interpolation changed current complete pixels or calibration: "+name)
            finally:
                operator.close()
            rows.append({"trial": trial, "sampling_API_seconds": times, "all_complete_image_checks": checks,
                "operator_trace": operator.trace, "cold_calibration_map_build_in_candidate_API_time": True})
            atomic_json(args.output / "STATE.json", {"status": "running", "rows": rows})
        atomic_json(args.output / "RESULT.json", {"status": "complete_current_RGB_sampling_outcomes_recorded", "rows": rows,
            "input_count": len(paths), "elapsed_seconds": time.perf_counter()-started,
            "sources_unchanged": source_ids == [data_identity(path) for path in source_paths],
            "images_unchanged": path_ids == [data_identity(path) for path in paths.values()],
            "primary_images_and_localization_unchanged": True})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_current_RGB_sampling_audit",
            "rows": rows, "error": repr(error), "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
