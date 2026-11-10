"""Stage native pixel oracles, run one owned CUDA child and compare complete outputs."""

import time
STARTED = time.perf_counter()

import argparse
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from camera_pixel_remap import CameraPixelRemap, camera_fields


def png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CODE / "camera_pixel_remap.py", CODE / "gpu_camera_remap_pixels.py",
               CODE / "camera_remap_gpu_worker.py", CODE / "mapped_image_linux_guard.py"]
    source_ids = [data_identity(path) for path in sources]
    saved = args.output / "source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(sources):
        shutil.copyfile(path, saved / (f"{ordinal:03d}_" + path.name))
    original = pycolmap.undistort_image
    operator = CameraPixelRemap(original)
    database = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p23_track_completion_001/database.db"
    reference = ROOT / "04_outputs/box2_lio_guided_sfm_v1_20261002T014330/00_contract"
    inputs = [database, reference / "split_manifest.json"]
    cases, groups, staging, native_seconds = [], {}, 0.0, 0.0

    def stage(pixels, camera, options, label, source_path=None):
        nonlocal staging, native_seconds
        started = time.perf_counter()
        bitmap = pycolmap.Bitmap.from_array(pixels)
        begin = time.perf_counter()
        native, native_calibration = original(options, bitmap, camera)
        elapsed = time.perf_counter() - begin
        native_seconds += elapsed
        calibration = pycolmap.undistort_camera(options, camera)
        if camera_fields(calibration) != camera_fields(native_calibration):
            raise ValueError("The independently constructed target camera differs from the native image result")
        signature = json.dumps({"camera": camera_fields(camera), "options": options.todict()}, sort_keys=True, allow_nan=False)
        if signature not in groups:
            group = f"map_{len(groups):03d}"
            operator.build(camera, calibration)
            directory = args.output / group
            directory.mkdir()
            paths = []
            for name, value in zip(("x0", "y0", "dx", "dy", "valid"), operator.maps):
                np.save(directory / (name + ".npy"), value, allow_pickle=False)
                paths.append(str((directory / (name + ".npy")).relative_to(args.output)))
            groups[signature] = {"group": group, "paths": paths, "map_bytes": sum(value.nbytes for value in operator.maps)}
        case = f"case_{len(cases):03d}"
        directory = args.output / case
        directory.mkdir()
        np.save(directory / "raw.npy", pixels, allow_pickle=False)
        np.save(directory / "native.npy", native.to_array(), allow_pickle=False)
        row = {"case": case, "label": label, "group": groups[signature]["group"],
               "camera": camera_fields(camera), "calibration": camera_fields(calibration),
               "native_calibration": camera_fields(native_calibration),
               "original_seconds": elapsed, "source": None if source_path is None else data_identity(source_path)}
        cases.append(row)
        staging += time.perf_counter() - started

    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            cid, mid, width, height, params = connection.execute("SELECT camera_id,model,width,height,params FROM cameras").fetchone()
        camera = pycolmap.Camera(camera_id=cid, model=pycolmap.CameraModelId(mid), width=width, height=height,
                                params=np.frombuffer(params, np.float64).copy())
        split = json.loads((reference / "split_manifest.json").read_text(encoding="utf-8"))
        for row in [row for row in split["extra_frames"] if row["role"] == "H"][:3]:
            path = reference / "extra_images" / row["image_name"]
            inputs.append(path)
            with Image.open(path) as photo:
                pixels = np.asarray(photo.convert("RGB"))
            stage(pixels, camera, pycolmap.UndistortCameraOptions(), path.name, path)
        generator = np.random.default_rng(20261006)
        for model in ("PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"):
            for width, height in ((73, 51), (121, 89)):
                for values in ({}, {"max_image_size": 48}, {"blank_pixels": .25}, {"roi_min_x": .1, "roi_max_y": .85}):
                    current = pycolmap.Camera.create(1, model, 1.1 * width, width, height)
                    current.params[-1] = .02 if model != "PINHOLE" else current.params[-1]
                    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
                    pixels[:, width // 2] = 255
                    stage(pixels, current, pycolmap.UndistortCameraOptions(**values), f"{model}_{width}_{height}_{values}")
    finally:
        operator.close()
    input_ids = [data_identity(path) for path in inputs]
    payload_ids = [data_identity(path) for path in sorted(args.output.rglob("*.npy"))]
    manifest = {"cases": cases, "groups": list(groups.values()), "sources": source_ids, "inputs": input_ids,
                "payloads": payload_ids, "preparing_seconds": time.perf_counter() - STARTED,
                "native_pixel_seconds": native_seconds, "staging_including_native_seconds": staging,
                "map_build_seconds": operator.trace["build_seconds"], "new_optimization_count": 0}
    atomic_json(args.output / "MANIFEST.json", manifest)
    linux_output = "/mnt/c/" + args.output.resolve().as_posix()[3:]
    command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
               "/mnt/c/" + (CODE / "camera_remap_gpu_worker.py").as_posix()[3:], "--output", linux_output]
    atomic_json(args.output / "CUDA_COMMAND.json", {"command": command, "cwd": str(ROOT)})
    begin = time.perf_counter()
    child = subprocess.run(command, cwd=ROOT, check=False)
    child_whole = time.perf_counter() - begin
    if child.returncode != 0:
        atomic_json(args.output / "FAILURE.json", {"status": "CUDA_child_failed_no_acceptance", "returncode": child.returncode,
                    "child_complete_seconds": child_whole, "sources": source_ids, "inputs": input_ids, "new_optimization_count": 0})
        raise RuntimeError("The owned CUDA pixel child failed; no later experimental stage was started")
    gpu = json.loads((args.output / "CUDA_RESULT.json").read_text(encoding="utf-8"))
    rows = []
    for row in cases:
        directory = args.output / row["case"]
        before = time.perf_counter()
        pixels = np.load(directory / "sampled.npy", allow_pickle=False)
        result = pycolmap.Bitmap.from_array(pixels)
        calibration = pycolmap.Camera(**row["calibration"])
        if (calibration.width, calibration.height) != (row["camera"]["width"], row["camera"]["height"]):
            result.rescale(calibration.width, calibration.height)
        actual = result.to_array()
        expected = np.load(directory / "native.npy", allow_pickle=False)
        same_shape = actual.shape == expected.shape and actual.dtype == expected.dtype
        differences = int(np.count_nonzero(actual != expected)) if same_shape else None
        checks = {"independent_target_camera_exact": row["native_calibration"] == camera_fields(calibration),
                  "complete_shape_dtype": same_shape, "all_pixels_exact": differences == 0,
                  "native_PNG_bytes_exact": same_shape and png(actual) == png(expected)}
        rows.append({"label": row["label"], "checks": checks, "differences": differences,
                     "maximum_absolute_pixel_error": int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()) if same_shape else None,
                     "postprocess_and_full_compare_seconds": time.perf_counter() - before})
    checks = {"all43_native_complete_pixels_and_PNG": len(rows) == 43 and all(all(row["checks"].values()) for row in rows),
              "original_sources_inputs_and_payloads_unchanged": source_ids == [data_identity(path) for path in sources]
                  and input_ids == [data_identity(path) for path in inputs]
                  and payload_ids == [data_identity(Path(row["path"])) for row in payload_ids],
              "owned_CPU_maps_and_native_binding_closed": operator.closed and operator.maps is None and pycolmap.undistort_image is original,
              "actual_CUDA_child_all43_and_closed": gpu["cases"] == 43 and gpu["all_cases_sampled"] and gpu["owned_tensors_released"],
              "original_owned_output_and_disk_limits": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= 30 * 1024**3
                  and shutil.disk_usage(args.output).free >= 20 * 1024**3}
    atomic_json(args.output / "RESULT.json", {"status": "CUDA_complete_pixel_preconditions_passed" if all(checks.values()) else "CUDA_required_pixel_guard_failed",
        "checks": checks, "rows": rows, "sources": source_ids, "inputs": input_ids, "manifest": str(args.output / "MANIFEST.json"),
        "CUDA": gpu, "native_pixel_seconds": native_seconds, "map_build_seconds": operator.trace["build_seconds"],
        "preparing_seconds": manifest["preparing_seconds"], "CUDA_full_child_seconds": child_whole,
        "whole_before_final_publication_seconds": time.perf_counter() - STARTED,
        "warm_CUDA_sampling_is_not_complete_entry_speed": True, "new_optimization_count": 0,
        "complete_entry_failure_metadata_recovery_external_and_final_cost": "not_run"})
    print(json.dumps({"status": "CUDA_complete_pixel_preconditions_passed" if all(checks.values()) else "CUDA_required_pixel_guard_failed", "checks": checks, "cases": len(rows), "new_optimization_count": 0}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
