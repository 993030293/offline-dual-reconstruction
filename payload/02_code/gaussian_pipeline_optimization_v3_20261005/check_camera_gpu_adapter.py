"""Verify the actual stdio adapter, original route guards and complete cleanup."""

import time
STARTED = time.perf_counter()

import argparse
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from camera_pixel_gpu_remap import GpuCameraPixelRemap, camera_pixel_gpu_remap
from camera_pixel_remap import camera_fields


def png(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001")
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = args.output.parent / "g0"
    manifest = json.loads((oracle / "MANIFEST.json").read_text(encoding="utf-8"))
    first = json.loads((oracle / "RESULT.json").read_text(encoding="utf-8"))
    if not all(first["checks"].values()) or len(manifest["cases"]) != 43:
        raise ValueError("The fixed complete original pixel oracle did not pass")
    paths = [Path(__file__), CODE / "camera_pixel_gpu_remap.py", CODE / "camera_remap_stdio_worker.py",
             CODE / "gpu_camera_remap_pixels.py", CODE / "camera_pixel_remap.py", CODE / "mapped_image_linux_guard.py"]
    sources = [data_identity(path) for path in paths]
    copies = args.output / "source_snapshot"
    copies.mkdir()
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{ordinal:03d}_" + path.name))
    inputs = [oracle / "MANIFEST.json", oracle / "RESULT.json"]
    inputs += [oracle / row["case"] / name for row in manifest["cases"] for name in ("raw.npy", "native.npy")]
    identities = [data_identity(path) for path in inputs]
    original = pycolmap.undistort_image
    rows = []
    with camera_pixel_gpu_remap(True, args.output / "pixels") as trace:
        for row in manifest["cases"]:
            raw = np.load(oracle / row["case"] / "raw.npy", allow_pickle=False)
            expected = np.load(oracle / row["case"] / "native.npy", allow_pickle=False)
            camera = pycolmap.Camera(**row["camera"])
            if row["label"].startswith("frame_"):
                options = pycolmap.UndistortCameraOptions()
            else:
                import ast
                options = pycolmap.UndistortCameraOptions(**ast.literal_eval(row["label"][row["label"].index("{"):]))
            actual, calibration = pycolmap.undistort_image(options, pycolmap.Bitmap.from_array(raw), camera)
            pixels = actual.to_array()
            checks = {"complete_shape_dtype": pixels.shape == expected.shape and pixels.dtype == expected.dtype,
                      "all_pixels_exact": np.array_equal(pixels, expected),
                      "complete_calibration_exact": camera_fields(calibration) == row["native_calibration"],
                      "complete_original_PNG_bytes_exact": png(pixels) == png(expected)}
            rows.append({"label": row["label"], "checks": checks})
        if not all(all(row["checks"].values()) for row in rows):
            raise ValueError("The actual stdio adapter changed a complete original pixel result")
    guard_checks = {}
    small = pycolmap.Camera.create(1, "PINHOLE", 64., 73, 51)
    raw = np.zeros((51, 73, 3), np.uint8)
    sentinel, calls = object(), []
    def untouched(*arguments):
        calls.append(arguments)
        return sentinel
    operator = GpuCameraPixelRemap(untouched, args.output / "budget", True, maximum_bytes=1)
    guard_checks["small_budget_uses_original_before_mapping_without_worker"] = operator(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(raw), small) is sentinel and operator.worker is None
    operator.close()
    try:
        operator(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(raw), small)
        guard_checks["closed_operator_rejects_reuse"] = False
    except RuntimeError:
        guard_checks["closed_operator_rejects_reuse"] = True
    try:
        GpuCameraPixelRemap(original, args.output / "metadata", False)
        guard_checks["unasserted_metadata_scope_rejected"] = False
    except ValueError:
        guard_checks["unasserted_metadata_scope_rejected"] = not (args.output / "metadata").exists()
    operator = GpuCameraPixelRemap(untouched, args.output / "unsupported", True)
    gray = pycolmap.Bitmap.from_array(np.zeros((51, 73), np.uint8))
    guard_checks["gray_uses_original_before_mapping"] = operator(pycolmap.UndistortCameraOptions(), gray, small) is sentinel
    unsupported = pycolmap.Camera.create(1, "FOV", 64., 73, 51)
    guard_checks["unsupported_camera_uses_original_before_mapping"] = operator(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(raw), unsupported) is sentinel and operator.worker is None
    operator.close()
    try:
        with camera_pixel_gpu_remap(True, args.output / "exception") as failed_trace:
            pycolmap.undistort_image(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(raw), small)
            raise ArithmeticError("injected_owned_reference_consumer_failure")
    except ArithmeticError:
        guard_checks["consumer_failure_restores_and_closes_actual_worker"] = failed_trace["closed"] and failed_trace["original_global_binding_restored"] and failed_trace["worker"]["closed"] and failed_trace["worker"]["returncode"] == 0
    try:
        with camera_pixel_gpu_remap(True, args.output / "wire_error") as wire_trace:
            pycolmap.undistort_image(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(raw), small)
            pycolmap.undistort_image.worker.rpc({"command": "injected_unsupported_wire_command"})
    except RuntimeError as error:
        failure = json.loads((args.output / "wire_error/WORKER_RESULT.json").read_text(encoding="utf-8"))
        guard_checks["actual_worker_error_propagates_without_CPU_fallback_and_closes"] = "unsupported" in str(error) and wire_trace["closed"] and wire_trace["original_global_binding_restored"] and wire_trace["worker"]["returncode"] == 1 and failure["status"] == "failed" and len(wire_trace["calls"]) == 1
    worker = trace["worker"]
    gpu = json.loads((args.output / "pixels/WORKER_RESULT.json").read_text(encoding="utf-8"))
    resource = json.loads((args.output / "pixels/WSL_RESOURCE.json").read_text(encoding="utf-8"))
    checks = {"all43_actual_stdio_complete_pixels_calibration_PNG": len(rows) == 43 and all(all(row["checks"].values()) for row in rows),
              "actual_one_worker_all43_map_and_CUDA_guards_closed": len(gpu["rows"]) == 43 and all(gpu["checks"].values()) and trace["closed"] and worker["closed"] and worker["returncode"] == 0 and resource["status"] == "completed",
              "all_registered_original_error_and_scope_guards": all(guard_checks.values()),
              "all_sources_and_oracle_payloads_unchanged": sources == [data_identity(path) for path in paths] and identities == [data_identity(path) for path in inputs],
              "original_global_native_binding_restored": pycolmap.undistort_image is original}
    atomic_json(args.output / "RESULT.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "guard_checks": guard_checks, "rows": rows, "sources": sources, "inputs": identities, "trace": trace,
        "consumer_failure_trace": failed_trace, "whole_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
    print(json.dumps({"status": "passed" if all(checks.values()) else "failed", "checks": checks}), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
