"""Exercise actual undistortion, exact PNGs and calibrated cache failure boundaries."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pycolmap

import calibrated_reference_cache as implementation
from calibrated_reference_cache import CalibratedReferenceCache, calibrated_reference_cache, camera_fields

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = np.random.default_rng(3).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    camera = pycolmap.Camera(model="OPENCV", width=320, height=240,
        params=[250., 248., 159.5, 119.5, .01, -.003, .001, -.001])
    options = pycolmap.UndistortCameraOptions()
    original_undistort, original_save = pycolmap.undistort_image, Image.Image.save
    oracle, oracle_camera = original_undistort(options, pycolmap.Bitmap.from_array(raw), camera)
    original_save(Image.fromarray(oracle.to_array()), args.output/"oracle.png")
    checks = {}

    def rejected(call, kind):
        try:
            call()
        except kind:
            return True
        return False

    def run(root, name, pixels=raw, selected_camera=camera, selected_options=options):
        with calibrated_reference_cache(root) as receipt:
            bitmap, calibration = pycolmap.undistort_image(selected_options, pycolmap.Bitmap.from_array(pixels), selected_camera)
            Image.fromarray(bitmap.to_array()).save(args.output/(name+".png"))
            value = receipt["cache"].receipt()
        return bitmap.to_array(), calibration, value

    root = args.output/"cache"
    cold = run(root, "cold")
    warm = run(root, "warm")
    checks["actual_cold_warm_pixels_and_PNG_bytes_exact"] = (np.array_equal(oracle.to_array(), cold[0])
        and np.array_equal(oracle.to_array(), warm[0])
        and (args.output/"oracle.png").read_bytes() == (args.output/"warm.png").read_bytes())
    checks["complete_returned_calibration_exact"] = camera_fields(oracle_camera) == camera_fields(cold[1]) == camera_fields(warm[1])
    checks["actual_cold_and_reopened_warm"] = cold[2]["misses"] == 1 and warm[2]["hits"] == 1
    checks["callbacks_restored_and_owner_released"] = (pycolmap.undistort_image is original_undistort
        and Image.Image.save is original_save and not (root/"OWNER.json").exists())
    changed = raw.copy()
    changed[100:120, 90:110] ^= 255
    checks["raw_pixels_invalidate"] = run(root, "changed_raw", pixels=changed)[2]["misses"] == 1
    changed_camera = pycolmap.Camera(camera.todict())
    changed_camera.params[4] += .01
    checks["full_camera_calibration_invalidate"] = run(root, "changed_camera", selected_camera=changed_camera)[2]["misses"] == 1
    changed_options = pycolmap.UndistortCameraOptions(max_image_size=200)
    checks["output_undistortion_options_invalidate"] = run(root, "changed_options", selected_options=changed_options)[2]["misses"] == 1
    original_identity = implementation.file_identity

    def changed_identity(path):
        value = original_identity(path)
        if Path(path).resolve() == Path(implementation.__file__).resolve():
            value["sha256"] = "0"*64
        return value

    with patch.object(implementation, "file_identity", changed_identity):
        checks["producer_source_change_invalidate"] = run(root, "changed_source")[2]["misses"] == 1
    with calibrated_reference_cache(root) as receipt:
        checks["concurrent_writer_rejected"] = rejected(
            lambda: CalibratedReferenceCache(root, original_undistort, original_save), FileExistsError)
    cache = CalibratedReferenceCache(args.output/"mutation", original_undistort, original_save)
    bitmap, calibration = cache.undistort(options, pycolmap.Bitmap.from_array(raw), camera)
    modified = bitmap.to_array().copy()
    modified[0, 0] ^= 255
    checks["changed_pixels_before_PNG_save_rejected"] = rejected(
        lambda: cache.save(Image.fromarray(modified), args.output/"mutated.png"), ValueError)
    checks["next_query_before_save_rejected"] = rejected(
        lambda: cache.undistort(options, pycolmap.Bitmap.from_array(raw), camera), RuntimeError)
    cache.close()
    tiny = CalibratedReferenceCache(args.output/"tiny", original_undistort, original_save, maximum_bytes=1)
    bitmap, calibration = tiny.undistort(options, pycolmap.Bitmap.from_array(raw), camera)
    checks["complete_cold_artifact_budget_failure_unpublished"] = rejected(
        lambda: tiny.save(Image.fromarray(bitmap.to_array()), args.output/"tiny.png"), MemoryError)
    checks["budget_failure_has_no_complete_metadata"] = not list((args.output/"tiny").glob("*/ENTRY.json"))
    tiny.close()
    key = cold[2]["records"][0]["key"]
    for kind, filename in (("corrupt_pixels", "pixels.npy"), ("corrupt_PNG", "reference.png"), ("incomplete", "ENTRY.json")):
        fixture = args.output/kind
        shutil.copytree(root, fixture)
        payload = fixture/key/filename
        if kind == "incomplete":
            payload.unlink()
            exception = RuntimeError
        else:
            value = bytearray(payload.read_bytes())
            value[-12] ^= 255
            payload.write_bytes(value)
            exception = ValueError
        checks[kind+"_rejected"] = rejected(lambda: run(fixture, kind), exception)
        checks[kind+"_owner_released_after_failure"] = not (fixture/"OWNER.json").exists()
    with calibrated_reference_cache(root) as receipt:
        bitmap, calibration = pycolmap.undistort_image(options, pycolmap.Bitmap.from_array(raw), camera)
        checks["fresh_target_overwrite_rejected"] = rejected(
            lambda: Image.fromarray(bitmap.to_array()).save(args.output/"warm.png"), FileExistsError)
    checks["all_failure_callbacks_restored"] = pycolmap.undistort_image is original_undistort and Image.Image.save is original_save
    result = {"status": "actual_calibrated_cache_lifecycle_passed" if len(checks) == 21 and all(checks.values()) else "failed",
        "checks": checks, "sources": [data_identity(Path(__file__)), data_identity(Path(implementation.__file__))],
        "fixture_role": "cache_calibration_and_lifecycle; actual24_performance_is_separate", "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] == "failed":
        raise AssertionError("Calibrated reference cache lifecycle guards failed")


if __name__ == "__main__":
    main()
