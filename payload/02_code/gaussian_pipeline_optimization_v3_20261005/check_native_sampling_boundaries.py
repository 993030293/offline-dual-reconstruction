"""Check native sampling ownership and actual unsupported input/error boundaries."""

import argparse
import ctypes
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from camera_pixel_remap import camera_fields
from native_camera_sampling import NativeCameraPixelRemap, native_camera_sampling


def reject(callback, error):
    try:
        callback()
    except error:
        return True
    raise AssertionError("An invalid native sampling input was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__), CODE/"native_camera_sampling.py", CODE/"camera_pixel_remap.py",
               CODE/"native_camera_sampling.cpp", args.library]
    identities = [data_identity(p) for p in sources]
    original = pycolmap.undistort_image
    options = pycolmap.UndistortCameraOptions()
    camera = pycolmap.Camera.create(1, "PINHOLE", 35., 32, 24)
    raw = np.random.default_rng(20261007).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    checks, traces = {}, []
    operator = NativeCameraPixelRemap(original, args.library)
    try:
        bitmap = pycolmap.Bitmap.from_array(raw)
        operator(options, bitmap, camera)
        maps = operator.maps
        checks["reject_strided_uint8_before_native_pointer"] = reject(lambda: operator.sample_array(raw[:, ::-1]), ValueError)
        checks["reject_changed_dtype_before_native_pointer"] = reject(lambda: operator.sample_array(raw.astype(np.float64)), ValueError)
        operator.maps = maps[:4]
        checks["reject_missing_map_before_native_pointer"] = reject(lambda: operator.sample_array(raw), ValueError)
        operator.maps = maps
        forged = list(maps)
        forged[0] = np.full(maps[0].shape, 32, np.int32)
        forged[0].setflags(write=False)
        operator.maps = tuple(forged)
        checks["native_outside_source_boundary_returns_error"] = reject(lambda: operator.sample_array(raw), ValueError)
        operator.maps = maps
        original_pixels, _ = original(options, bitmap, camera)
        checks["reuse_after_rejected_input_pixels_exact"] = np.array_equal(operator(options, bitmap, camera)[0].to_array(), original_pixels.to_array())
    finally:
        operator.close()
        traces.append(operator.trace)
    checks["closed_operator_rejects_reuse"] = reject(lambda: operator(options, bitmap, camera), RuntimeError)
    checks["closed_maps_and_DLL"] = operator.maps is None and operator.trace["native_DLL_closed"]
    for label, pixels, current, maximum in (
            ("one_pixel_original_route", np.zeros((1, 1, 3), np.uint8), pycolmap.Camera.create(1, "PINHOLE", 1., 1, 1), 256*1024**2),
            ("grayscale_original_route", raw[:, :, 0].copy(), camera, 256*1024**2),
            ("budget_original_route", raw, camera, 1)):
        bitmap = pycolmap.Bitmap.from_array(pixels)
        expected, expected_camera = original(options, bitmap, current)
        operator = NativeCameraPixelRemap(original, args.library, maximum_bytes=maximum)
        try:
            actual, actual_camera = operator(options, bitmap, current)
            checks[label] = np.array_equal(expected.to_array(), actual.to_array()) and camera_fields(expected_camera) == camera_fields(actual_camera)
            checks[label+"_before_mapping"] = operator.trace["routes"] == ["original_before_mapping"] and operator.maps is None
        finally:
            operator.close()
            traces.append(operator.trace)
    injected = False
    try:
        with native_camera_sampling(True, args.library) as trace:
            pycolmap.undistort_image(options, pycolmap.Bitmap.from_array(raw), camera)
            raise InterruptedError("Owned consumer cancellation fixture")
    except InterruptedError:
        injected = True
    checks["consumer_cancel_propagated_and_callback_restored"] = injected and trace["closed"] and trace["original_binding_restored"] and pycolmap.undistort_image is original
    checks["consumer_cancel_maps_DLL_closed"] = trace["operator"]["closed"] and trace["operator"]["native_DLL_closed"]
    checks["all_DLL_owners_closed"] = all(t["closed"] and t["native_DLL_closed"] for t in traces)
    module_handle = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleHandleW
    module_handle.argtypes, module_handle.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    checks["actual_DLL_module_unloaded_after_close"] = module_handle(str(args.library.resolve())) is None
    loader = ctypes.CDLL
    def changed_abi(path):
        library = loader(path)
        library.native_camera_sampling_abi = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 99)
        return library
    try:
        ctypes.CDLL = changed_abi
        checks["actual_DLL_with_injected_invalid_ABI_rejected"] = reject(lambda: NativeCameraPixelRemap(original, args.library), ValueError)
    finally:
        ctypes.CDLL = loader
    checks["invalid_ABI_module_unloaded_and_loader_restored"] = module_handle(str(args.library.resolve())) is None and ctypes.CDLL is loader
    checks["global_callback_still_original"] = pycolmap.undistort_image is original
    checks["sources_unchanged"] = identities == [data_identity(p) for p in sources]
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
              "sources": identities, "traces": traces, "cancellation": trace,
              "whole_seconds": time.perf_counter()-started, "new_optimization_count": 0}
    atomic_json(args.output/"RESULT.json", result)
    print(result["status"], len(checks), flush=True)
    if not all(checks.values()):
        raise AssertionError("An actual native sampling ownership guard failed")


if __name__ == "__main__":
    main()
