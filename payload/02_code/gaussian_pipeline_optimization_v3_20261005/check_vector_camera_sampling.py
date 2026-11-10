"""Retain all old pixel fixtures and verify strict SIMD error and alias prefixes."""

import argparse
import _ctypes
import ctypes
import difflib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pycolmap

from probe_dataset_sampling_route import ROOT, LIBRARY as SCALAR_LIBRARY
from runtime import atomic_json, data_identity
from camera_pixel_remap import camera_fields
import vector_camera_sampling as vector

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001"


def bind(library):
    function = library.native_camera_sample_rgb8
    function.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_size_t] * 2
    function.restype = ctypes.c_int
    return function


def boundary_cases(library):
    generator = np.random.default_rng(20261007)
    dlls = [ctypes.CDLL(str(path)) for path in (SCALAR_LIBRARY, library)]
    rows = []
    try:
        functions = [bind(dll) for dll in dlls]
        for height, width in ((2, 2), (9, 17), (64, 65)):
            shape = height, width
            pixels = generator.integers(0, 256, (*shape, 3), dtype=np.uint8)
            xs = generator.integers(0, width - 1, shape, dtype=np.int32)
            ys = generator.integers(0, height - 1, shape, dtype=np.int32)
            wx, wy = generator.random(shape), generator.random(shape)
            # Cover float conversion and half-step rounding with exact weights.
            weights = np.array([0., np.nextafter(0., 1.), .5, np.nextafter(.5, 0.),
                                np.nextafter(.5, 1.), 1 / 3, np.nextafter(1., 0.)])
            wx.flat[:min(wx.size, len(weights))] = weights[:min(wx.size, len(weights))]
            wy.flat[:min(wy.size, len(weights))] = weights[::-1][:min(wy.size, len(weights))]
            valid = np.ones(shape, np.uint8)
            valid.flat[::11] = 0
            for case in ("normal", "alias0", "alias1", "alias3", "x_negative", "y_upper",
                         "dx_nan", "dy_inf", "dx_negative", "dy_one"):
                arrays = [a.copy() for a in (xs, ys, wx, wy, valid)]
                position = min(7, wx.size - 1)
                if case == "x_negative": arrays[0].flat[position] = -1
                if case == "y_upper": arrays[1].flat[position] = height - 1
                if case == "dx_nan": arrays[2].flat[position] = np.nan
                if case == "dy_inf": arrays[3].flat[position] = np.inf
                if case == "dx_negative": arrays[2].flat[position] = -.1
                if case == "dy_one": arrays[3].flat[position] = 1.
                arrays[4].flat[position] = 1
                actual = []
                for function in functions:
                    if case.startswith("alias"):
                        shift = int(case[5:])
                        storage = np.zeros(pixels.size + 8, np.uint8)
                        storage[:pixels.size] = pixels.reshape(-1)
                        input_address, output_address = storage.ctypes.data, storage.ctypes.data + shift
                        output = storage
                    else:
                        storage = pixels.copy()
                        output = np.full_like(pixels, 173)
                        input_address, output_address = storage.ctypes.data, output.ctypes.data
                    code = function(input_address, *[a.ctypes.data for a in arrays], output_address, width, height)
                    actual.append((code, output.tobytes(), storage.tobytes()))
                if actual[0] != actual[1]:
                    raise AssertionError("SIMD full output or native error prefix differs: " + case)
                rows.append({"shape": list(shape), "case": case, "status": actual[0][0], "full_bytes_and_prefix_exact": True})
        shape = 3, 4
        pixels = np.zeros((*shape, 3), np.uint8)
        arrays = [np.zeros(shape, np.int32), np.zeros(shape, np.int32),
                  np.zeros(shape, np.float64), np.zeros(shape, np.float64), np.ones(shape, np.uint8)]
        for missing in range(7):
            codes = []
            for function in functions:
                output = np.full_like(pixels, 173)
                args = [pixels.ctypes.data, *[a.ctypes.data for a in arrays], output.ctypes.data]
                args[missing] = None
                codes.append(function(*args, 4, 3))
                if not np.all(output == 173):
                    raise AssertionError("An invalid pointer guard modified output")
            if codes != [1, 1]: raise AssertionError("A native pointer guard changed")
            rows.append({"case": "null" + str(missing), "status": 1, "untouched_output": True})
    finally:
        for dll in dlls:
            _ctypes.FreeLibrary(dll._handle)
    return rows


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    proof = output.with_name(output.name + "_adapter")
    proof.mkdir(parents=True, exist_ok=False)
    if not vector.library_supports_simd(args.library):
        raise RuntimeError("The current host did not admit the vector sampler")
    original = Path(__file__).with_name("check_native_camera_sampling.py")
    before = original.read_text(encoding="utf-8")
    source = before.replace('from native_camera_sampling import NativeCameraPixelRemap',
                            'from vector_camera_sampling import VectorCameraPixelRemap as NativeCameraPixelRemap')
    source = source.replace('gaussian_pipeline_optimization_v3_20261005/rm_native001',
                            'gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001')
    source = source.replace('with_name("native_camera_sampling.cpp")', 'with_name("native_vector_camera_sampling.cpp")')
    if source == before or source.count('from vector_camera_sampling') != 1:
        raise ValueError("The original43 fixture adapter anchors changed")
    (proof / "COMPLETE_EXECUTED_GENERIC_ADAPTER.py").write_text(source, encoding="utf-8")
    (proof / "GENERIC_ADAPTER.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
        fromfile=original.name, tofile="vector_sampler_exact43")), encoding="utf-8")
    sources = [Path(__file__), Path(vector.__file__), Path(__file__).with_name("native_vector_camera_sampling.cpp"),
               original, Path(__file__).with_name("native_camera_sampling.py"), SCALAR_LIBRARY, args.library.resolve()]
    identities = [data_identity(path) for path in sources]
    snapshots = proof / "source_snapshot"
    snapshots.mkdir()
    for path in sources:
        if path.suffix in (".py", ".cpp"): shutil.copyfile(path, snapshots / path.name)
    previous = sys.argv
    try:
        sys.argv = [str(original), "--library", str(args.library), "--output", str(output)]
        namespace = {"__file__": str(original), "__name__": "retained_vector_sampler43"}
        exec(compile(source, "<retained_original43_with_vector_sampler>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous
    legacy = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    boundaries = boundary_cases(args.library)
    original_callback = pycolmap.undistort_image
    old_probe = vector.library_supports_simd
    operator = None
    try:
        vector.library_supports_simd = lambda path: False
        operator = vector.VectorCameraPixelRemap(original_callback, args.library)
        camera = pycolmap.Camera.create(1, "OPENCV", 16, 17, 13)
        bitmap = pycolmap.Bitmap.from_array(np.full((13, 17, 3), 129, np.uint8))
        a, ac = original_callback(pycolmap.UndistortCameraOptions(), bitmap, camera)
        b, bc = operator(pycolmap.UndistortCameraOptions(), bitmap, camera)
        fallback = np.array_equal(a.to_array(), b.to_array()) and camera_fields(ac) == camera_fields(bc) and operator.maps is None
    finally:
        if operator is not None:
            operator.close()
        vector.library_supports_simd = old_probe
    checks = {"original43_all_full_pixels_K_PNG_and_close_guards": len(legacy["rows"]) == 43 and all(legacy["checks"].values()),
        "scalar_vector_complete37_boundary_alias_and_native_prefix_cases": len(boundaries) == 37,
        "injected_nonadmission_original_before_mapping": fallback and operator.closed,
        "actual_current_host_vector_ISA_admitted": vector.library_supports_simd(args.library),
        "original_callback_and_actual_source_identities_restored": pycolmap.undistort_image is original_callback
            and identities == [data_identity(path) for path in sources]}
    if not all(checks.values()): raise AssertionError("A vector sampling semantic protection failed")
    atomic_json(proof / "RESULT.json", {"status": "exact43_plus37_boundary_and_nonadmission_checks_passed_not_acceptance",
        "checks": checks, "boundary_cases": boundaries, "sources": identities,
        "actual_unsupported_host_not_tested_injected_nonadmission_only": True,
        "legacy_H3_semantic_fixtures_already_previously_used_not_blind_GT": True,
        "complete_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    print(json.dumps({"status": "semantic_checks_passed", "legacy43": len(legacy["rows"]),
                      "boundary37": len(boundaries), "checks": checks, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
