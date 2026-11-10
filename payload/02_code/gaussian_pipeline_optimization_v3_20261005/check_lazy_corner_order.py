"""Check exact priority selection and complete current corner calls, without counting."""

import argparse
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from build_lazy_corner_order import build_library
from lazy_corner_order import LazyCornerWorkspace
from planar_corner import PlanarCornerWorkspace, ORIGINAL_CORNER, PARAMETERS


def identity(value):
    return None if value is None else (value.shape, value.strides, str(value.dtype), value.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OUT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    parent_library = OUT.parent / "qc001/b0/native_planar_corner.dll"
    paths = [Path(__file__), CODE / "lazy_corner_order.py", CODE / "native_lazy_corner_order.cpp",
        CODE / "build_lazy_corner_order.py", CODE / "planar_corner.py", CODE / "native_planar_corner.cpp",
        parent_library, Path(baseline.__file__), baseline.TOOLS / "mcap_audit.py",
        baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py"]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        if path.suffix != ".dll":
            (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
    library = build_library(output / "build")
    native, lazy = PlanarCornerWorkspace(parent_library), LazyCornerWorkspace(parent_library, library)
    old_threads, original_api = cv2.getNumThreads(), cv2.goodFeaturesToTrack
    rng = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))
    cv2.setNumThreads(1)
    checks, rows, components = {}, [], []
    try:
        for shape in ((1, 1), (1, 13), (13, 1), (2, 2), (3, 3), (7, 11), (32, 48), (620, 812)):
            for kind in ("zero", "ties", "signed", "sparse"):
                response = np.ones(shape, np.float32) if kind == "ties" else np.zeros(shape, np.float32)
                if kind == "signed":
                    response[:] = (np.arange(response.size).reshape(shape) % 7 - 3).astype(np.float32)
                if kind == "sparse":
                    response[::6, ::6] = 1.25
                dilated = response.copy()
                results = []
                for workspace in (native, lazy):
                    output_xy = np.full((800, 1, 2), -129., np.float32)
                    count = ctypes.c_int(-1)
                    code = workspace.select(response.ctypes.data, dilated.ctypes.data, *shape,
                        output_xy.ctypes.data, ctypes.byref(count))
                    results.append((code, count.value, output_xy[:count.value].tobytes()))
                rows.append({"shape": shape, "kind": kind, "complete_coords_order_and_None_exact": results[0] == results[1],
                    "selected": results[0][1]})
        checks["all32_ties_signed_dense_sparse_small_and800_priority_cases_exact"] = len(rows) == 32 and all(row["complete_coords_order_and_None_exact"] for row in rows)
        private_rng = np.random.default_rng(160)
        yy, xx = np.indices((47, 61))
        fixtures = [np.zeros((1, 1), np.uint8), np.zeros((47, 61), np.uint8), np.full((47, 61), 255, np.uint8),
            ((xx + yy) % 2 * 255).astype(np.uint8), (xx * 7 % 256).astype(np.uint8),
            ((xx * xx + yy * yy) % 256).astype(np.uint8), private_rng.integers(0, 256, (47, 61), dtype=np.uint8),
            private_rng.integers(0, 256, (620, 812), dtype=np.uint8)]
        checks["all8_generic_current_response_original_OpenCV_and_heap_exact"] = all(
            identity(native.corners(image)) == identity(lazy.corners(image)) == identity(ORIGINAL_CORNER(image, **PARAMETERS)) for image in fixtures)
        fallback = private_rng.random((32, 48), dtype=np.float32)
        checks["float32_original_route_exact"] = identity(native.goodFeaturesToTrack(fallback, **PARAMETERS)) == identity(lazy.goodFeaturesToTrack(fallback, **PARAMETERS))
        checks["changed_parameter_original_route_exact"] = identity(native.goodFeaturesToTrack(fixtures[-1], maxCorners=17, qualityLevel=.03, minDistance=4, blockSize=3)) == identity(lazy.goodFeaturesToTrack(fixtures[-1], maxCorners=17, qualityLevel=.03, minDistance=4, blockSize=3))
        errors = []
        for workspace in (native, lazy):
            try:
                workspace.goodFeaturesToTrack(np.zeros((8, 8), np.uint16), **PARAMETERS)
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
        checks["original_unsupported_dtype_error_propagates"] = len(errors) == 2 and errors[0] == errors[1]
        raw_stat = baseline.RAW.stat()
        raw_identity = (raw_stat.st_size, raw_stat.st_mtime_ns)
        index = McapIndex(baseline.RAW)
        if index.message_count(baseline.CAMERA_TOPIC) != 7338:
            raise ValueError("The original raw frame denominator changed")
        frozen = json.loads((OUT.parent / "bg001/s0/CONTRACT.json").read_text(encoding="utf-8"))["selected_ids"]
        selected = [frozen[position] for position in np.linspace(0, len(frozen) - 1, 12, dtype=int)]
        images = []
        for number, message in index.iter_selected_messages(baseline.CAMERA_TOPIC, selected):
            stamp, optical, raw = decode_bayer(message["payload"])
            gray = cv2.cvtColor(demosaic_rggb8_to_bgr(raw), cv2.COLOR_BGR2GRAY)
            image = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            original_bytes = image.tobytes()
            one, two, oracle = native.corners(image), lazy.corners(image), ORIGINAL_CORNER(image, **PARAMETERS)
            components.append({"frame": number, "header_ns": int(stamp), "optical_frame": optical,
                "all_current_and_original_coords_order_exact": identity(one) == identity(two) == identity(oracle),
                "image_unchanged": image.tobytes() == original_bytes})
            images.append(image)
        checks["12_frozen_full_real_RAW_frames_original_current_and_heap_exact"] = len(images) == 12 and all(row["all_current_and_original_coords_order_exact"] and row["image_unchanged"] for row in components)
        pairs = []
        for trial in range(3):
            row = {}
            outputs = {}
            for mode in (("native", "lazy") if trial % 2 == 0 else ("lazy", "native")):
                begin = time.perf_counter()
                values = [(native if mode == "native" else lazy).corners(image) for image in images]
                row[mode] = time.perf_counter() - begin
                outputs[mode] = [identity(value) for value in values]
            if outputs["native"] != outputs["lazy"]:
                raise AssertionError("The complete component changed the original corner output")
            pairs.append(row)
        medians = {mode: statistics.median(row[mode] for row in pairs) for mode in ("native", "lazy")}
        stat = baseline.RAW.stat()
        checks["raw_size_mtime_unchanged_not_full_RAW_SHA_claim"] = (stat.st_size, stat.st_mtime_ns) == raw_identity
        one, two = lazy.corners(fixtures[-1]), lazy.corners(fixtures[-1])
        if one is None or two is None:
            raise AssertionError("The independent-storage fixture lost all corners")
        saved = identity(two)
        one[0] = -1
        checks["outputs_have_original_independent_mutable_storage"] = identity(two) == saved and not np.shares_memory(one, two)
        checks["global_Python_NumPy_RNG_unchanged"] = rng == (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))
        checks["all_original_scientific_sources_and_library_unchanged"] = sources == [data_identity(path) for path in paths]
    finally:
        native.close()
        lazy.close()
        cv2.setNumThreads(old_threads)
    checks["original_OpenCV_api_and_threads_restored"] = cv2.goodFeaturesToTrack is original_api and cv2.getNumThreads() == old_threads
    checks["both_response_planes_and_new_priority_DLL_closed"] = native.receipt()["closed"] and lazy.receipt()["closed"] and lazy.order_dll is None
    atomic_json(output / "RESULT.json", {"status": "exact_generic_and_real_component_passed_full_consumers_pending0" if all(checks.values()) else "failed_verification",
        "checks": checks, "priority_cases": rows, "actual_RAW_components": components, "component_pairs": pairs,
        "component_medians": medians, "component_raw_reduction_not_whole_acceptance": 1 - medians["lazy"] / medians["native"],
        "sources": sources, "library": data_identity(library), "actual_seconds_before_publication": time.perf_counter() - started,
        "complete128_full7338_public_fault_cancel_recovery_training_production": "not_run", "new_optimization_count": 0})
    print(json.dumps({"checks": checks, "component_medians": medians, "component_raw_reduction_not_whole_acceptance": 1 - medians["lazy"] / medians["native"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A current-response heap corner oracle failed")


if __name__ == "__main__":
    main()
