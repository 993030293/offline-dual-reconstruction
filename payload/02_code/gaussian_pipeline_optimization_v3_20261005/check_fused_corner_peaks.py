"""Verify original finite peak semantics, complete corner calls, and native ownership."""

import argparse
import ctypes
import importlib
import json
from pathlib import Path
import pickle
import random
import shutil
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fcp001"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from build_fused_corner_peaks import build_library
from fused_corner_peaks import FusedPeakCornerWorkspace
from planar_corner import PlanarCornerWorkspace, ORIGINAL_CORNER, PARAMETERS


def identity(value):
    return None if value is None else (value.shape, value.strides, str(value.dtype), value.tobytes())


def direct(native, fused, response):
    old, new = response.copy(), response.copy()
    _, maximum, _, _ = cv2.minMaxLoc(old)
    cv2.threshold(old, maximum * .01, 0, cv2.THRESH_TOZERO, dst=old)
    dilation = cv2.dilate(old, None)
    values = []
    for candidate in (False, True):
        output = np.full((800, 1, 2), -129., np.float32)
        count = ctypes.c_int(-1)
        code = fused.peak_select(new.ctypes.data, *new.shape, output.ctypes.data, ctypes.byref(count)) if candidate else native.select(
            old.ctypes.data, dilation.ctypes.data, *old.shape, output.ctypes.data, ctypes.byref(count))
        values.append((code, count.value, output[:count.value].tobytes()))
    return values[0] == values[1] and old.tobytes() == new.tobytes(), values[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OUT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise OSError("The registered free-disk reserve is unavailable")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    parent_library = OUT.parent / "qc001/b0/native_planar_corner.dll"
    paths = [Path(__file__), CODE / "fused_corner_peaks.py", CODE / "native_fused_corner_peaks.cpp", CODE / "build_fused_corner_peaks.py",
        CODE / "planar_corner.py", CODE / "native_planar_corner.cpp", parent_library, Path(baseline.__file__),
        baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py"]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        if path.suffix != ".dll":
            (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
    library = build_library(output / "build")
    native = PlanarCornerWorkspace(parent_library)
    fused = FusedPeakCornerWorkspace(parent_library, library)
    old_threads, original_api = cv2.getNumThreads(), cv2.goodFeaturesToTrack
    rng = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))
    cv2.setNumThreads(1)
    checks, priority_rows, components = {}, [], []
    try:
        for shape in ((1, 1), (1, 13), (13, 1), (2, 2), (3, 3), (7, 11), (32, 48), (620, 812)):
            for kind in ("zero", "ties", "signed", "sparse"):
                response = np.ones(shape, np.float32) if kind == "ties" else np.zeros(shape, np.float32)
                if kind == "signed":
                    response[:] = (np.arange(response.size).reshape(shape) % 7 - 3).astype(np.float32)
                if kind == "sparse":
                    response[::6, ::6] = 1.25
                exact, count = direct(native, fused, response)
                priority_rows.append({"shape": shape, "kind": kind, "exact_response_and_coords": exact, "selected": count})
        checks["all32_original_peak_tie_signed_sparse_and800_fixtures_exact"] = all(row["exact_response_and_coords"] for row in priority_rows)
        edge_bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
            0x3F7FFFFF, 0x3F800000, 0x3F800001, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32).view(np.float32)
        edges = []
        for shape in ((7, 11), (32, 48), (620, 812)):
            exact, count = direct(native, fused, np.resize(edge_bits, shape).copy())
            edges.append({"shape": shape, "exact_response_and_coords": exact, "selected": count})
        checks["three_finite_float32_boundary_threshold_cases_exact"] = all(row["exact_response_and_coords"] for row in edges)
        invalids = []
        for value in (np.inf, -np.inf, np.nan):
            response = np.ones((7, 11), np.float32)
            response[3, 4] = value
            saved = response.tobytes()
            output_xy = np.full((800, 1, 2), -129., np.float32)
            output_saved = output_xy.tobytes()
            count = ctypes.c_int(-1)
            code = fused.peak_select(response.ctypes.data, *response.shape, output_xy.ctypes.data, ctypes.byref(count))
            invalids.append(code == 3 and count.value == -1 and response.tobytes() == saved and output_xy.tobytes() == output_saved)
        checks["private_nonfinite_inputs_error_before_any_partial_modification"] = all(invalids)
        private_rng = np.random.default_rng(161)
        yy, xx = np.indices((47, 61))
        fixtures = [np.zeros((1, 1), np.uint8), np.zeros((47, 61), np.uint8), np.full((47, 61), 255, np.uint8),
            ((xx + yy) % 2 * 255).astype(np.uint8), (xx * 7 % 256).astype(np.uint8),
            ((xx * xx + yy * yy) % 256).astype(np.uint8), private_rng.integers(0, 256, (47, 61), dtype=np.uint8),
            private_rng.integers(0, 256, (620, 812), dtype=np.uint8)]
        generic = []
        for image in fixtures:
            left, right = native.corners(image), fused.corners(image)
            generic.append(identity(left) == identity(right) == identity(ORIGINAL_CORNER(image, **PARAMETERS))
                and native.state(image)["planes"]["response"].tobytes() == fused.state(image)["planes"]["response"].tobytes())
        checks["all8_generic_original_current_corner_and_response_bytes_exact"] = all(generic)
        fallback = private_rng.random((32, 48), dtype=np.float32)
        checks["float32_original_API_route_exact"] = identity(native.goodFeaturesToTrack(fallback, **PARAMETERS)) == identity(fused.goodFeaturesToTrack(fallback, **PARAMETERS))
        checks["changed_parameter_original_API_route_exact"] = identity(native.goodFeaturesToTrack(fixtures[-1], maxCorners=17, qualityLevel=.03, minDistance=4, blockSize=3)) == identity(fused.goodFeaturesToTrack(fixtures[-1], maxCorners=17, qualityLevel=.03, minDistance=4, blockSize=3))
        errors = []
        for owner in (native, fused):
            try:
                owner.goodFeaturesToTrack(np.zeros((8, 8), np.uint16), **PARAMETERS)
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
        checks["original_unsupported_dtype_error_propagates"] = len(errors) == 2 and errors[0] == errors[1]
        raw_stat = baseline.RAW.stat()
        raw_identity = (raw_stat.st_size, raw_stat.st_mtime_ns)
        index = McapIndex(baseline.RAW)
        if index.message_count(baseline.CAMERA_TOPIC) != 7338:
            raise ValueError("The original RAW frame denominator changed")
        frozen = json.loads((OUT.parent / "bg001/s0/CONTRACT.json").read_text(encoding="utf-8"))["selected_ids"]
        selected = [frozen[position] for position in np.linspace(0, len(frozen) - 1, 12, dtype=int)]
        images = []
        for number, message in index.iter_selected_messages(baseline.CAMERA_TOPIC, selected):
            stamp, optical, raw = decode_bayer(message["payload"])
            gray = cv2.cvtColor(demosaic_rggb8_to_bgr(raw), cv2.COLOR_BGR2GRAY)
            image = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            saved = image.tobytes()
            left, right = native.corners(image), fused.corners(image)
            components.append({"frame": number, "header_ns": int(stamp), "optical_frame": optical,
                "all_original_and_current_coords_and_response_bytes_exact": identity(left) == identity(right) == identity(ORIGINAL_CORNER(image, **PARAMETERS))
                    and native.state(image)["planes"]["response"].tobytes() == fused.state(image)["planes"]["response"].tobytes(),
                "image_unchanged": image.tobytes() == saved})
            images.append(image)
        checks["all12_frozen_RAW_full_corner_and_response_bytes_exact"] = len(images) == 12 and all(row["all_original_and_current_coords_and_response_bytes_exact"] and row["image_unchanged"] for row in components)
        pairs = []
        for trial in range(3):
            row, values = {}, {}
            for mode in (("native", "fused") if trial % 2 == 0 else ("fused", "native")):
                begin = time.perf_counter()
                result = [(native if mode == "native" else fused).corners(image) for image in images]
                row[mode] = time.perf_counter() - begin
                values[mode] = [identity(value) for value in result]
            if values["native"] != values["fused"]:
                raise AssertionError("A measured complete original corner output changed")
            pairs.append(row)
        medians = {mode: statistics.median(row[mode] for row in pairs) for mode in ("native", "fused")}
        reduction = 1 - medians["fused"] / medians["native"]
        eligible = reduction >= .05 and all(row["fused"] / row["native"] <= 1.1 for row in pairs)
        stat = baseline.RAW.stat()
        checks["RAW_size_mtime_unchanged_not_whole_RAW_SHA"] = (stat.st_size, stat.st_mtime_ns) == raw_identity
        left, right = fused.corners(fixtures[-1]), fused.corners(fixtures[-1])
        saved = identity(right)
        left[0] = -1
        checks["original_independent_mutable_output_storage"] = identity(right) == saved and not np.shares_memory(left, right)
        checks["global_Python_NumPy_RNG_preserved"] = rng == (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))
        checks["all_executed_original_and_private_sources_unchanged"] = sources == [data_identity(path) for path in paths]
    finally:
        native.close()
        fused.close()
        cv2.setNumThreads(old_threads)
    checks["original_OpenCV_API_and_thread_count_restored"] = cv2.goodFeaturesToTrack is original_api and cv2.getNumThreads() == old_threads
    checks["both_original_response_and_private_peak_DLLs_closed"] = native.receipt()["closed"] and fused.receipt()["closed"] and fused.peak_dll is None
    checks["registered_output_and_disk_limits_kept"] = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 1024**3 and shutil.disk_usage(output).free >= 20 * 1024**3
    status = "generic_passed_component_screen_positive128_pending0" if eligible and all(checks.values()) else "closed_negative_component_screen0" if all(checks.values()) else "failed_verification"
    atomic_json(output / "RESULT.json", {"status": status, "checks": checks, "priority_cases": priority_rows,
        "float32_boundary_cases": edges, "actual_RAW_components": components, "component_pairs": pairs,
        "component_medians": medians, "component_reduction_not_full_acceptance": reduction,
        "sources": sources, "library": data_identity(library), "actual_seconds_before_publication": time.perf_counter() - started,
        "complete128_full7338_public_fault_cancel_recovery_training_production": "not_run", "new_optimization_count": 0})
    print(json.dumps({"status": status, "checks": checks, "component_medians": medians, "component_reduction": reduction}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original fused-peak contract failed")


if __name__ == "__main__":
    main()
