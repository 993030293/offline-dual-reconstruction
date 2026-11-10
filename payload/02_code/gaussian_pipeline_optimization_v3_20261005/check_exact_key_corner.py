"""Compare both exact-key variants, original corner outputs, and registered scope."""

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
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco002"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from build_exact_key_corner import build_library
from exact_key_corner import KeyCornerWorkspace
from planar_corner import PlanarCornerWorkspace, ORIGINAL_CORNER, PARAMETERS


def identity(value):
    return None if value is None else (value.shape, value.strides, str(value.dtype), value.tobytes())


def direct(workspace, response, dilated):
    output = np.full((800, 1, 2), -129., np.float32)
    count = ctypes.c_int(-1)
    code = workspace.select(response.ctypes.data, dilated.ctypes.data, *response.shape,
        output.ctypes.data, ctypes.byref(count))
    return code, count.value, output[:count.value].tobytes()


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
    paths = [Path(__file__), CODE / "exact_key_corner.py", CODE / "native_exact_key_corner.cpp", CODE / "build_exact_key_corner.py",
        CODE / "planar_corner.py", CODE / "native_planar_corner.cpp", parent_library, Path(baseline.__file__),
        baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py"]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        if path.suffix != ".dll":
            (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
    library = build_library(output / "build")
    owners = {"native": PlanarCornerWorkspace(parent_library),
        **{mode: KeyCornerWorkspace(parent_library, library, mode) for mode in ("packed", "radix")}}
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
                values = [direct(owner, response, response.copy()) for owner in owners.values()]
                priority_rows.append({"shape": shape, "kind": kind, "exact": values[0] == values[1] == values[2], "selected": values[0][1]})
        checks["all32_original_priority_cases_both_exact"] = len(priority_rows) == 32 and all(row["exact"] for row in priority_rows)
        edge_bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
            0x3F7FFFFF, 0x3F800000, 0x3F800001, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
            0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00001], np.uint32).view(np.float32)
        edges = []
        for shape in ((7, 11), (32, 48), (620, 812)):
            response = np.resize(edge_bits, shape).copy()
            values = [direct(owner, response, response.copy()) for owner in owners.values()]
            edges.append({"shape": shape, "all_float32_boundaries_exact": values[0] == values[1] == values[2]})
        checks["three_bit_boundary_inf_NaN_signed_zero_tie_cases_exact"] = all(row["all_float32_boundaries_exact"] for row in edges)
        private_rng = np.random.default_rng(16002)
        yy, xx = np.indices((47, 61))
        fixtures = [np.zeros((1, 1), np.uint8), np.zeros((47, 61), np.uint8), np.full((47, 61), 255, np.uint8),
            ((xx + yy) % 2 * 255).astype(np.uint8), (xx * 7 % 256).astype(np.uint8),
            ((xx * xx + yy * yy) % 256).astype(np.uint8), private_rng.integers(0, 256, (47, 61), dtype=np.uint8),
            private_rng.integers(0, 256, (620, 812), dtype=np.uint8)]
        checks["all8_generic_both_original_current_and_OpenCV_exact"] = all(
            len(set(identity(owner.corners(image)) for owner in owners.values())) == 1
            and identity(owners["native"].corners(image)) == identity(ORIGINAL_CORNER(image, **PARAMETERS)) for image in fixtures)
        fallback = private_rng.random((32, 48), dtype=np.float32)
        checks["float32_original_API_route_exact"] = len(set(identity(owner.goodFeaturesToTrack(fallback, **PARAMETERS)) for owner in owners.values())) == 1
        checks["changed_parameter_original_API_route_exact"] = len(set(identity(owner.goodFeaturesToTrack(fixtures[-1], maxCorners=17, qualityLevel=.03, minDistance=4, blockSize=3)) for owner in owners.values())) == 1
        errors = []
        for owner in owners.values():
            try:
                owner.goodFeaturesToTrack(np.zeros((8, 8), np.uint16), **PARAMETERS)
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
        checks["original_unsupported_dtype_error_propagates"] = len(errors) == 3 and len(set(errors)) == 1
        raw_stat = baseline.RAW.stat()
        raw_identity = (raw_stat.st_size, raw_stat.st_mtime_ns)
        index = McapIndex(baseline.RAW)
        if index.message_count(baseline.CAMERA_TOPIC) != 7338:
            raise ValueError("The original RAW denominator changed")
        frozen = json.loads((OUT.parent / "bg001/s0/CONTRACT.json").read_text(encoding="utf-8"))["selected_ids"]
        selected = [frozen[position] for position in np.linspace(0, len(frozen) - 1, 12, dtype=int)]
        images = []
        for number, message in index.iter_selected_messages(baseline.CAMERA_TOPIC, selected):
            stamp, optical, raw = decode_bayer(message["payload"])
            gray = cv2.cvtColor(demosaic_rggb8_to_bgr(raw), cv2.COLOR_BGR2GRAY)
            image = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            saved = image.tobytes()
            values = [identity(owner.corners(image)) for owner in owners.values()] + [identity(ORIGINAL_CORNER(image, **PARAMETERS))]
            components.append({"frame": number, "header_ns": int(stamp), "optical_frame": optical,
                "all_three_current_and_original_coords_exact": len(set(values)) == 1, "image_unchanged": image.tobytes() == saved})
            images.append(image)
        checks["all12_original_frozen_RAW_full_corner_outputs_exact"] = len(images) == 12 and all(row["all_three_current_and_original_coords_exact"] and row["image_unchanged"] for row in components)
        pairs = []
        orders = (("native", "packed", "radix"), ("radix", "packed", "native"), ("native", "radix", "packed"))
        for order in orders:
            row, values = {}, {}
            for mode in order:
                begin = time.perf_counter()
                result = [owners[mode].corners(image) for image in images]
                row[mode] = time.perf_counter() - begin
                values[mode] = [identity(value) for value in result]
            if values["native"] != values["packed"] or values["native"] != values["radix"]:
                raise AssertionError("A measured complete corner call changed its original output")
            pairs.append(row)
        medians = {mode: statistics.median(row[mode] for row in pairs) for mode in owners}
        reductions = {mode: 1 - medians[mode] / medians["native"] for mode in ("packed", "radix")}
        eligible = [mode for mode in reductions if reductions[mode] >= .05 and all(row[mode] / row["native"] <= 1.1 for row in pairs)]
        chosen = min(eligible, key=lambda mode: (medians[mode], mode)) if eligible else None
        stat = baseline.RAW.stat()
        checks["RAW_size_mtime_preserved_not_whole_RAW_SHA_claim"] = (stat.st_size, stat.st_mtime_ns) == raw_identity
        for mode in ("packed", "radix"):
            one, two = owners[mode].corners(fixtures[-1]), owners[mode].corners(fixtures[-1])
            saved = identity(two)
            one[0] = -1
            checks[mode + "_output_independent_mutable_storage"] = identity(two) == saved and not np.shares_memory(one, two)
        checks["global_Python_NumPy_RNG_preserved"] = rng == (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))
        checks["all_executed_original_and_private_sources_unchanged"] = sources == [data_identity(path) for path in paths]
    finally:
        for owner in owners.values():
            owner.close()
        cv2.setNumThreads(old_threads)
    checks["original_OpenCV_API_and_thread_count_restored"] = cv2.goodFeaturesToTrack is original_api and cv2.getNumThreads() == old_threads
    checks["all_three_response_and_both_priority_DLLs_closed"] = all(owner.receipt()["closed"] for owner in owners.values()) and all(owners[mode].order_dll is None for mode in ("packed", "radix"))
    checks["registered_output_and_disk_limits_kept"] = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 1024**3 and shutil.disk_usage(output).free >= 20 * 1024**3
    status = "generic_passed_component_screen_chosen128_pending0" if chosen and all(checks.values()) else "closed_negative_component_screen0" if all(checks.values()) else "failed_verification"
    atomic_json(output / "RESULT.json", {"status": status, "checks": checks, "priority_cases": priority_rows,
        "float32_boundary_cases": edges, "actual_RAW_components": components, "component_pairs": pairs, "component_orders": orders,
        "component_medians": medians, "component_reductions_not_full_acceptance": reductions, "eligible_modes": eligible,
        "chosen_mode_by_registered_single_selection": chosen, "sources": sources, "library": data_identity(library),
        "actual_seconds_before_publication": time.perf_counter() - started, "new_optimization_count": 0,
        "complete128_full7338_public_fault_cancel_recovery_training_production": "not_run"})
    print(json.dumps({"status": status, "checks": checks, "component_medians": medians, "component_reductions": reductions, "chosen_mode": chosen}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original exact-key contract failed")


if __name__ == "__main__":
    main()
