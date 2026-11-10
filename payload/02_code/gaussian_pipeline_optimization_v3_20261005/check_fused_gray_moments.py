"""Check exact complete original quality and integer gray moments before timing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import inspect
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from quality_engine import compact_quality
import quality_histogram as histogram
from fused_gray_moments import NativeGrayMoments, fused_quality_execution
from native_quality_workspace import native_quality_workspace_execution


def outcome(function, image):
    try:
        return "value", function(image)
    except cv2.error as error:
        return "cv2_error", (error.code, error.func, error.err)
    except (TypeError, ValueError) as error:
        return type(error).__name__, str(error)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gm001")
    output.mkdir(parents=True, exist_ok=False)
    args.library = args.library.resolve(strict=True)
    paths = [Path(__file__), Path(__file__).with_name("fused_gray_moments.py"), Path(__file__).with_name("native_gray_moments.cpp"),
             Path(__file__).with_name("native_quality_workspace.py"), Path(histogram.__file__), V1 / "quality_engine.py", args.library]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths[:-1]):
        (snapshot / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    atomic_json(output / "CONTRACT.json", {"sources": sources, "integer_moments_and_full_quality_must_be_exact": True,
        "minimum_complete_pilot_reduction_after_generic": .05, "worker_count": 4, "counted": False})
    checks, rows, integer_rows, errors = {}, [], [], []
    old_threads = cv2.getNumThreads()
    original_builder = histogram.make_quality
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261007)
    engine = NativeGrayMoments(args.library)
    try:
        grays = [np.full((31, 43), value, np.uint8) for value in (0, 5, 127, 250, 255)]
        for shape in ((1, 1), (1, 7), (7, 1), (2, 2), (3, 3), (31, 43), (128, 97), (577, 1009)):
            grays.append(rng.integers(0, 256, shape, np.uint8))
            grays.append(((np.indices(shape).sum(axis=0) % 2) * 255).astype(np.uint8))
        for index, gray in enumerate(grays):
            counts, total, squares = engine.moments(gray)
            derivative = cv2.Laplacian(gray, cv2.CV_16S).astype(np.int64)
            expected = np.bincount(gray.ravel(), minlength=256).astype(np.uint64)
            _, deviation = cv2.meanStdDev(derivative.astype(np.int16))
            expected_sharpness = float(deviation[0, 0] ** 2)
            sharpness, brightness, clipped = engine.quality_statistics(gray)
            integer_rows.append({"index": index, "shape": list(gray.shape), "histogram_exact": bool(np.array_equal(counts, expected)),
                "sum_exact": total == int(derivative.sum(dtype=np.int64)), "squared_sum_exact": squares == int((derivative * derivative).sum(dtype=np.int64)),
                "sharpness_bits_exact": np.float64(sharpness).tobytes() == np.float64(expected_sharpness).tobytes(),
                "sharpness": sharpness, "reference_sharpness": expected_sharpness,
                "brightness_and_clipped_exact": (brightness, clipped) == histogram.gray_moments(gray)})
        checks["all21_integer_moments_histogram_and_three_statistics_exact"] = len(integer_rows) == 21 and all(
            all(row[name] for name in ("histogram_exact", "sum_exact", "squared_sum_exact", "sharpness_bits_exact", "brightness_and_clipped_exact")) for row in integer_rows)
    finally:
        engine.close()
    checks["direct_DLL_closed_unloaded_and_outputs_cleared"] = engine.receipt()["DLL_unloaded"] and engine.receipt()["owned_output_bytes_after_close"] == 0
    try:
        with native_quality_workspace_execution(True):
            reference, _ = histogram.make_quality(compact_quality)
            with fused_quality_execution(True, args.library):
                raise AssertionError("A nested private workspace must not silently replace its already modified source")
    except (ValueError, OSError) as error:
        checks["nested_foreign_source_ownership_refused"] = "source" in str(error) or "anchors" in str(error)
    with native_quality_workspace_execution(True):
        reference, _ = histogram.make_quality(compact_quality)
        images = [np.full((32, 48, 3), value, np.uint8) for value in (0, 1, 5, 6, 127, 249, 250, 254, 255)]
        for shape in ((1, 1, 3), (1, 7, 3), (7, 1, 3), (2, 2, 3), (31, 43, 3), (80, 64, 3), (577, 1009, 3)):
            image = rng.integers(0, 256, shape, np.uint8)
            images.extend((image, np.asfortranarray(image), image[:, ::-1], image.copy()))
        expected = [outcome(reference, image) for image in images]
    try:
        with fused_quality_execution(True, args.library) as single:
            candidate, candidate_trace = histogram.make_quality(compact_quality)
            for index, image in enumerate(images):
                actual = outcome(candidate, image)
                rows.append({"index": index, "shape": list(image.shape), "exact": actual == expected[index],
                             "reference": expected[index], "candidate": actual})
            unsupported = []
            reference, _ = original_builder(compact_quality)
            for image in (np.zeros((0, 3, 3), np.uint8), np.zeros((8, 8, 3), np.uint16), np.zeros((8, 8, 3), np.float32),
                          np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8), None):
                before = candidate_trace["native_calls"]
                unsupported.append(outcome(reference, image) == outcome(candidate, image) and candidate_trace["native_calls"] == before + 1)
        checks["all37_original_complete_quality_or_errors_exact"] = len(rows) == 37 and all(row["exact"] for row in rows)
        checks["six_unsupported_before_work_original_results_preserved"] = len(unsupported) == 6 and all(unsupported)
        checks["single_builder_DLL_and_workspace_closed"] = single["fused_builder_restored"] and single["engine"]["DLL_unloaded"] and single["engine"]["owned_output_bytes_after_close"] == 0 and single["workspace"]["workspace"]["closed"] and single["workspace"]["workspace"]["owned_plane_bytes_after_close"] == 0
        with fused_quality_execution(True, args.library) as threaded:
            candidate, _ = histogram.make_quality(compact_quality)
            implementation = inspect.getclosurevars(candidate).nonlocals["implementation"]
            owner = implementation.__globals__["_native_quality_workspace"]
            barrier = threading.Barrier(4)

            def thread(index):
                barrier.wait(timeout=10)
                local_rng = np.random.default_rng(3921 + index)
                reference, _ = original_builder(compact_quality)
                equal = True
                for repeat in range(6):
                    image = local_rng.integers(0, 256, (61 + index, 83 + index, 3), np.uint8)
                    equal &= outcome(reference, image) == outcome(candidate, image)
                state = owner.local.state
                return {"thread": threading.get_ident(), "exact": bool(equal),
                        "pointers": [int(state[name].ctypes.data) for name in ("gray", "reduced")]}

            with ThreadPoolExecutor(max_workers=4) as pool:
                thread_rows = list(pool.map(thread, range(4)))
        checks["four_thread_24_complete_quality_and_eight_planes_exact_disjoint"] = len({row["thread"] for row in thread_rows}) == 4 and all(row["exact"] for row in thread_rows) and len({p for row in thread_rows for p in row["pointers"]}) == 8
        checks["threaded_DLL_unloaded_after_original_worker_drain"] = threaded["engine"]["DLL_unloaded"] and threaded["engine"]["active_calls"] == 0 and threaded["workspace"]["workspace"]["closed"]
        for kind in (RuntimeError, KeyboardInterrupt, SystemExit):
            try:
                with fused_quality_execution(True, args.library) as trace:
                    with histogram.histogram_quality_execution(True):
                        import quality_engine
                        quality_engine.compact_quality(np.full((64, 80, 3), 127, np.uint8))
                        raise kind("injected_fused_quality_consumer_failure")
            except kind as error:
                errors.append({"type": kind.__name__, "exact_message": str(error) == "injected_fused_quality_consumer_failure", "trace": trace})
        checks["three_original_consumer_errors_all_aliases_DLL_and_planes_restored"] = len(errors) == 3 and all(row["exact_message"] and row["trace"]["fused_builder_restored"] and row["trace"]["engine"]["DLL_unloaded"] and row["trace"]["workspace"]["original_builder_restored"] for row in errors)
        refused = []
        for value in (np.zeros((0, 2), np.uint8), np.zeros((2, 2), np.int16), np.zeros((2, 2, 1), np.uint8), np.zeros((2, 2), np.uint8)[:, ::-1], None):
            try:
                engine.moments(value)
            except ValueError:
                refused.append(True)
        try:
            engine.moments(np.zeros((2, 2), np.uint8))
        except RuntimeError as error:
            refused.append("closed" in str(error))
        checks["five_invalid_gray_planes_and_closed_DLL_refused"] = len(refused) == 6 and all(refused)
    finally:
        histogram.make_quality = original_builder
        cv2.setNumThreads(old_threads)
    checks["all_sources_original_builder_and_OpenCV_threads_restored"] = sources == [data_identity(path) for path in paths] and histogram.make_quality is original_builder and cv2.getNumThreads() == old_threads
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    checks["actual_Windows_private_DLL_module_unloaded"] = kernel.GetModuleHandleW(str(args.library)) is None
    result = {"status": "generic_passed_pilot_and_complete_stage_pending" if all(checks.values()) else "failed_verification",
              "checks": checks, "sources": sources, "integer_rows": integer_rows, "rows": rows,
              "thread_rows": thread_rows, "single_trace": single, "threaded_trace": threaded, "error_traces": errors,
              "seconds": time.perf_counter() - started, "accepted_count_contribution": 0, "counted": False}
    atomic_json(output / "RESULT.json", result)
    print({"status": result["status"], "checks": checks, "accepted_count_contribution": 0}, flush=True)
    if not all(checks.values()):
        raise ValueError("The original integer moments or full quality contract failed")


if __name__ == "__main__":
    before_threads = cv2.getNumThreads()
    before_builder = histogram.make_quality
    before_main = time.perf_counter()
    try:
        main()
    except BaseException as error:
        if "--output" in sys.argv:
            folder = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
            folder.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gm001")
            if folder.is_dir():
                atomic_json(folder / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
                    "seconds": time.perf_counter() - before_main, "accepted_count_contribution": 0})
        raise
    finally:
        histogram.make_quality = before_builder
        cv2.setNumThreads(before_threads)
