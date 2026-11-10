"""Check exact integers, float bits, complete scores and owned native lifetimes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
from native_quality_workspace import NativeQualityWorkspace
import vector_gray_stats as vector

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def original_statistics(gray):
    _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
    brightness, clipped = reference.quality_histogram.gray_moments(gray)
    return float(deviation[0, 0] ** 2), brightness, clipped


def bits(values):
    return np.array(values, dtype=np.float64).view(np.uint64).tolist()


def same_quality(left, right):
    return left == right and bits(list(left.values())) == bits(list(right.values()))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT / "vector_gray_stats_001")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    import sys
    sys.path.insert(0, str(baseline.TOOLS))
    from camera_motion import decode_bayer
    from mcap_audit import McapIndex
    from color_contract import demosaic_rggb8_to_bgr
    paths = [Path(__file__), Path(vector.__file__), Path(__file__).with_name("native_vector_gray_stats.cpp"),
             Path(exact_bayer_gray.__file__), Path(__file__).with_name("native_quality_workspace.py"),
             Path(reference.quality_histogram.__file__), reference.V1 / "quality_engine.py",
             reference.V2 / "single_copy_raw_reader.py", Path(baseline.__file__), baseline.TOOLS / "camera_motion.py",
             baseline.TOOLS / "color_contract.py", baseline.TOOLS / "mcap_audit.py", args.library.resolve(strict=True),
             OUT / "bg001/b0/native_bayer_gray.dll"]
    sources = [reference.data_identity(path) for path in paths]
    build = reference.load(args.library.parent / "RESULT.json")
    if build["status"] != "built_numerical_verification_pending" or build["DLL"] != sources[-2]:
        raise ValueError("The executed private vector build identity changed")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths[:-2]):
        (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    reference.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    raw_stat = baseline.RAW.stat()
    old_threads = cv2.getNumThreads()
    old_optimized = cv2.useOptimized()
    original_compiler = exact_bayer_gray.compile_quality
    rng = np.random.default_rng(20261007)
    engine = vector.VectorGrayStats(args.library)
    workspace = NativeQualityWorkspace()
    gray_engine = None
    raw_workspace = None
    result = {"status": "running", "accepted_count_contribution": 0}
    checks = {}
    try:
        cv2.setNumThreads(1)
        if not engine.available or cv2.__version__ != "4.13.0" or not old_optimized:
            raise ValueError("The preregistered existing native consumer is unavailable")
        stat_cases = []
        shapes = [(1, 1), (1, 31), (31, 1), (2, 2), (3, 3), (3, 17), (17, 3), (17, 18),
                  (18, 17), (33, 31), (64, 65), (257, 259), (4096, 4096)]
        for shape in shapes:
            stat_cases.append(rng.integers(0, 256, shape, dtype=np.uint8))
        for value in (0, 5, 6, 249, 250, 255):
            stat_cases.append(np.full((31, 33), value, np.uint8))
        stat_cases.append(((np.indices((65, 67)).sum(axis=0) % 2) * 255).astype(np.uint8))
        stat_cases.append(np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
        stat_records = []
        for gray in stat_cases:
            immutable = gray.copy()
            derivative = cv2.Laplacian(gray, cv2.CV_16S).astype(np.int64)
            wanted = (int(gray.sum(dtype=np.uint64)), int(np.count_nonzero((gray <= 5) | (gray >= 250))),
                      int(derivative.sum(dtype=np.int64)), int((derivative * derivative).sum(dtype=np.int64)))
            actual = engine.moments(gray)
            exact = actual == wanted and bits(engine.quality_statistics(gray)) == bits(original_statistics(gray))
            if not exact or not np.array_equal(gray, immutable):
                raise ValueError("An original complete gray integer or float-bit result changed")
            stat_records.append({"shape": list(gray.shape), "integers": actual, "all_three_float_bits_exact": exact})
        del stat_cases, derivative, gray, immutable
        checks["all21_integer_and_three_float_bits_extreme_border_tail_max_budget_cases"] = len(stat_records) == 21
        native = reference.quality_engine.compact_quality
        fallback = exact_bayer_gray.compile_quality(native, workspace, reference.quality_histogram)
        candidate = vector.compile_vector_quality(native, workspace, reference.quality_histogram, engine, fallback)
        generic = []
        for number in range(48):
            height = (2, 3, 7, 17, 31, 33, 65, 97, 129, 257, 301, 479)[number % 12]
            width = (3, 5, 9, 18, 32, 34, 67, 101, 131, 259, 307, 641)[number % 12]
            if number < 16:
                image = np.full((height, width, 3), (number * 17) % 256, np.uint8)
            elif number < 32:
                image = np.repeat((((np.indices((height, width)).sum(axis=0) % 2) * 255).astype(np.uint8))[..., None], 3, axis=2)
            else:
                image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            expected = native(image)
            actual = candidate(image)
            if not same_quality(actual, expected):
                raise ValueError("A generic original six-field quality result changed")
            generic.append({"case": number, "shape": list(image.shape), "quality": actual})
        checks["all48_generic_full_six_fields_and_float_bits_exact"] = len(generic) == 48
        raw_contract = reference.load(OUT / "bg001/s0/CONTRACT.json")
        selected = [raw_contract["selected_ids"][position] for position in np.linspace(0, 127, 32, dtype=int)]
        gray_engine = exact_bayer_gray.BayerGrayEngine(paths[-1])
        raw_workspace = exact_bayer_gray.BayerGrayWorkspace(gray_engine)
        raw_fallback = exact_bayer_gray.compile_quality(native, raw_workspace, reference.quality_histogram)
        raw_quality = vector.compile_vector_quality(native, raw_workspace, reference.quality_histogram, engine, raw_fallback)
        raw_records = []
        iterator = reference.iter_single_copy_messages(McapIndex(baseline.RAW), baseline.CAMERA_TOPIC, selected)
        try:
            for frame, message in iterator:
                _, _, raw = decode_bayer(message["payload"])
                expected = native(demosaic_rggb8_to_bgr(raw))
                actual = raw_quality(raw)
                if not same_quality(actual, expected):
                    raise ValueError("A fixed real RAW original six-field result changed")
                raw_records.append({"camera_frame": frame, "quality": actual})
        finally:
            iterator.close()
        checks["all32_fixed_actual_RAW_original_six_fields_and_float_bits_exact"] = len(raw_records) == 32
        base = rng.integers(0, 256, (33, 65), dtype=np.uint8)
        fallback_checks = []
        for gray in (base[:, ::-1], base[::2], base.T):
            fallback_checks.append(bits(engine.quality_statistics(gray)) == bits(original_statistics(gray)))
        cv2.setUseOptimized(False)
        fallback_checks.append(bits(engine.quality_statistics(base)) == bits(original_statistics(base)))
        cv2.setUseOptimized(old_optimized)
        checks["four_actual_noncontiguous_or_unoptimized_stat_fallbacks_exact"] = all(fallback_checks)
        invalid_inputs = [np.zeros((7, 9, 3), np.float32), np.zeros((7, 9, 3), np.float64),
                          np.zeros((0, 9, 3), np.uint8), None]
        errors = []
        for image in invalid_inputs:
            outcomes = []
            for function in (fallback, candidate):
                try:
                    outcomes.append(("result", function(image)))
                except (cv2.error, ValueError, TypeError, AttributeError) as error:
                    outcomes.append((type(error).__name__, str(error)))
            errors.append(outcomes[0] == outcomes[1])
        checks["four_original_invalid_full_quality_exceptions_preserved"] = all(errors)
        invalid_native = [engine.function(None, 3, 3, None, 4), engine.function(base.ctypes.data, 0, 65, None, 4)]
        target = np.empty(4, np.int64)
        invalid_native.append(engine.function(base.ctypes.data, *base.shape, target.ctypes.data, 3))
        checks["three_actual_native_invalid_pointer_shape_or_output_count_rejected"] = invalid_native == [1, 1, 1]
        expected_base = bits(original_statistics(base))
        original_function = engine.function
        engine.function = lambda *arguments: 1
        try:
            try:
                engine.quality_statistics(base)
                checks["native_error_propagates_and_active_calls_drains"] = False
            except ValueError as error:
                checks["native_error_propagates_and_active_calls_drains"] = "code 1" in str(error) and engine.active_calls == 0
        finally:
            engine.function = original_function
        old_abi = vector.ABI_VERSION
        vector.ABI_VERSION = 3
        try:
            try:
                wrong = vector.VectorGrayStats(args.library)
                wrong.close()
                checks["actual_wrong_ABI_load_rejected"] = False
            except ValueError as error:
                checks["actual_wrong_ABI_load_rejected"] = "ABI differs" in str(error)
        finally:
            vector.ABI_VERSION = old_abi
        barrier = threading.Barrier(4)
        readonly = [rng.integers(0, 256, (129, 131), dtype=np.uint8) for _ in range(16)]
        originals = [image.copy() for image in readonly]
        for image in readonly:
            image.flags.writeable = False

        def worker(number):
            barrier.wait(timeout=10)
            return [bits(engine.quality_statistics(readonly[index])) for index in range(number, 16, 4)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, number) for number in range(4)]
            returned = [future.result(timeout=30) for future in futures]
        checks["four_real_workers16_readonly_inputs_exact_private_outputs"] = all(
            returned[number] == [bits(original_statistics(readonly[index])) for index in range(number, 16, 4)] for number in range(4))
        outputs = [state["output"] for state in engine.states]
        checks["all_thread_output_memory_private_and_original_inputs_unchanged"] = len(outputs) >= 5 and all(
            not np.shares_memory(left, right) for index, left in enumerate(outputs) for right in outputs[index + 1:]) and all(
                np.array_equal(image, original) for image, original in zip(readonly, originals))
        entered, release = threading.Event(), threading.Event()

        def held_native(*arguments):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("The actual held native caller was not released")
            return original_function(*arguments)

        engine.function = held_native
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine.quality_statistics, base)
            try:
                if not entered.wait(10):
                    raise TimeoutError("The actual native caller did not enter")
                try:
                    engine.close()
                    checks["actual_active_native_caller_prevents_premature_DLL_unload"] = False
                except RuntimeError:
                    checks["actual_active_native_caller_prevents_premature_DLL_unload"] = engine.active_calls == 1 and not engine.closed
            finally:
                release.set()
            if bits(future.result(timeout=20)) != expected_base:
                raise ValueError("The drained original held native call changed")
        engine.function = original_function
        checks["all_original_sources_RAW_identity_and_compiler_unchanged"] = sources == [reference.data_identity(path) for path in paths]
        checks["all_original_sources_RAW_identity_and_compiler_unchanged"] &= (
            raw_stat.st_size, raw_stat.st_mtime_ns) == (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns)
        checks["all_original_sources_RAW_identity_and_compiler_unchanged"] &= exact_bayer_gray.compile_quality is original_compiler
        before_close = engine.receipt()
        workspace.close()
        raw_workspace.close()
        gray_engine.close()
        engine.close()
        try:
            candidate(np.zeros((9, 11, 3), np.uint8))
            checks["closed_full_consumer_rejects_calls"] = False
        except RuntimeError:
            checks["closed_full_consumer_rejects_calls"] = True
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [ctypes.c_wchar_p], ctypes.c_void_p
        checks["actual_DLL_unloaded_no_retained_output_bytes_or_active_calls"] = not kernel.GetModuleHandleW(str(args.library.resolve()))
        checks["actual_DLL_unloaded_no_retained_output_bytes_or_active_calls"] &= (
            engine.receipt()["DLL_unloaded"] and engine.receipt()["retained_output_bytes"] == 0 and engine.active_calls == 0)
        if not all(checks.values()):
            raise ValueError("An original exact vector generic or lifecycle contract failed")
        result.update(status="exact_generic_and32RAW_passed_complete128_pending", checks=checks, sources=sources,
                      integer_stat_cases=stat_records, generic_full_quality=generic, original_RAW32=raw_records,
                      before_close=before_close, engine=engine.receipt(), original_fields=list(reference.quality_engine.FIELDS),
                      node_seconds=time.perf_counter() - started, complete128_full7338_recovery_external_training_not_run=True)
        reference.atomic_json(output / "RESULT.json", result)
        print({"status": result["status"], "checks": checks, "seconds": result["node_seconds"]}, flush=True)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), checks=checks, seconds=time.perf_counter() - started)
        reference.atomic_json(output / "FAILURE.json", result)
        raise
    finally:
        engine.close()
        workspace.close()
        if raw_workspace is not None:
            raw_workspace.close()
        if gray_engine is not None:
            gray_engine.close()
        cv2.setUseOptimized(old_optimized)
        cv2.setNumThreads(old_threads)
        reference.atomic_json(output / "RESOURCE_CLOSE.json", {"original_threads_restored": cv2.getNumThreads() == old_threads,
            "original_optimized_restored": cv2.useOptimized() == old_optimized,
            "original_compiler_unchanged": exact_bayer_gray.compile_quality is original_compiler,
            "engine": engine.receipt(), "counted": False})


if __name__ == "__main__":
    main()
