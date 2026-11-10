"""Verify exact RGGB fusion on all gray coefficients and generic image contracts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import quality_engine
import quality_histogram
from native_quality_workspace import NativeQualityWorkspace
from exact_bayer_gray import BayerGrayEngine, BayerGrayWorkspace, raw_gray_quality_execution, compile_quality


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from color_contract import demosaic_rggb8_to_bgr
    paths = [Path(__file__), Path(__file__).with_name("exact_bayer_gray.py"),
             Path(__file__).with_name("native_bayer_gray.cpp"), Path(__file__).with_name("native_quality_workspace.py"),
             Path(quality_histogram.__file__), V1 / "quality_engine.py", baseline.TOOLS / "color_contract.py", args.library.resolve(strict=True)]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths[:-1]):
        (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time(), "scope": "Windows"})
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    engine = BayerGrayEngine(args.library)
    workspace, reference_workspace = BayerGrayWorkspace(engine), NativeQualityWorkspace()
    checks, cases, state = {}, [], {"status": "running", "accepted_count_contribution": 0}
    try:
        if not engine.available:
            raise RuntimeError("The existing machine cannot execute AVX2")
        triplets = 0
        for first in range(0, 2**24, 2**20):
            indices = np.arange(first, first+2**20, dtype=np.uint32)
            blue, green, red = indices & 255, (indices >> 8) & 255, indices >> 16
            bgr = np.stack((blue, green, red), axis=1).astype(np.uint8).reshape(1024, 1024, 3)
            actual = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            expected = ((blue*3735+green*19235+red*9798+16384) >> 15).astype(np.uint8).reshape(1024, 1024)
            if not np.array_equal(actual, expected):
                raise ValueError("An actual native BGR gray triplet differs from the registered integer coefficients")
            triplets += 2**20
        checks["all_16777216_actual_native_BGR_gray_triplets_exact"] = triplets == 2**24
        rng = np.random.default_rng(1471)
        reference_quality = compile_quality(quality_engine.compact_quality, reference_workspace, quality_histogram)
        fused_quality = compile_quality(quality_engine.compact_quality, workspace, quality_histogram)
        shapes = [(1, 1), (2, 23), (23, 2), (3, 3), (4, 5), (5, 4), (7, 21), (21, 7),
                  (19, 20), (20, 19), (31, 32), (32, 31), (37, 49), (47, 53), (64, 65), (65, 64)]
        for position in range(64):
            shape = shapes[position % len(shapes)]
            padded = np.zeros((shape[0], shape[1]+3), dtype=np.uint8)
            raw = padded[:, :shape[1]]
            if position < 16:
                raw[:] = rng.integers(0, 256, shape, dtype=np.uint8)
            elif position < 32:
                raw[:] = 0 if position % 2 else 255
            elif position < 48:
                for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    raw[y::2, x::2] = rng.integers(0, 256)
            else:
                raw[:] = ((np.indices(shape).sum(axis=0) % 3)*127).astype(np.uint8)
            immutable = padded.copy()
            actual = workspace.gray(raw).copy()
            bgr = demosaic_rggb8_to_bgr(raw)
            reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            exact = np.array_equal(actual, reference)
            quality_results = []
            for scorer, value in ((fused_quality, raw), (reference_quality, bgr)):
                try:
                    quality_results.append(("value", scorer(value)))
                except cv2.error as error:
                    quality_results.append(("error", str(error)))
            quality_exact = quality_results[0] == quality_results[1]
            row = {"case": position, "shape": list(shape), "gray_exact": exact, "quality_exact": quality_exact,
                   "quality_result_kind": quality_results[0][0],
                   "input_unchanged": np.array_equal(padded, immutable)}
            cases.append(row)
            if not all(row[name] for name in ("gray_exact", "quality_exact", "input_unchanged")):
                np.savez(output / "FIRST_FAILURE.npz", raw=raw, candidate=actual, original=reference)
                raise ValueError("A generic exact gray or original complete quality field changed")
        for shape in ((1240, 1624), (1241, 1625)):
            raw = rng.integers(0, 256, shape, dtype=np.uint8)
            bgr = demosaic_rggb8_to_bgr(raw)
            if not np.array_equal(workspace.gray(raw), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
                raise ValueError("A complete generic RAW-size SIMD or edge pixel changed")
            if fused_quality(raw) != reference_quality(bgr):
                raise ValueError("A complete generic RAW-size quality field changed")
        checks["all64_small_and_two_full_size_generic_gray_quality_and_input_exact"] = len(cases) == 64
        with raw_gray_quality_execution(args.library, demosaic_rggb8_to_bgr) as execution:
            quality, trace = quality_histogram.make_quality(quality_engine.compact_quality)
            raw = rng.integers(0, 256, (64, 65), dtype=np.uint8)
            for unsupported in (raw[::-1], raw[:, ::2], raw.T):
                if quality(unsupported) != reference_quality(demosaic_rggb8_to_bgr(unsupported)):
                    raise ValueError("An unsupported raw layout native fallback changed")
            expected = None
            try:
                demosaic_rggb8_to_bgr(raw.astype(np.float32))
            except ValueError as error:
                expected = str(error)
            try:
                quality(raw.astype(np.float32))
            except ValueError as error:
                if str(error) != expected:
                    raise ValueError("The original raw dtype error changed")
            else:
                raise ValueError("An invalid raw dtype did not preserve the original error")
            checks["unsupported_layouts_and_original_raw_dtype_error_preserved"] = trace["fallback_calls"] == 4
        checks["fallback_lifecycle_builder_planes_DLL_closed"] = execution["builder_restored"] and execution["engine"][
            "DLL_unloaded"] and execution["fused_workspace"]["owned_plane_bytes_after_close"] == execution[
                "fallback_workspace"]["owned_plane_bytes_after_close"] == 0
        try:
            quality(raw)
        except RuntimeError:
            checks["closed_quality_cannot_execute"] = True
        else:
            raise ValueError("A closed exact quality escaped its owned lifecycle")

        def concurrent(seed):
            random = np.random.default_rng(seed)
            for _ in range(8):
                value = random.integers(0, 256, (61, 67), dtype=np.uint8)
                if not np.array_equal(workspace.gray(value), cv2.cvtColor(demosaic_rggb8_to_bgr(value), cv2.COLOR_BGR2GRAY)):
                    raise ValueError("A private fused plane crossed concurrent threads")
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            checks["four_thread_private_planes_all32_outputs_exact_and_pool_drained"] = all(pool.map(concurrent, range(4)))
        checks["all_original_sources_unchanged"] = sources == [data_identity(path) for path in paths]
        if not all(checks.values()):
            raise ValueError("An exact Bayer gray generic contract failed")
        workspace.close()
        reference_workspace.close()
        engine.close()
        checks["own_final_planes_and_DLL_closed"] = workspace.receipt()["owned_plane_bytes_after_close"] == reference_workspace.receipt()[
            "owned_plane_bytes_after_close"] == 0 and engine.receipt()["DLL_unloaded"] and engine.receipt()["active_calls"] == 0
        state.update(status="generic_passed_component_and_complete_stage_pending", sources=sources, checks=checks,
            cases=cases, coefficient_triplets=triplets, workspace=workspace.receipt(), engine=engine.receipt(),
            node_seconds=time.perf_counter()-started, current_RAW128_full7338_and_formal_not_run=True)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(checks), "seconds": state["node_seconds"], "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", cases=cases, checks=checks, error_type=type(error).__name__, error=str(error), seconds=time.perf_counter()-started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        workspace.close()
        reference_workspace.close()
        engine.close()
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"thread_count_restored": cv2.getNumThreads() == old_threads,
            "DLL_closed": engine.closed, "all_owned_planes_closed": workspace.closed and reference_workspace.closed,
            "node_seconds": time.perf_counter()-started, "counted": False})


if __name__ == "__main__":
    main()
