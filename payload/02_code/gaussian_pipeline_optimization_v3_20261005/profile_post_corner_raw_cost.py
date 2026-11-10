"""Measure current RAW quality and durable-cache costs without changing scores."""

import argparse
import importlib
import os
from pathlib import Path
import threading
import time

import cv2
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
from native_quality_workspace import NativeQualityWorkspace
from planar_corner import PlanarCornerWorkspace, planar_corner_execution

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OUT / "post_corner_cost_001")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    import sys
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    atomic_json = reference.atomic_json
    contract = reference.load(OUT / "bg001/s0/CONTRACT.json")
    expected = reference.load(OUT / "qc001/s0/t0/planar/SCORES.json")["rows"]
    cache_type = reference.quality_engine.ScoreCache
    methods = {"exact_gray": (exact_bayer_gray.BayerGrayWorkspace, "gray"),
               "Laplacian": (NativeQualityWorkspace, "derivative"),
               "meanStdDev": (cv2, "meanStdDev"),
               "gray_histogram_moments": (reference.quality_histogram, "gray_moments"),
               "resize": (NativeQualityWorkspace, "reduced"),
               "planar_corner": (PlanarCornerWorkspace, "corners"),
               "cache_key": (cache_type, "key"), "cache_read": (cache_type, "read"),
               "cache_write": (cache_type, "write"), "fsync": (os, "fsync")}
    originals = {name: getattr(owner, attribute) for name, (owner, attribute) in methods.items()}
    factory = exact_bayer_gray.compile_quality
    histogram_factory = reference.quality_histogram.make_quality
    old_threads = cv2.getNumThreads()
    lock = threading.Lock()
    totals = {}
    sources_paths = [Path(__file__), Path(__file__).with_name("planar_corner.py"),
                    Path(__file__).with_name("exact_bayer_gray.py"), Path(__file__).with_name("native_quality_workspace.py"),
                    Path(reference.quality_histogram.__file__), reference.V1 / "quality_engine.py",
                    reference.V2 / "single_copy_raw_reader.py", Path(baseline.__file__),
                    baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", baseline.TOOLS / "mcap_audit.py",
                    OUT / "qc001/b0/native_planar_corner.dll", OUT / "bg001/b0/native_bayer_gray.dll"]
    sources = [reference.data_identity(path) for path in sources_paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(sources_paths[:-2]):
        (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    raw_stat = baseline.RAW.stat()
    index = McapIndex(baseline.RAW)
    if index.message_count(baseline.CAMERA_TOPIC) != 7338 or len(contract["selected_ids"]) != 128:
        raise ValueError("The fixed original RAW denominator changed")
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    state = {"status": "running", "accepted_count_contribution": 0, "phases": {}}

    def wrap(name):
        original = originals[name]

        def measured(*parameters, **keywords):
            before = time.perf_counter()
            try:
                return original(*parameters, **keywords)
            finally:
                elapsed = time.perf_counter() - before
                with lock:
                    totals[name]["calls"] += 1
                    totals[name]["worker_seconds_sum"] += elapsed

        return measured

    try:
        cv2.setNumThreads(1)
        for phase in ("cold", "warm"):
            totals = {name: {"calls": 0, "worker_seconds_sum": 0.0} for name in methods}
            for name, (owner, attribute) in methods.items():
                setattr(owner, attribute, wrap(name))
            before = time.perf_counter()
            try:
                with planar_corner_execution(sources_paths[-2]) as corner_trace:
                    with exact_bayer_gray.raw_gray_quality_execution(sources_paths[-1], demosaic_rggb8_to_bgr) as gray_trace:
                        with reference.quality_histogram.histogram_quality_execution(True) as histogram_trace:
                            cache = cache_type(output / "quality", contract["cache_contract"])
                            rows, detail = reference.quality_engine.score_messages(
                                reference.iter_single_copy_messages(index, baseline.CAMERA_TOPIC, contract["selected_ids"]),
                                decode_bayer, lambda raw: raw, reference.quality_engine.compact_quality,
                                workers=4, capacity=16, cache=cache, maximum_seconds=60)
            finally:
                for name, (owner, attribute) in methods.items():
                    setattr(owner, attribute, originals[name])
            whole = time.perf_counter() - before
            fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
            actual = [{name: row[name] for name in fields} for row in rows]
            wanted = [dict(row, cache_hit=phase == "warm") for row in expected]
            counts_ok = all(value["calls"] == (128 if phase == "cold" or name in ("cache_key", "cache_read") else 0)
                            for name, value in totals.items())
            checks = {"original_metadata_six_fields_order_exact": actual == wanted,
                      "all_expected_operator_cache_and_fsync_call_counts": counts_ok,
                      "original_cold_or_warm_cache_hits": detail["cache_hits"] == (128 if phase == "warm" else 0),
                      "all128_submitted": detail["submitted"] == 128,
                      "all_owned_aliases_restored_and_DLLs_unloaded": corner_trace["original_factory_restored"]
                          and corner_trace["original_cv2_corner_unchanged"] and corner_trace["workspace"]["DLL_unloaded"]
                          and gray_trace["builder_restored"] and gray_trace["engine"]["DLL_unloaded"]
                          and histogram_trace["aliases_restored"]}
            state["phases"][phase] = {"complete_phase_seconds": whole, "worker_costs": totals, "detail": detail,
                                      "checks": checks, "gray_execution": gray_trace, "corner_execution": corner_trace}
            atomic_json(output / (phase.upper() + "_SCORES.json"), {"rows": actual, "counted": False})
            if not all(checks.values()):
                raise ValueError("A current exact RAW cost diagnostic consumer failed")
            print({"phase": phase, "seconds": whole, "costs": totals}, flush=True)
        cache_reference = OUT / "qc001/s0/t0/planar/quality"
        names = sorted(path.name for path in cache_reference.iterdir())
        checks = {"all128_cold_cache_bytes_exact": len(names) == 128
                      and names == sorted(path.name for path in (output / "quality").iterdir())
                      and all((cache_reference / name).read_bytes() == (output / "quality" / name).read_bytes() for name in names),
                  "all_original_scientific_sources_unchanged": sources == [reference.data_identity(path) for path in sources_paths],
                  "original_RAW_size_mtime_unchanged": (raw_stat.st_size, raw_stat.st_mtime_ns)
                      == (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns),
                  "all_timed_original_functions_and_compilers_restored":
                      all(getattr(owner, attribute) is originals[name] for name, (owner, attribute) in methods.items())
                      and exact_bayer_gray.compile_quality is factory
                      and reference.quality_histogram.make_quality is histogram_factory}
        if not all(checks.values()):
            raise ValueError("An original RAW cache or ownership contract failed")
        state.update(status="current_complete_RAW128_cold_warm_cost_diagnostic_finished", checks=checks,
                     sources=sources, worker_sums_not_wall_share=True, instrumentation_in_wall_included=True,
                     cache_write_contains_fsync_not_additive=True, node_seconds=time.perf_counter() - started,
                     full7338_new_algorithm_formal_training_GT_and_production_not_run=True)
        atomic_json(output / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        for name, (owner, attribute) in methods.items():
            setattr(owner, attribute, originals[name])
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"original_threads_restored": cv2.getNumThreads() == old_threads,
            "original_functions_restored": all(getattr(owner, attribute) is originals[name]
                for name, (owner, attribute) in methods.items()), "original_compilers_restored":
                exact_bayer_gray.compile_quality is factory and reference.quality_histogram.make_quality is histogram_factory,
            "owned_pools_drained": True, "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
