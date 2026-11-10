"""Measure complete RAW128 scoring with original or owned durable cache writing."""

import argparse
from contextlib import nullcontext
import importlib
from pathlib import Path
import statistics
import sys
import time

import cv2
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
from planar_corner import planar_corner_execution
from vector_gray_stats import vector_gray_stats_execution
import native_quality_cache_writer as native

OUT = reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "library"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(native.BASE)
    generic = reference.load(args.generic / "RESULT.json")
    if generic["status"] != "original_cache_generic_error_concurrency_and_lifecycle_passed_complete128_pending" or not all(generic["checks"].values()):
        raise ValueError("The original cache interface or native lifecycle gate failed")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    corner = OUT / "qc001/b0/native_planar_corner.dll"
    gray = OUT / "bg001/b0/native_bayer_gray.dll"
    vector = OUT / "vector_gray_stats_001/b0/native_vector_gray_stats.dll"
    paths = [Path(__file__), Path(native.__file__), Path(native.__file__).with_suffix(".cpp"),
        Path(exact_bayer_gray.__file__), Path(__file__).with_name("native_quality_workspace.py"),
        Path(__file__).with_name("planar_corner.py"), Path(__file__).with_name("vector_gray_stats.py"),
        Path(reference.quality_histogram.__file__), reference.V1 / "quality_engine.py", reference.V2 / "single_copy_raw_reader.py",
        Path(baseline.__file__), baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py",
        baseline.TOOLS / "mcap_audit.py", args.library.resolve(strict=True), corner, gray, vector]
    sources = [reference.data_identity(path) for path in paths]
    if [sources[index] for index in (1, 2, 8, 14)] != generic["sources"][1:]:
        raise ValueError("The executed generic cache source or native DLL changed")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for number, path in enumerate(paths):
        if path.suffix != ".dll":
            (snapshot / (str(number) + "_" + path.name)).write_bytes(path.read_bytes())
    contract = reference.load(OUT / "bg001/s0/CONTRACT.json")
    expected_folder = OUT / "vector_gray_stats_001/s0/t0/vector"
    expected = reference.load(expected_folder / "SCORES.json")["rows"]
    reference.atomic_json(output / "CONTRACT.json", contract)
    process = psutil.Process()
    reference.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    index, raw_stat = McapIndex(baseline.RAW), baseline.RAW.stat()
    if index.message_count(baseline.CAMERA_TOPIC) != 7338 or len(contract["selected_ids"]) != 128:
        raise ValueError("The registered original RAW or fixed128 denominator changed")
    old_threads, original_write = cv2.getNumThreads(), reference.quality_engine.ScoreCache.write
    original_factory = exact_bayer_gray.compile_quality
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
    state = {"status": "running", "pairs": [], "accepted_count_contribution": 0}
    try:
        cv2.setNumThreads(1)
        for trial, order in enumerate((("current", "native"), ("native", "current"), ("current", "native"))):
            pair, rows_by_mode = {"trial": trial, "order": order, "modes": {}}, {}
            for mode in order:
                folder = output / ("t" + str(trial)) / mode
                folder.mkdir(parents=True, exist_ok=False)
                before = time.perf_counter()
                with (native.native_quality_cache_execution(args.library) if mode == "native" else nullcontext()) as cache_trace:
                    with planar_corner_execution(corner) as corner_trace:
                        with vector_gray_stats_execution(vector) as vector_trace:
                            with exact_bayer_gray.raw_gray_quality_execution(gray, demosaic_rggb8_to_bgr) as gray_trace:
                                with reference.quality_histogram.histogram_quality_execution(True) as histogram_trace:
                                    cache = reference.quality_engine.ScoreCache(folder / "quality", contract["cache_contract"])
                                    rows, detail = reference.quality_engine.score_messages(
                                        reference.iter_single_copy_messages(index, baseline.CAMERA_TOPIC, contract["selected_ids"]),
                                        decode_bayer, lambda raw: raw, reference.quality_engine.compact_quality,
                                        workers=4, capacity=16, cache=cache, maximum_seconds=60)
                whole = time.perf_counter() - before
                rows_by_mode[mode] = [{name: row[name] for name in fields} for row in rows]
                if detail["submitted"] != 128 or detail["cache_hits"] or rows_by_mode[mode] != expected:
                    raise ValueError("The original RAW128 scores, metadata, order or cold cache changed")
                if not (corner_trace["original_factory_restored"] and corner_trace["workspace"]["DLL_unloaded"]
                    and corner_trace["workspace"]["native_calls"] == 128 and vector_trace["original_factory_restored"]
                    and vector_trace["engine"]["DLL_unloaded"] and vector_trace["engine"]["native_calls"] == 128
                    and gray_trace["builder_restored"] and gray_trace["engine"]["DLL_unloaded"] and histogram_trace["aliases_restored"]):
                    raise ValueError("An incumbent RAW128 native consumer or lifecycle changed")
                if mode == "native" and not (cache_trace["original_write_restored"] and cache_trace["engine"]["DLL_unloaded"]
                    and cache_trace["engine"]["native_calls"] == cache_trace["engine"]["actual_flushes"] == cache_trace["engine"]["closed_handles"] == 128
                    and cache_trace["engine"]["create_attempts"] == 128 and not cache_trace["engine"]["fallback_calls"]
                    and not cache_trace["engine"]["retained_output_bytes"] and not cache_trace["engine"]["active_calls"]):
                    raise ValueError("The actual four-worker durable cache denominator or close failed")
                pair["modes"][mode] = {"complete_phase_seconds": whole, "detail": detail, "cache_execution": cache_trace,
                    "corner_execution": corner_trace, "vector_execution": vector_trace, "gray_execution": gray_trace,
                    "histogram": histogram_trace, "peak_RSS_bytes": process.memory_info().peak_wset}
                reference.atomic_json(folder / "SCORES.json", {"rows": rows_by_mode[mode], "counted": False})
                reference.atomic_json(folder / "TRACE.json", pair["modes"][mode])
            before = time.perf_counter()
            directories = [output / ("t" + str(trial)) / mode / "quality" for mode in ("current", "native")]
            directories.append(expected_folder / "quality")
            names = sorted(path.name for path in directories[0].iterdir())
            pair["checks"] = {"all128_original_six_fields_metadata_order_exact": rows_by_mode["current"] == rows_by_mode["native"] == expected,
                "all128_original_cold_cache_bytes_exact_including_frozen_reference": len(names) == 128
                    and all(names == sorted(path.name for path in directory.iterdir()) for directory in directories)
                    and all(len({(directory / name).read_bytes() for directory in directories}) == 1 for name in names),
                "original_cache_write_and_quality_factory_restored": reference.quality_engine.ScoreCache.write is original_write
                    and exact_bayer_gray.compile_quality is original_factory}
            pair["comparison_seconds_charged_each_side"] = time.perf_counter() - before
            pair["phase_seconds"] = {mode: pair["modes"][mode]["complete_phase_seconds"] + pair["comparison_seconds_charged_each_side"] for mode in order}
            state["pairs"].append(pair)
            reference.atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": pair["phase_seconds"], "checks": pair["checks"]}, flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("An original complete paired RAW128 output or cache byte check failed")
        medians = {mode: statistics.median(pair["phase_seconds"][mode] for pair in state["pairs"]) for mode in ("current", "native")}
        reduction = 1 - medians["native"] / medians["current"]
        if sources != [reference.data_identity(path) for path in paths] or (raw_stat.st_size, raw_stat.st_mtime_ns) != (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns):
            raise ValueError("A frozen original cache consumer, DLL or RAW identity changed")
        state.update(status="complete128_screen_positive_full7338_pending" if reduction >= .05 else "closed_negative_complete128_below5percent",
            medians=medians, reduction=reduction, sources=sources, three_same_process128_pairs_not_formal=True,
            full7338_recovery_external_training_GT_and_production_not_run=True, node_seconds=time.perf_counter()-started, counted=False)
        reference.atomic_json(output / "RESULT.json", state)
        print({name: state[name] for name in ("status", "medians", "reduction", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), node_seconds=time.perf_counter()-started)
        reference.atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        reference.atomic_json(output / "RESOURCE_CLOSE.json", {"original_threads_restored": cv2.getNumThreads() == old_threads,
            "original_write_restored": reference.quality_engine.ScoreCache.write is original_write,
            "original_quality_factory_restored": exact_bayer_gray.compile_quality is original_factory,
            "owned_pools_drained": True, "counted": False, "node_seconds": time.perf_counter()-started})


if __name__ == "__main__":
    main()
