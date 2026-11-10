"""Measure three complete cold current-versus-vector RAW128 scoring pairs."""

import argparse
from contextlib import nullcontext
import importlib
from pathlib import Path
import statistics
import time

import cv2
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
from planar_corner import planar_corner_execution
from vector_gray_stats import vector_gray_stats_execution

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "library"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT / "vector_gray_stats_001")
    generic = reference.load(args.generic / "RESULT.json")
    if generic["status"] != "exact_generic_and32RAW_passed_complete128_pending" or not all(generic["checks"].values()):
        raise ValueError("The original vector exactness and lifecycle gate failed")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    import sys
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    common = OUT / "qc001/b0/native_planar_corner.dll"
    gray_library = OUT / "bg001/b0/native_bayer_gray.dll"
    sources_paths = [Path(__file__), Path(__file__).with_name("vector_gray_stats.py"), Path(__file__).with_name("native_vector_gray_stats.cpp"),
        Path(__file__).with_name("exact_bayer_gray.py"), Path(__file__).with_name("native_quality_workspace.py"),
        Path(reference.quality_histogram.__file__), reference.V1 / "quality_engine.py", reference.V2 / "single_copy_raw_reader.py",
        Path(baseline.__file__), baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", baseline.TOOLS / "mcap_audit.py",
        args.library.resolve(strict=True), gray_library, Path(__file__).with_name("planar_corner.py"), common]
    sources = [reference.data_identity(path) for path in sources_paths]
    if sources[1:14] != generic["sources"][1:]:
        raise ValueError("An executed generic vector source or library changed")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(sources_paths):
        if path.suffix != ".dll":
            (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    contract = reference.load(OUT / "bg001/s0/CONTRACT.json")
    expected = reference.load(OUT / "qc001/s0/t0/planar/SCORES.json")["rows"]
    reference.atomic_json(output / "CONTRACT.json", contract)
    process = psutil.Process()
    reference.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    index = McapIndex(baseline.RAW)
    raw_stat = baseline.RAW.stat()
    old_threads = cv2.getNumThreads()
    original_factory = exact_bayer_gray.compile_quality
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
    state = {"status": "running", "pairs": [], "accepted_count_contribution": 0}
    try:
        cv2.setNumThreads(1)
        for trial, order in enumerate((("current", "vector"), ("vector", "current"), ("current", "vector"))):
            pair = {"trial": trial, "order": order, "modes": {}}
            rows_by_mode = {}
            for mode in order:
                folder = output / ("t" + str(trial)) / mode
                folder.mkdir(parents=True, exist_ok=False)
                before = time.perf_counter()
                with planar_corner_execution(common) as corner_trace:
                    with (vector_gray_stats_execution(args.library) if mode == "vector" else nullcontext()) as vector_trace:
                        with exact_bayer_gray.raw_gray_quality_execution(gray_library, demosaic_rggb8_to_bgr) as gray_trace:
                            with reference.quality_histogram.histogram_quality_execution(True) as histogram_trace:
                                cache = reference.quality_engine.ScoreCache(folder / "quality", contract["cache_contract"])
                                rows, detail = reference.quality_engine.score_messages(
                                    reference.iter_single_copy_messages(index, baseline.CAMERA_TOPIC, contract["selected_ids"]),
                                    decode_bayer, lambda raw: raw, reference.quality_engine.compact_quality,
                                    workers=4, capacity=16, cache=cache, maximum_seconds=60)
                whole = time.perf_counter() - before
                rows_by_mode[mode] = [{name: row[name] for name in fields} for row in rows]
                if detail["submitted"] != 128 or detail["cache_hits"] or rows_by_mode[mode] != expected:
                    raise ValueError("The original complete RAW128 metadata, scores or cache changed")
                if not (corner_trace["original_factory_restored"] and corner_trace["workspace"]["DLL_unloaded"]
                        and corner_trace["workspace"]["native_calls"] == 128 and gray_trace["builder_restored"]
                        and gray_trace["engine"]["DLL_unloaded"] and histogram_trace["aliases_restored"]):
                    raise ValueError("An owned original full-quality lifecycle changed")
                if mode == "vector" and not (vector_trace["original_factory_restored"] and vector_trace["engine"]["DLL_unloaded"]
                        and vector_trace["engine"]["native_calls"] == 128 and not vector_trace["engine"]["fallback_calls"]
                        and not vector_trace["engine"]["retained_output_bytes"]):
                    raise ValueError("The actual four-worker vector consumer was not verified")
                pair["modes"][mode] = {"complete_phase_seconds": whole, "detail": detail, "gray_execution": gray_trace,
                    "corner_execution": corner_trace, "vector_execution": vector_trace, "histogram": histogram_trace,
                    "peak_RSS_bytes": process.memory_info().peak_wset}
                reference.atomic_json(folder / "SCORES.json", {"rows": rows_by_mode[mode], "counted": False})
                reference.atomic_json(folder / "TRACE.json", pair["modes"][mode])
            before = time.perf_counter()
            left, right = (output / ("t" + str(trial)) / mode / "quality" for mode in ("current", "vector"))
            names = sorted(path.name for path in left.iterdir())
            pair["checks"] = {"all128_original_metadata_six_fields_order_exact": rows_by_mode["current"] == rows_by_mode["vector"] == expected,
                "all128_original_cold_cache_bytes_exact": len(names) == 128 and names == sorted(path.name for path in right.iterdir())
                    and all((left / name).read_bytes() == (right / name).read_bytes() for name in names),
                "original_quality_compiler_restored": exact_bayer_gray.compile_quality is original_factory}
            pair["comparison_seconds_charged_each_side"] = time.perf_counter() - before
            pair["phase_seconds"] = {mode: pair["modes"][mode]["complete_phase_seconds"] + pair["comparison_seconds_charged_each_side"] for mode in order}
            state["pairs"].append(pair)
            reference.atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": pair["phase_seconds"], "checks": pair["checks"]}, flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("An original RAW128 paired exactness check failed")
        medians = {mode: statistics.median(pair["phase_seconds"][mode] for pair in state["pairs"]) for mode in ("current", "vector")}
        reduction = 1 - medians["vector"] / medians["current"]
        if sources != [reference.data_identity(path) for path in sources_paths] or (raw_stat.st_size, raw_stat.st_mtime_ns) != (
                baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns):
            raise ValueError("An original vector source or RAW identity changed")
        state.update(status="complete128_screen_positive_complete7338_pending" if reduction >= .05 else "closed_negative_complete128_below5percent",
            medians=medians, reduction=reduction, sources=sources, three_same_process128_pairs_not_formal=True,
            full7338_recovery_external_and_training_not_run=True, node_seconds=time.perf_counter() - started, counted=False)
        reference.atomic_json(output / "RESULT.json", state)
        print({key: state[key] for key in ("status", "medians", "reduction", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        reference.atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        reference.atomic_json(output / "RESOURCE_CLOSE.json", {"original_threads_restored": cv2.getNumThreads() == old_threads,
            "original_compiler_restored": exact_bayer_gray.compile_quality is original_factory,
            "owned_pools_drained": True, "counted": False, "node_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
