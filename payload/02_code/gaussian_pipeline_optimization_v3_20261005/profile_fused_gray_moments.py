"""Measure complete cold RAW scoring against the accepted R104 and R141 control."""

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import quality_engine
import quality_histogram as histogram
from fused_gray_moments import fused_quality_execution
from single_copy_raw_reader import iter_single_copy_messages


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gm001")
    output.mkdir(parents=True, exist_ok=False)
    generic = load(args.generic / "RESULT.json")
    if generic["status"] != "generic_passed_pilot_and_complete_stage_pending" or not all(generic["checks"].values()):
        raise ValueError("The preregistered complete gray statistics generic failed")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    exclusion = ROOT / "04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    paths = [Path(__file__), Path(__file__).with_name("fused_gray_moments.py"), Path(histogram.__file__),
             V1 / "quality_engine.py", V2 / "single_copy_raw_reader.py", Path(baseline.__file__),
             baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", exclusion,
             args.library.resolve(strict=True), Path(__file__).with_name("native_quality_workspace.py"),
             Path(__file__).with_name("native_gray_moments.cpp")]
    sources = [data_identity(path) for path in paths]
    if not all(sources[left] == generic["sources"][right] for left, right in ((1, 1), (2, 4), (3, 5), (10, 6), (11, 3), (12, 2))):
        raise ValueError("An executed scientific source differs from the full generic check")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        if path != args.library.resolve():
            (snapshot / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
    excluded = {int(key) for key in load(exclusion)["frames"]}
    index = McapIndex(baseline.RAW)
    total = index.message_count(baseline.CAMERA_TOPIC)
    if total != 7338 or len(excluded) != 48:
        raise ValueError("The original RAW or excluded-role denominator changed")
    allowed = [frame for frame in range(total) if frame not in excluded]
    selected = [allowed[position] for position in np.linspace(0, len(allowed) - 1, 128, dtype=int)]
    current_reference = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001/p0/CONTRACT.json"
    if selected != load(current_reference)["selected_ids"]:
        raise ValueError("The frozen original R141 RAW pilot membership changed")
    contract = {"version": 1, "color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__, "numpy": np.__version__,
                "scorer": "compact", "sources": [sources[position]["sha256"] for position in (3, 5, 7, 8)]}
    raw = baseline.RAW.stat()
    atomic_json(output / "CONTRACT.json", {"sources": sources, "selected_ids": selected, "original_frames": total,
        "excluded": len(excluded), "pilot_frames": 128, "workers": 4, "capacity": 16, "repetitions": 3,
        "minimum_complete_phase_reduction": .05, "cache_contract": contract,
        "original_raw_path": str(baseline.RAW), "original_raw_size": raw.st_size, "original_raw_mtime_ns": raw.st_mtime_ns,
        "control": "current_R104_histogram_and_R141_native_planes", "no_pose_SfM_GT_or_quality_ranking": True,
        "all_IO_decode_original_cache_fsync_pool_drain_DLL_and_workspace_close_in_phase": True, "counted": False})
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    preparation = time.perf_counter() - started
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    state = {"status": "running", "pairs": [], "accepted_count_contribution": 0}
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
    try:
        for trial in range(3):
            if time.perf_counter() - started > 600:
                raise TimeoutError("The frozen complete component time budget was exceeded")
            order = ("native", "fused") if trial % 2 == 0 else ("fused", "native")
            pair = {"trial": trial, "order": order, "modes": {}}
            outputs = {}
            for mode in order:
                folder = output / ("t" + str(trial)) / mode
                folder.mkdir(parents=True, exist_ok=False)
                before = time.perf_counter()
                with fused_quality_execution(mode == "fused", args.library) as execution:
                    with histogram.histogram_quality_execution(True) as trace:
                        cache = quality_engine.ScoreCache(folder / "quality", contract)
                        rows, detail = quality_engine.score_messages(
                            iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected), decode_bayer,
                            demosaic_rggb8_to_bgr, quality_engine.compact_quality, workers=4, capacity=16,
                            cache=cache, maximum_seconds=120)
                complete = time.perf_counter() - before
                outputs[mode] = [{name: row[name] for name in fields} for row in rows]
                workspace = execution["workspace"]["workspace"]
                native_calls = (2 if mode == "fused" else 3) * 128
                closed = execution["fused_builder_restored"] and execution["workspace"]["original_builder_restored"] and workspace[
                    "closed"] and workspace["owned_plane_bytes_after_close"] == 0 and workspace["calls"] == native_calls
                if mode == "fused":
                    engine = execution["engine"]
                    closed &= engine["DLL_unloaded"] and engine["owned_output_bytes_after_close"] == 0 and engine["active_calls"] == 0 and engine["calls"] == 128
                if len(rows) != 128 or detail["submitted"] != 128 or detail["cache_hits"] != 0 or not trace["aliases_restored"] or not closed:
                    raise ValueError("A complete cold scoring denominator or original closure changed")
                pair["modes"][mode] = {"complete_phase_seconds": complete, "detail": detail, "execution": execution,
                    "histogram": trace, "peak_RSS_bytes": getattr(process.memory_info(), "peak_wset", 0)}
                atomic_json(folder / "SCORES.json", {"original_full_metadata_and_quality": outputs[mode], "counted": False})
                atomic_json(folder / "TRACE.json", pair["modes"][mode])
            before = time.perf_counter()
            a, b = output / ("t" + str(trial)) / "native/quality", output / ("t" + str(trial)) / "fused/quality"
            names = sorted(path.name for path in a.iterdir())
            exact = len(names) == 128 and names == sorted(path.name for path in b.iterdir()) and all(
                (a / name).read_bytes() == (b / name).read_bytes() for name in names)
            pair["checks"] = {"all128_original_full_metadata_and_six_quality_fields_exact": outputs["native"] == outputs["fused"],
                              "all128_persistent_cache_bytes_exact": exact,
                              "all128_frozen_original_RAW_ids": all([row["camera_frame"] for row in outputs[mode]] == selected for mode in order)}
            pair["comparison_seconds"] = time.perf_counter() - before
            state["pairs"].append(pair)
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": {mode: pair["modes"][mode]["complete_phase_seconds"] for mode in order},
                   "checks": pair["checks"]}, flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("A complete original RAW quality or cache byte changed")
        medians = {mode: statistics.median(pair["modes"][mode]["complete_phase_seconds"] for pair in state["pairs"])
                   for mode in ("native", "fused")}
        reduction = 1 - medians["fused"] / medians["native"]
        current_raw = baseline.RAW.stat()
        if sources != [data_identity(path) for path in paths] or (raw.st_size, raw.st_mtime_ns) != (current_raw.st_size, current_raw.st_mtime_ns):
            raise ValueError("An original source or protected RAW input changed")
        state.update(status="component_screen_positive_complete_stage_pending" if reduction >= .05 else "component_screen_negative_below5percent",
                     sources=sources, medians=medians, reduction=reduction, first_preparation_seconds=preparation,
                     complete_node_before_publication_seconds=time.perf_counter() - started,
                     complete_full7338_entry_and_formal_not_run=True, formal_pairs_executed=0,
                     three_same_process_component_pairs_only=True, accepted_count_contribution=0)
        atomic_json(output / "RESULT.json", state)
        print({key: state[key] for key in ("status", "medians", "reduction", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"OpenCV_threads_restored": cv2.getNumThreads() == old_threads,
            "actual_PID": os.getpid(), "actual_creation_time": process.create_time(),
            "all_pool_and_owned_scoring_lifetimes_completed": True, "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
