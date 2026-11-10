"""Screen complete cold raw scoring with original workers and durable caches."""

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
from native_quality_workspace import native_quality_workspace_execution
from single_copy_raw_reader import iter_single_copy_messages


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metadata(rows):
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
    return [{name: row[name] for name in fields} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    generic = load(args.generic / "RESULT.json")
    if generic["status"] != "generic_passed_pilot_and_complete_stage_pending" or not all(generic["checks"].values()):
        raise ValueError("The frozen complete quality workspace generic guard failed")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    exclusion = ROOT / "04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    paths = [Path(__file__), Path(__file__).with_name("native_quality_workspace.py"), Path(histogram.__file__),
             V1 / "quality_engine.py", V2 / "single_copy_raw_reader.py", Path(baseline.__file__),
             baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", exclusion]
    sources = [data_identity(path) for path in paths]
    if sources[1] != generic["sources"][1] or sources[2] != generic["sources"][2] or sources[3] != generic["sources"][3]:
        raise ValueError("A scientific source differs from the complete generic check")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        (snapshot / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
    excluded = {int(key) for key in load(exclusion)["frames"]}
    index = McapIndex(baseline.RAW)
    total = index.message_count(baseline.CAMERA_TOPIC)
    if total != 7338 or len(excluded) != 48:
        raise ValueError("The original RAW or excluded-role denominator changed")
    allowed = [frame for frame in range(total) if frame not in excluded]
    selected = [allowed[index] for index in np.linspace(0, len(allowed) - 1, 128, dtype=int)]
    cache_contract = {"version": 1, "color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__,
                      "numpy": np.__version__, "scorer": "compact",
                      "sources": [sources[index]["sha256"] for index in (3, 5, 7, 8)]}
    original_raw = baseline.RAW.stat()
    atomic_json(output / "CONTRACT.json", {"sources": sources, "selected_ids": selected, "selection": "uniform_RAW_order_exclusions_only",
        "original_frames": total, "excluded": len(excluded), "pilot_frames": 128, "workers": 4, "capacity": 16,
        "minimum_complete_pilot_phase_reduction": .05, "repetitions": 3, "formal_fresh_pairs_executed": 0,
        "cache_contract": cache_contract, "original_raw_path": str(baseline.RAW), "original_raw_size": original_raw.st_size,
        "original_raw_mtime_ns": original_raw.st_mtime_ns, "no_pose_SfM_GT_or_quality_ranking": True,
        "all_original_cold_cache_commits_and_worker_shutdown_in_phase": True, "counted": False})
    first_preparation = time.perf_counter() - started
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    state = {"status": "running", "pairs": [], "counted": False}
    try:
        for trial in range(3):
            if time.perf_counter() - started > 600:
                raise TimeoutError("The original component node time budget was exceeded")
            order = ("native", "workspace") if trial % 2 == 0 else ("workspace", "native")
            pair = {"trial": trial, "order": order, "modes": {}}
            outputs = {}
            for mode in order:
                folder = output / ("t" + str(trial)) / mode
                folder.mkdir(parents=True, exist_ok=False)
                before = time.perf_counter()
                with native_quality_workspace_execution(mode == "workspace") as workspace_trace:
                    with histogram.histogram_quality_execution(True) as histogram_trace:
                        cache = quality_engine.ScoreCache(folder / "quality", cache_contract)
                        rows, detail = quality_engine.score_messages(
                            iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected), decode_bayer,
                            demosaic_rggb8_to_bgr, quality_engine.compact_quality, workers=4, capacity=16,
                            cache=cache, maximum_seconds=120)
                complete = time.perf_counter() - before
                outputs[mode] = metadata(rows)
                workspace = workspace_trace["workspace"]
                if len(rows) != 128 or detail["submitted"] != 128 or detail["cache_hits"] != 0 or not histogram_trace[
                        "aliases_restored"] or not workspace_trace["original_builder_restored"]:
                    raise ValueError("An original complete cold scorer denominator, cache or alias changed")
                if workspace is not None and (not workspace["closed"] or workspace["owned_plane_bytes_after_close"] != 0 or workspace["calls"] != 384):
                    raise ValueError("The complete thread-owned plane execution or closure changed")
                pair["modes"][mode] = {"complete_phase_seconds": complete, "detail": detail, "workspace": workspace_trace,
                                       "histogram": histogram_trace, "peak_process_RSS_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0)}
                atomic_json(folder / "SCORES.json", {"original_full_metadata_and_quality": outputs[mode], "counted": False})
                atomic_json(folder / "TRACE.json", pair["modes"][mode])
            before = time.perf_counter()
            a, b = output / ("t" + str(trial)) / "native/quality", output / ("t" + str(trial)) / "workspace/quality"
            names = sorted(path.name for path in a.iterdir())
            cache_equal = len(names) == 128 and names == sorted(path.name for path in b.iterdir()) and all(
                (a / name).read_bytes() == (b / name).read_bytes() for name in names)
            pair["checks"] = {"all128_full_metadata_and_six_quality_fields_exact": outputs["native"] == outputs["workspace"],
                              "all128_original_cache_files_bytes_exact": cache_equal,
                              "all128_original_RAW_ids_in_registered_order": all([row["camera_frame"] for row in outputs[mode]] == selected for mode in order)}
            pair["complete_pair_comparison_seconds"] = time.perf_counter() - before
            state["pairs"].append(pair)
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "complete_phase_seconds": {mode: pair["modes"][mode]["complete_phase_seconds"] for mode in order},
                   "checks": pair["checks"]}, flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("A complete original RAW pilot science or cache byte changed")
        medians = {mode: statistics.median(pair["modes"][mode]["complete_phase_seconds"] for pair in state["pairs"])
                   for mode in ("native", "workspace")}
        reduction = 1 - medians["workspace"] / medians["native"]
        current_raw = baseline.RAW.stat()
        if sources != [data_identity(path) for path in paths] or (original_raw.st_size, original_raw.st_mtime_ns) != (
                current_raw.st_size, current_raw.st_mtime_ns):
            raise ValueError("An original protected raw input or scientific source changed")
        state.update(status="component_screen_positive_complete_stage_pending" if reduction >= .05 else "component_screen_negative_below5percent",
                     sources=sources, medians=medians, reduction=reduction, first_preparation_seconds=first_preparation,
                     complete_node_before_publication_seconds=time.perf_counter() - started,
                     complete_full7338_initial_entry_and_formal_not_run=True,
                     raw_IO_CDR_demosaic_original_workers_cache_fsync_and_all_close_inside_pilot_phase=True,
                     six_sides_same_process_not_formal_fresh_pairs=True, formal_pairs_executed=0,
                     accepted_count_contribution=0)
        atomic_json(output / "RESULT.json", state)
        print({name: state[name] for name in ("status", "medians", "reduction", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"OpenCV_threads_restored": cv2.getNumThreads() == old_threads,
            "actual_PID": os.getpid(), "actual_creation_time": psutil.Process().create_time(), "workspace_pool_lifetimes_completed": True,
            "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
