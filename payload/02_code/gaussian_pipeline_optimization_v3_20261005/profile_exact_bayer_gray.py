"""Compare complete original cold RAW scoring with exact fused grayscale input."""

import argparse
import importlib
import json
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
import quality_histogram
from native_quality_workspace import native_quality_workspace_execution
from exact_bayer_gray import BayerGrayEngine, BayerGrayWorkspace, raw_gray_quality_execution
from single_copy_raw_reader import iter_single_copy_messages


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "library"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    generic = load(args.generic / "RESULT.json")
    if generic["status"] != "generic_passed_component_and_complete_stage_pending" or not all(generic["checks"].values()):
        raise ValueError("The registered exact Bayer gray generic verification failed")
    output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    excluded_path = ROOT / "04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    paths = [Path(__file__), Path(__file__).with_name("exact_bayer_gray.py"), Path(__file__).with_name("native_bayer_gray.cpp"),
             Path(__file__).with_name("native_quality_workspace.py"), Path(quality_histogram.__file__), V1 / "quality_engine.py",
             V2 / "single_copy_raw_reader.py", Path(baseline.__file__), baseline.TOOLS / "camera_motion.py",
             baseline.TOOLS / "color_contract.py", baseline.TOOLS / "mcap_audit.py", args.library.resolve(strict=True), excluded_path]
    sources = [data_identity(path) for path in paths]
    if not all(sources[left] == generic["sources"][right] for left, right in ((1,1),(2,2),(3,3),(4,4),(5,5),(9,6),(11,7))):
        raise ValueError("An exact Bayer gray source differs from the executed generic verification")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths):
        if path != args.library.resolve():
            (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time(), "scope": "Windows"})
    index = McapIndex(baseline.RAW)
    raw_stat = baseline.RAW.stat()
    excluded = {int(value) for value in load(excluded_path)["frames"]}
    allowed = [frame for frame in range(7338) if frame not in excluded]
    selected = [allowed[position] for position in np.linspace(0, len(allowed)-1, 128, dtype=int)]
    if index.message_count(baseline.CAMERA_TOPIC) != 7338 or len(excluded) != 48 or selected != load(
            ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001/p0/CONTRACT.json")["selected_ids"]:
        raise ValueError("The original RAW or fixed chronological pilot changed")
    contract = {"version": 1, "color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__, "numpy": np.__version__,
                "scorer": "compact", "sources": [sources[position]["sha256"] for position in (5,7,8,9)]}
    atomic_json(output / "CONTRACT.json", {"sources": sources, "selected_ids": selected, "workers": 4, "capacity": 16,
        "repetitions": 3, "control": "accepted_Q78_R104_R141", "cache_contract": contract, "counted": False})
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    state = {"status": "running", "pairs": [], "accepted_count_contribution": 0}
    try:
        engine = BayerGrayEngine(args.library)
        workspace = BayerGrayWorkspace(engine)
        compared = 0
        iterator = iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected)
        try:
            for frame, message in iterator:
                _, _, raw = decode_bayer(message["payload"])
                expected = cv2.cvtColor(demosaic_rggb8_to_bgr(raw), cv2.COLOR_BGR2GRAY)
                actual = workspace.gray(raw)
                if not np.array_equal(expected, actual):
                    np.savez(output / "FIRST_RAW_FAILURE.npz", original=expected, candidate=actual, raw=raw)
                    raise ValueError("An original full RAW gray pixel changed in frame " + str(frame))
                compared += 1
        finally:
            iterator.close()
            workspace.close()
            engine.close()
        state["all128_complete_gray_arrays_bits_exact"] = compared == 128
        fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality", "cache_hit")
        for trial in range(3):
            pair = {"trial": trial, "order": ("native", "fused") if trial % 2 == 0 else ("fused", "native"), "modes": {}}
            outputs = {}
            for mode in pair["order"]:
                folder = output / ("t" + str(trial)) / mode
                folder.mkdir(parents=True, exist_ok=False)
                before = time.perf_counter()
                context = native_quality_workspace_execution(True) if mode == "native" else raw_gray_quality_execution(args.library, demosaic_rggb8_to_bgr)
                with context as execution:
                    with quality_histogram.histogram_quality_execution(True) as histogram_trace:
                        cache = quality_engine.ScoreCache(folder / "quality", contract)
                        converter = demosaic_rggb8_to_bgr if mode == "native" else (lambda raw: raw)
                        rows, detail = quality_engine.score_messages(iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected),
                            decode_bayer, converter, quality_engine.compact_quality, workers=4, capacity=16, cache=cache, maximum_seconds=60)
                complete = time.perf_counter()-before
                workspace_trace = execution["workspace"] if mode == "native" else execution["fused_workspace"]
                closed = (workspace_trace["closed"] and workspace_trace["owned_plane_bytes_after_close"] == 0
                          and workspace_trace["calls"] == 384 and histogram_trace["aliases_restored"])
                if mode == "native":
                    closed &= execution["original_builder_restored"]
                else:
                    closed &= execution["builder_restored"] and execution["engine"]["DLL_unloaded"] and execution["engine"][
                        "active_calls"] == 0 and execution["engine"]["calls"] == 128 and execution["fallback_workspace"]["calls"] == 0
                if not closed or detail["submitted"] != 128 or detail["cache_hits"] != 0 or len(rows) != 128:
                    raise ValueError("An original complete RAW cold consumer or owned close contract failed")
                outputs[mode] = [{field: row[field] for field in fields} for row in rows]
                pair["modes"][mode] = {"complete_phase_seconds": complete, "detail": detail, "execution": execution,
                                       "histogram": histogram_trace, "peak_RSS_bytes": process.memory_info().peak_wset}
                atomic_json(folder / "SCORES.json", {"rows": outputs[mode], "counted": False})
                atomic_json(folder / "TRACE.json", pair["modes"][mode])
            before = time.perf_counter()
            left, right = output / ("t"+str(trial)) / "native/quality", output / ("t"+str(trial)) / "fused/quality"
            names = sorted(path.name for path in left.iterdir())
            pair["checks"] = {"all128_metadata_and_six_quality_fields_exact": outputs["native"] == outputs["fused"],
                "all128_frozen_RAW_ids": all([row["camera_frame"] for row in outputs[mode]] == selected for mode in pair["order"]),
                "all128_original_cache_file_bytes_exact": len(names) == 128 and names == sorted(path.name for path in right.iterdir())
                    and all((left/name).read_bytes() == (right/name).read_bytes() for name in names)}
            pair["comparison_seconds"] = time.perf_counter()-before
            state["pairs"].append(pair)
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": {mode: pair["modes"][mode]["complete_phase_seconds"] for mode in pair["order"]},
                   "checks": pair["checks"]}, flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("An original complete RAW pilot science or persistent cache changed")
        medians = {mode: statistics.median(pair["modes"][mode]["complete_phase_seconds"] for pair in state["pairs"])
                   for mode in ("native", "fused")}
        reduction = 1-medians["fused"]/medians["native"]
        if sources != [data_identity(path) for path in paths] or (raw_stat.st_size, raw_stat.st_mtime_ns) != (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns):
            raise ValueError("An original source or protected RAW input changed")
        state.update(status="component_screen_positive_complete_stage_pending" if reduction >= .05 else "component_screen_negative_below5percent",
            sources=sources, medians=medians, reduction=reduction, node_seconds=time.perf_counter()-started,
            three_same_process_component_pairs_not_formal=True, full7338_original1002_and_formal_not_run=True, counted=False)
        atomic_json(output / "RESULT.json", state)
        print({key: state[key] for key in ("status", "medians", "reduction", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.perf_counter()-started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"OpenCV_threads_restored": cv2.getNumThreads() == old_threads,
            "all_owned_scoring_pool_lifetimes_completed": True, "node_seconds": time.perf_counter()-started, "counted": False})


if __name__ == "__main__":
    main()
