"""Check original scorer failures, partial-cache recovery and external image consumers."""

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import quality_engine
import quality_histogram as histogram
import run_initial_pipeline as entry
from native_quality_workspace import native_quality_workspace_execution
from single_copy_raw_reader import iter_single_copy_messages
from external_quality_histogram import score_all


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metadata(rows):
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality")
    return [{name: row[name] for name in fields} for row in rows]


def cache_equal(left, right, expected):
    names = sorted(path.name for path in left.iterdir())
    return len(names) == expected and all(name.endswith(".json") for name in names) and names == sorted(
        path.name for path in right.iterdir()) and all((left / name).read_bytes() == (right / name).read_bytes() for name in names)


def closed(workspace, trace):
    owner = workspace["workspace"]
    return workspace["original_builder_restored"] and trace["aliases_restored"] and (owner is None or
        owner["closed"] and owner["owned_plane_bytes_after_close"] == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    current = psutil.Process()
    atomic_json(output / "EXECUTION.json", {"pid": current.pid, "creation_time": current.create_time(),
        "actual_self_and_parents": [{"pid": process.pid, "creation_time": process.create_time()}
                                    for process in [current] + current.parents()]})
    pilot = load(args.pilot / "CONTRACT.json")
    selected, cache_contract = pilot["selected_ids"], pilot["cache_contract"]
    if len(selected) != 128 or pilot["excluded"] != 48:
        raise ValueError("The frozen original pilot role or frame count changed")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    index = McapIndex(baseline.RAW)
    original_raw = baseline.RAW.stat()
    paths = [Path(__file__), Path(__file__).with_name("native_quality_workspace.py"), Path(histogram.__file__),
             V1 / "quality_engine.py", V1 / "run_initial_pipeline.py", V2 / "single_copy_raw_reader.py",
             Path(__file__).with_name("external_quality_histogram.py"), baseline.TOOLS / "mcap_audit.py",
             baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py"]
    sources = [data_identity(path) for path in paths]
    if any(sources[index]["sha256"] != pilot["sources"][other]["sha256"] for index, other in ((1, 1), (2, 2), (3, 3), (5, 4))):
        raise ValueError("The original sources differ from the complete RAW pilot")
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for number, path in enumerate(paths):
        (snapshots / (str(number) + "_" + path.name)).write_bytes(path.read_bytes())
    images = sorted(args.image_root.glob("*.png"))
    if len(images) != 360 or index.message_count(baseline.CAMERA_TOPIC) != 7338:
        raise ValueError("The original full RAW or external image denominator changed")
    identities = {str(path): data_identity(path) for path in images}
    original_aliases = histogram.make_quality, quality_engine.compact_quality, entry.compact_quality
    original_threads = {thread.ident for thread in threading.enumerate()}
    original_cv_threads = cv2.getNumThreads()
    reference = load(args.pilot / "t0/native/SCORES.json")["original_full_metadata_and_quality"]
    reference = [{key: value for key, value in row.items() if key != "cache_hit"} for row in reference]
    state = {"status": "running", "sources": sources, "input_images": list(identities.values()),
             "failure_recovery_pairs": [], "real_directory_errors": [], "external_pairs": [], "checks": {},
             "original_selected128_ids": selected, "original_raw_path": str(baseline.RAW),
             "original_raw_size": original_raw.st_size, "original_raw_mtime_ns": original_raw.st_mtime_ns,
             "public360_is_development_image_only_component_no_GT_or_unseen_whole_pipeline": True,
             "external_performance_recorded_not_required_for_scoped_Box2_RAW_acceptance": True,
             "counted": False}
    atomic_json(output / "PROTOCOL.json", state)
    cv2.setNumThreads(1)

    def require(name, condition):
        state["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def execute(cache, mode, progress=None):
        messages = iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected)
        traces = {}
        try:
            with native_quality_workspace_execution(mode == "workspace") as workspace:
                traces["workspace"] = workspace
                with histogram.histogram_quality_execution(True) as hist:
                    traces["histogram"] = hist
                    rows, detail = quality_engine.score_messages(messages, decode_bayer, demosaic_rggb8_to_bgr,
                        quality_engine.compact_quality, workers=4, capacity=16, cache=cache,
                        maximum_seconds=120, progress=progress)
            return rows, detail, traces
        finally:
            messages.close()
            state["last_execution_traces"] = traces

    try:
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
            modes = {}
            name = error_type.__name__
            for mode in ("native", "workspace"):
                folder = output / "faults" / name / mode
                cache = quality_engine.ScoreCache(folder / "quality", cache_contract)
                progress = []

                def fail_at_eight(completed):
                    progress.append(completed)
                    if completed == 8:
                        raise error_type("registered_original_scoring_consumer_failure")

                before = time.perf_counter()
                outcome = None
                try:
                    execute(cache, mode, fail_at_eight)
                except error_type as error:
                    outcome = [type(error).__name__, str(error)]
                trace = state["last_execution_traces"]
                modes[mode] = {"failure": outcome, "progress": progress, "failure_seconds": time.perf_counter() - before,
                               "failure_trace": trace, "partial_cache_files": len(list(cache.directory.iterdir()))}
                require(name + "_" + mode + "_original_failure23_commits_and_closure", outcome == [name,
                    "registered_original_scoring_consumer_failure"] and progress == list(range(1, 9)) and
                    modes[mode]["partial_cache_files"] == 23 and closed(trace["workspace"], trace["histogram"]))
            left = output / "faults" / name / "native/quality"
            right = output / "faults" / name / "workspace/quality"
            require(name + "_all23_partial_original_cache_bytes_exact", cache_equal(left, right, 23))
            for mode in ("native", "workspace"):
                folder = output / "faults" / name / mode
                before = time.perf_counter()
                rows, detail, trace = execute(quality_engine.ScoreCache(folder / "quality", cache_contract), mode)
                modes[mode].update(recovery_seconds=time.perf_counter() - before, recovery_detail=detail, recovery_trace=trace)
                atomic_json(folder / "RECOVERED_SCORES.json", {"full_metadata_and_quality": metadata(rows), "detail": detail})
                require(name + "_" + mode + "_original128_recovery23_hits_and_closure", len(rows) == detail["submitted"] == 128
                    and detail["cache_hits"] == 23 and metadata(rows) == reference and closed(trace["workspace"], trace["histogram"]))
            require(name + "_all128_recovered_original_cache_bytes_exact", cache_equal(left, right, 128))
            state["failure_recovery_pairs"].append({"error": name, "modes": modes})
            atomic_json(output / "STATE.json", state)
        for mode in ("native", "workspace"):
            folder = output / "real_directory_error" / mode
            cache = quality_engine.ScoreCache(folder / "unused_cache", cache_contract)
            sentinel = folder / "regular_file_cannot_be_cache_directory"
            sentinel.write_text("owned_error_fixture", encoding="ascii")
            cache.directory = sentinel
            error_record = None
            before = time.perf_counter()
            try:
                execute(cache, mode)
            except OSError as error:
                error_record = {"type": type(error).__name__, "errno": error.errno,
                                "message": str(error), "winerror": getattr(error, "winerror", None)}
            trace = state["last_execution_traces"]
            state["real_directory_errors"].append({"mode": mode, "error": error_record,
                "seconds": time.perf_counter() - before, "trace": trace})
            require(mode + "_original_real_disk_failure_propagated_and_closed", error_record is not None
                and closed(trace["workspace"], trace["histogram"]) and sentinel.read_text(encoding="ascii") == "owned_error_fixture")
        a, b = [row["error"] for row in state["real_directory_errors"]]
        require("real_disk_failure_original_type_errno_winerror_exact", all(a[key] == b[key] for key in ("type", "errno", "winerror")))
        for trial in range(3):
            modes, records = {}, {}
            order = ("native", "workspace") if trial % 2 == 0 else ("workspace", "native")
            for mode in order:
                if time.perf_counter() - started > 600:
                    raise TimeoutError("The registered complete supplement component budget expired")
                before = time.perf_counter()
                with native_quality_workspace_execution(mode == "workspace") as workspace:
                    function, hist = histogram.make_quality(original_aliases[1])
                    records[mode] = score_all(images, identities, function)
                modes[mode] = {"complete_component_seconds": time.perf_counter() - before,
                               "workspace": workspace, "histogram": hist}
                atomic_json(output / ("external_t" + str(trial) + "_" + mode + "_SCORES.json"), records[mode])
                owner = workspace["workspace"]
                require("external_t" + str(trial) + "_" + mode + "_all360_native_calls_and_closure", len(records[mode]) == hist[
                    "histogram_calls"] == 360 and hist["native_calls"] == 0 and workspace["original_builder_restored"]
                    and (owner is None or owner["calls"] == 1080 and owner["closed"] and owner["owned_plane_bytes_after_close"] == 0))
            require("external_t" + str(trial) + "_all360_original_quality_metadata_exact", records["native"] == records["workspace"])
            state["external_pairs"].append({"trial": trial, "order": order, "modes": modes})
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "external_complete_seconds": {mode: modes[mode]["complete_component_seconds"] for mode in order},
                   "all360_quality_exact": records["native"] == records["workspace"]}, flush=True)
        median = {mode: statistics.median(row["modes"][mode]["complete_component_seconds"] for row in state["external_pairs"])
                  for mode in ("native", "workspace")}
        state["external_complete_medians"] = median
        state["external_complete_reduction"] = 1 - median["workspace"] / median["native"]
        require("all_current_sources_and_full360_image_identities_unchanged", sources == [data_identity(path) for path in paths]
            and identities == {str(path): data_identity(path) for path in images})
        raw = baseline.RAW.stat()
        require("original_RAW_size_mtime_and_all_role_ids_unchanged", (raw.st_size, raw.st_mtime_ns) == (
            original_raw.st_size, original_raw.st_mtime_ns) and selected == load(args.pilot / "CONTRACT.json")["selected_ids"])
        require("no_owned_worker_alive_and_original_all_aliases_restored", not any(thread.is_alive() and thread.ident not in original_threads
            for thread in threading.enumerate()) and original_aliases == (histogram.make_quality, quality_engine.compact_quality, entry.compact_quality))
        state.update(status="original_failures_recovery_and_all360_external_quality_passed_scope_performance_recorded",
                     elapsed_seconds=time.perf_counter() - started, accepted_count_contribution=0)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(state["checks"]), "external_medians": median,
               "external_reduction": state["external_complete_reduction"], "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        cv2.setNumThreads(original_cv_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"pid": current.pid, "creation_time": current.create_time(),
            "OpenCV_threads_restored": cv2.getNumThreads() == original_cv_threads,
            "original_aliases_restored": original_aliases == (histogram.make_quality, quality_engine.compact_quality, entry.compact_quality),
            "no_new_worker_alive": not any(thread.is_alive() and thread.ident not in original_threads for thread in threading.enumerate()),
            "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
