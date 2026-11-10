"""Audit exact Bayer grayscale semantics and the current complete scoring cost."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import threading
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
from native_quality_workspace import NativeQualityWorkspace, native_quality_workspace_execution
import quality_histogram
from single_copy_raw_reader import iter_single_copy_messages


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time(), "scope": "Windows"})
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    excluded_path = ROOT / "04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    prior_path = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001/p0/CONTRACT.json"
    paths = [Path(__file__), V1 / "quality_engine.py", Path(quality_histogram.__file__),
             Path(__file__).with_name("native_quality_workspace.py"), V2 / "single_copy_raw_reader.py",
             Path(baseline.__file__), baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py",
             baseline.TOOLS / "mcap_audit.py", excluded_path, prior_path,
             ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R147_RAW_GRAY_INPUT_TASK.md"]
    sources = [data_identity(path) for path in paths]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for position, path in enumerate(paths):
        (snapshots / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    raw_stat = baseline.RAW.stat()
    index = McapIndex(baseline.RAW)
    excluded = {int(value) for value in load(excluded_path)["frames"]}
    if index.message_count(baseline.CAMERA_TOPIC) != 7338 or len(excluded) != 48:
        raise ValueError("The original RAW or evaluation exclusion denominator changed")
    allowed = [frame for frame in range(7338) if frame not in excluded]
    selected = [allowed[position] for position in np.linspace(0, len(allowed) - 1, 128, dtype=int)]
    if selected != load(prior_path)["selected_ids"]:
        raise ValueError("The original accepted R141 chronological pilot changed")
    old_threads, old_gray = cv2.getNumThreads(), NativeQualityWorkspace.gray
    cv2.setNumThreads(1)
    comparisons = []
    state = {"status": "running", "accepted_count_contribution": 0}
    first_failure_saved = False

    def compare(name, raw, real=False):
        nonlocal first_failure_saved
        reference_bgr = demosaic_rggb8_to_bgr(raw)
        reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        direct = cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2GRAY)
        absolute = np.abs(reference.astype(np.int16) - direct.astype(np.int16))
        row = {"name": name, "shape": list(raw.shape), "strides": list(raw.strides),
               "exact": np.array_equal(reference, direct), "different_pixels": int(np.count_nonzero(absolute)),
               "pixels": int(raw.size), "maximum_absolute_difference": int(absolute.max()), "real_RAW": real}
        if real:
            gray_scorer_source = __import__("inspect").getsource(quality_engine.compact_quality)
            anchor = "gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)"
            if gray_scorer_source.count(anchor) != 1:
                raise ValueError("The original native quality gray input anchor changed")
            namespace = dict(quality_engine.compact_quality.__globals__)
            exec(compile(gray_scorer_source.replace(anchor, "gray = image"), "<raw_gray_diagnostic_only>", "exec"), namespace)
            original_fields = quality_engine.compact_quality(reference_bgr)
            direct_fields = namespace["compact_quality"](direct)
            row.update(original_six_fields=original_fields, direct_six_fields=direct_fields,
                       six_fields_exact=original_fields == direct_fields)
        if not row["exact"] and not first_failure_saved:
            np.savez(output / "FIRST_GRAY_FAILURE.npz", raw=raw, original_gray=reference, direct_gray=direct)
            row["first_failure_arrays"] = "FIRST_GRAY_FAILURE.npz"
            first_failure_saved = True
        comparisons.append(row)

    try:
        rng = np.random.default_rng(147)
        shapes = [(1, 1), (2, 9), (3, 3), (4, 7), (7, 4), (5, 6), (6, 5), (31, 32), (32, 31), (49, 53)]
        for position, shape in enumerate(shapes):
            compare("random_" + str(position), rng.integers(0, 256, shape, dtype=np.uint8))
            compare("constant_" + str(position), np.full(shape, 255 if position % 2 else 0, dtype=np.uint8))
        for position in range(8):
            raw = np.zeros((31, 35), dtype=np.uint8)
            raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2] = rng.integers(0, 256, 4)
            compare("color_mosaic_" + str(position), raw)
        padded = rng.integers(0, 256, (37, 49), dtype=np.uint8)
        for name, value in (("padded_rows", padded[:, :41]), ("negative_rows", padded[::-1, :41]),
                            ("column_stride", padded[:, ::2]), ("transpose", padded.T)):
            compare(name, value)
        real_ids = [selected[position] for position in np.linspace(0, 127, 16, dtype=int)]
        iterator = iter_single_copy_messages(index, baseline.CAMERA_TOPIC, real_ids)
        try:
            for frame, message in iterator:
                _, _, raw = decode_bayer(message["payload"])
                compare("RAW_" + str(frame), raw, True)
        finally:
            iterator.close()
        atomic_json(output / "EQUIVALENCE.json", {"comparisons": comparisons, "all_gray_exact": all(row["exact"] for row in comparisons),
            "generic_count": 32, "real_count": 16, "fixed_real_ids": real_ids,
            "direct_native_replacement_rejected_if_any_gray_or_quality_difference": True, "counted": False})
        totals, mutex = {"demosaic_seconds_sum": 0., "gray_seconds_sum": 0., "demosaic_calls": 0, "gray_calls": 0}, threading.Lock()

        def timed_demosaic(raw):
            before = time.perf_counter()
            value = demosaic_rggb8_to_bgr(raw)
            elapsed = time.perf_counter() - before
            with mutex:
                totals["demosaic_seconds_sum"] += elapsed
                totals["demosaic_calls"] += 1
            return value

        def timed_gray(workspace, image):
            before = time.perf_counter()
            value = old_gray(workspace, image)
            elapsed = time.perf_counter() - before
            with mutex:
                totals["gray_seconds_sum"] += elapsed
                totals["gray_calls"] += 1
            return value

        contract = {"version": 1, "color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__, "numpy": np.__version__,
                    "scorer": "compact", "sources": [sources[position]["sha256"] for position in (1, 5, 6, 7)]}
        before = time.perf_counter()
        NativeQualityWorkspace.gray = timed_gray
        with native_quality_workspace_execution(True) as workspace_trace:
            with quality_histogram.histogram_quality_execution(True) as histogram_trace:
                cache = quality_engine.ScoreCache(output / "quality", contract)
                rows, detail = quality_engine.score_messages(iter_single_copy_messages(index, baseline.CAMERA_TOPIC, selected),
                    decode_bayer, timed_demosaic, quality_engine.compact_quality, workers=4, capacity=16, cache=cache, maximum_seconds=60)
        NativeQualityWorkspace.gray = old_gray
        complete = time.perf_counter() - before
        workspace = workspace_trace["workspace"]
        checks = {"all128_original_order": [row["camera_frame"] for row in rows] == selected,
                  "all128_cold_cache_misses": detail["cache_hits"] == 0 and detail["submitted"] == 128,
                  "all128_original_six_fields": all(set(row["quality"]) == set(quality_engine.FIELDS) for row in rows),
                  "all128_demosaic_and_gray_calls": totals["demosaic_calls"] == totals["gray_calls"] == 128,
                  "accepted_R104_R141_restored_and_closed": histogram_trace["aliases_restored"] and workspace_trace[
                      "original_builder_restored"] and workspace["closed"] and workspace["owned_plane_bytes_after_close"] == 0,
                  "accepted_R141_all384_native_planes": workspace["calls"] == 384,
                  "all128_original_cache_files": len(list((output / "quality").glob("*.json"))) == 128,
                  "original_sources_unchanged": sources == [data_identity(path) for path in paths],
                  "original_RAW_unchanged": (raw_stat.st_size, raw_stat.st_mtime_ns) == (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns)}
        atomic_json(output / "SCORES.json", {"rows": rows, "counted": False})
        if not all(checks.values()):
            raise ValueError("A complete original RAW diagnostic contract failed")
        state.update(status="cost_and_native_equivalence_diagnostic_complete", sources=sources, checks=checks,
                     complete_cold_RAW128_seconds=complete, worker_detail=detail, conversion_worker_totals=totals,
                     worker_conversion_sum_not_wall_clock_share=True, native_direct_gray_exact=all(row["exact"] for row in comparisons),
                     native_direct_real_six_fields_exact=all(row["six_fields_exact"] for row in comparisons if row["real_RAW"]),
                     generic_cases=32, real_equivalence_cases=16, workspace=workspace_trace, histogram=histogram_trace,
                     OpenCV=cv2.__version__, NumPy=np.__version__, node_before_publication_seconds=time.perf_counter() - started,
                     full7338_entry_formal_pairs_new_algorithm_and_new_training_not_run=True, counted=False)
        atomic_json(output / "RESULT.json", state)
        print({key: state[key] for key in ("status", "complete_cold_RAW128_seconds", "conversion_worker_totals",
                                          "native_direct_gray_exact", "native_direct_real_six_fields_exact", "accepted_count_contribution")}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        NativeQualityWorkspace.gray = old_gray
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"original_class_method_restored": NativeQualityWorkspace.gray is old_gray,
            "OpenCV_threads_restored": cv2.getNumThreads() == old_threads, "all_owned_pool_lifetimes_completed": True,
            "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
