"""Run a prespecified, equal-output CPU scoring and cache comparison."""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import cv2
import numpy as np
import psutil

from quality_engine import compact_quality, score_messages, ScoreCache, FIELDS
from runtime import atomic_json, data_identity


def assert_scores(reference, candidate):
    if [row["camera_frame"] for row in reference] != [row["camera_frame"] for row in candidate]:
        raise AssertionError("Chronological frame identity changed")
    maximum = {field: 0. for field in FIELDS}
    for old, new in zip(reference, candidate):
        for field in FIELDS:
            a, b = old["quality"][field], new["quality"][field]
            maximum[field] = max(maximum[field], abs(a-b))
            if not np.isclose(a, b, rtol=1e-12, atol=1e-10):
                raise AssertionError(f"Quality changed for frame {old['camera_frame']}, {field}: {a}, {b}")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.frames != 128 or args.repetitions != 3:
        raise ValueError("This registered comparison uses 128 frames and 3 repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    from mcap_audit import McapIndex
    decisions = project/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/frame_decisions.jsonl"
    all_rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    threshold_path = decisions.with_name("SELECTION_RECEIPT.json")
    sharpness_min = json.loads(threshold_path.read_text(encoding="utf-8"))["sharpness_min"]
    usable = [row for row in all_rows if not row.get("excluded_role")]
    selected = [usable[i] for i in np.linspace(0, len(usable)-1, args.frames, dtype=int)]
    ids = [row["camera_frame"] for row in selected]
    variants = [("baseline_w1_cv4", 1, 4, "baseline"), ("baseline_w1_cv1", 1, 1, "baseline"),
                ("baseline_w4_cv1", 4, 1, "baseline"), ("baseline_w8_cv1", 8, 1, "baseline"),
                ("compact_w1_cv1", 1, 1, "compact"), ("compact_w4_cv1", 4, 1, "compact")]
    sources = [Path(__file__), Path(__file__).with_name("quality_engine.py"), Path(baseline.__file__),
               baseline.TOOLS/"camera_motion.py", baseline.TOOLS/"color_contract.py", baseline.TOOLS/"mcap_audit.py"]
    source_identity = [data_identity(path) for path in sources]
    raw_stat = baseline.RAW.stat()
    contract = {"schema": 1, "frames": ids, "selection": "uniform_non_HD_raw_frame_indices_no_SfM_or_GT",
        "frozen_historical_sharpness_threshold": sharpness_min,
        "threshold_identity": data_identity(threshold_path),
        "source": source_identity, "opencv": cv2.__version__, "numpy": np.__version__,
        "raw": {"path": str(baseline.RAW), "bytes": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns},
        "variants": variants, "repetitions": 3, "maximum_seconds_per_configuration": 300,
        "rss_budget_bytes": 32*1024**3, "pending_payload_limit": 16,
        "equivalence": {"rtol": 1e-12, "atol": 1e-10, "same_frame_and_selection_ids": True},
        "speed_gate": "at_least_10_percent_median_stage_wall_reduction",
        "cache_state": "OS cache uncontrolled; first access separate, repeated trials warm",
        "training_parallel": False, "score_cache_scope": "image only; never cache pose-dependent decisions"}
    atomic_json(args.output/"CONTRACT.json", contract)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
    cv2.setNumThreads(4)
    before = time.perf_counter()
    index = McapIndex(baseline.RAW)
    index_seconds = time.perf_counter()-before
    cache_contract = {"color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__, "numpy": np.__version__,
                      "source": [item["sha256"] for item in source_identity], "scorer": "compact"}
    cache = ScoreCache(args.output/"quality_cache", cache_contract)
    process = psutil.Process()
    records = []

    def run(label, workers, threads, scorer_name, repetition, use_cache=False):
        cv2.setNumThreads(threads)
        scorer = baseline.quality if scorer_name == "baseline" else compact_quality
        before = time.perf_counter()
        rows, timings = score_messages(index.iter_selected_messages(baseline.CAMERA_TOPIC, ids),
            decode_bayer, demosaic_rggb8_to_bgr, scorer, workers=workers, capacity=16,
            cache=cache if use_cache else None)
        elapsed = time.perf_counter()-before
        memory = process.memory_info()
        peak = getattr(memory, "peak_wset", memory.rss)
        if elapsed > 300 or peak > 32*1024**3:
            raise RuntimeError("Registered time or RAM budget exceeded")
        reference = [{"camera_frame": row["camera_frame"], "quality": row["quality"]} for row in selected]
        maximum = assert_scores(reference, rows)
        choice_reference, choice_new = copy.deepcopy(selected), copy.deepcopy(selected)
        for row, scores in zip(choice_new, rows):
            row["quality"] = scores["quality"]
        old_selection = baseline.choose(choice_reference, baseline.POLICY, contract["frozen_historical_sharpness_threshold"])
        new_selection = baseline.choose(choice_new, baseline.POLICY, contract["frozen_historical_sharpness_threshold"])
        old_ids = [row["camera_frame"] for row in old_selection[0]]
        new_ids = [row["camera_frame"] for row in new_selection[0]]
        if old_ids != new_ids:
            raise AssertionError("The fixed-policy diagnostic selection changed")
        result = {"variant": label, "repetition": repetition, "wall_seconds": elapsed,
                  "peak_rss_bytes": peak, "timings": timings, "maximum_absolute_score_differences": maximum,
                  "selected_ids": new_ids, "selection_is_128_frame_diagnostic": True}
        records.append(result)
        atomic_json(args.output/"PROGRESS.json", {"status": "running", "trials": records})
        print(json.dumps({"variant": label, "repetition": repetition, "seconds": elapsed,
                          "cache_hits": timings["cache_hits"]}), flush=True)
        return rows

    run(*variants[0], repetition="first_access_not_cold_guaranteed")
    for repetition in range(3):
        offset = repetition*2
        for variant in variants[offset:]+variants[:offset]:
            run(*variant, repetition=repetition)
    run("cache_fill_w4_cv1", 4, 1, "compact", "fill", True)
    for repetition in range(3):
        run("cache_hit_w4_cv1", 4, 1, "compact", repetition, True)
    summary = {}
    for label in [variant[0] for variant in variants]+["cache_hit_w4_cv1"]:
        values = [row["wall_seconds"] for row in records if row["variant"] == label and isinstance(row["repetition"], int)]
        summary[label] = {"values_seconds": values, "median_seconds": statistics.median(values)}
    pairs = {"parallel_scoring": ("baseline_w1_cv4", "baseline_w4_cv1"),
             "compact_derivative": ("baseline_w1_cv1", "compact_w1_cv1"),
             "content_keyed_score_reuse": ("compact_w4_cv1", "cache_hit_w4_cv1")}
    comparisons = {name: {"baseline": a, "candidate": b,
        "wall_reduction_fraction": 1-summary[b]["median_seconds"]/summary[a]["median_seconds"]}
        for name, (a, b) in pairs.items()}
    for value in comparisons.values():
        value["stage_speed_gate_passed"] = value["wall_reduction_fraction"] >= .1
        value["full_sequence_integration_pending"] = True
        value["counted"] = False
    atomic_json(args.output/"RESULT.json", {"status": "128_frame_profiling_completed",
        "index_seconds": index_seconds, "trials": records, "summary": summary, "comparisons": comparisons,
        "end_to_end_pipeline_speed_claim": False, "strict_blind_test": False})
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
