"""Check original cache bytes, bounded submissions, real errors and full shutdown."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import quality_engine as native
import run_initial_pipeline as entry
from quality_persistence_pipeline import QualityWriter, quality_persistence_execution


def scores(value):
    return {"sharpness": value*value/9, "clipped_fraction": value % 17/17, "mean_brightness": float(value % 256),
        "corner_count_half_resolution": value % 800, "occupied_grid_cells_4x4": value % 16,
        "quality_score": math.log1p(value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("quality_persistence_pipeline.py"), V1/"quality_engine.py", V1/"run_initial_pipeline.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "all_cache_records_original_bytes_and_readback": True,
        "all_pending_writes_drained_before_complete_scoring_return": True,
        "queue64_not_a_frame_or_feature_limit": True, "object_GT_or_SfM_inputs": False,
        "actual_full_three_cost_pairs_pending": True})
    parity = []
    for count, capacity, producers in ((0, 1, 1), (1, 1, 1), (257, 4, 4)):
        folder = args.output/f"records_{count}"
        left = native.ScoreCache(folder/"native", {"contract": "generic_original_quality_bytes"})
        right = native.ScoreCache(folder/"pipeline", {"contract": "generic_original_quality_bytes"})
        inputs = [(hashlib.sha256(str(i).encode()).hexdigest(), scores(i)) for i in range(count)]
        for key, value in inputs:
            left.write(key, value)
        writer = QualityWriter(capacity)
        try:
            def submit(item):
                key, value = item
                copied = dict(value)
                writer.submit(native.ScoreCache.write, right, key, copied)
                copied["quality_score"] = -123456.
            with ThreadPoolExecutor(max_workers=producers) as pool:
                list(pool.map(submit, inputs))
        finally:
            writer.close()
        closed_rejects = False
        try:
            writer.submit(native.ScoreCache.write, right, "a"*64, scores(0))
        except RuntimeError:
            closed_rejects = True
        writer.close()
        names = sorted(path.name for path in left.directory.iterdir())
        parity.append({"count": count, "trace": writer.trace,
            "complete_original_files_and_dict_snapshots_exact": names == sorted(path.name for path in right.directory.iterdir())
                and all((left.directory/name).read_bytes() == (right.directory/name).read_bytes() for name in names)
                and all(right.read(key) == value for key, value in inputs),
            "queue_and_complete_close": writer.trace["maximum_pending"] <= capacity
                and writer.trace["submitted"] == writer.trace["finished"] == writer.trace["committed"] == count
                and writer.trace["closed"] and closed_rejects})
    faulty = native.ScoreCache(args.output/"faulty", {"contract": "original_JSON_error"})
    bad = scores(0)
    bad["quality_score"] = float("nan")
    try:
        faulty.write("b"*64, bad)
    except ValueError as error:
        original_error = str(error)
    writer = QualityWriter(4)
    writer.submit(native.ScoreCache.write, faulty, "b"*64, bad)
    try:
        writer.close()
    except ValueError as error:
        failure = str(error) == original_error and writer.trace["closed"] and writer.trace["committed"] == 0
    else:
        failure = False
    messages = [(i, {"payload": bytes([i])}) for i in range(64)]
    def decode(payload):
        return int(payload[0])*1000000, "generic_optical", np.full((8, 8), payload[0], np.uint8)
    def demosaic(raw):
        return np.repeat(raw[:, :, None], 3, axis=2)
    original_score, original_write = native.score_messages, native.ScoreCache.write
    caches = [native.ScoreCache(args.output/name, {"contract": "complete_scoring_boundary"}) for name in ("scoring_native", "scoring_pipeline")]
    left_rows, _ = native.score_messages(messages, decode, demosaic, native.compact_quality, workers=4, capacity=16, cache=caches[0])
    with quality_persistence_execution(True, caches[1].directory) as trace:
        right_rows, _ = entry.score_messages(messages, decode, demosaic, native.compact_quality, workers=4, capacity=16, cache=caches[1])
        boundary_closed = trace["calls"][0]["closed"] and trace["calls"][0]["committed"] == 64
        warm, _ = entry.score_messages(messages, decode, demosaic, native.compact_quality, workers=4, capacity=16, cache=caches[1])
    def relevant(rows):
        return [{key: row[key] for key in ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality")} for row in rows]
    names = sorted(path.name for path in caches[0].directory.iterdir())
    complete_scoring = relevant(left_rows) == relevant(right_rows) == relevant(warm)
    complete_scoring = complete_scoring and names == sorted(path.name for path in caches[1].directory.iterdir())
    complete_scoring = complete_scoring and all((caches[0].directory/name).read_bytes() == (caches[1].directory/name).read_bytes() for name in names)
    error_cache = native.ScoreCache(args.output/"consumer_error", {"contract": "consumer_failure"})
    def broken(payload):
        if payload[0] == 5:
            raise ValueError("injected_raw_decode_failure")
        return decode(payload)
    try:
        with quality_persistence_execution(True, error_cache.directory) as failed_trace:
            entry.score_messages(messages, broken, demosaic, native.compact_quality, workers=4, capacity=16, cache=error_cache)
    except ValueError as error:
        consumer_failure = str(error) == "injected_raw_decode_failure"
    else:
        consumer_failure = False
    checks = {"all3_empty_single_concurrent_native_cache_byte_and_dict_isolation_oracles": len(parity) == 3
            and all(row["complete_original_files_and_dict_snapshots_exact"] and row["queue_and_complete_close"] for row in parity),
        "real_original_nonfinite_JSON_write_error_preserved_and_closed": failure,
        "full64_original_cold_and_warm_scores_metadata_and_cache_bytes_exact": complete_scoring,
        "all64_writes_complete_before_cold_score_returns_and_warm_has_no_writer_work": boundary_closed
            and len(trace["calls"]) == 2 and trace["calls"][1]["submitted"] == 0 and trace["calls"][1]["closed"],
        "consumer_failure_preserved_all_pending_writes_and_threads_drained": consumer_failure
            and failed_trace["calls"][0]["closed"] and trace["original_functions_restored"] and failed_trace["original_functions_restored"],
        "all_original_functions_restored": native.score_messages is original_score and entry.score_messages is original_score
            and native.ScoreCache.write is original_write,
        "no_owned_writer_threads_after_all_closes": not any(thread.name.startswith("owned-quality-write") for thread in threading.enumerate()),
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks, "sources": sources,
        "parity": parity, "fault_trace": writer.trace, "score_trace": trace, "consumer_failure_trace": failed_trace,
        "counted": False, "full_actual_cold_entry_three_cost_pairs_and_new_object_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original bounded durable quality writer contract failed")


if __name__ == "__main__":
    main()
