"""Check reader ownership, ordered real scoring, errors, early close and restoration."""

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
import quality_engine
import run_initial_pipeline
from raw_reader_prefetch import OwnedRawPrefetch, raw_reader_prefetch_execution


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWN.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), CODE / "raw_reader_prefetch.py", Path(quality_engine.__file__), Path(run_initial_pipeline.__file__)]
    sources = [data_identity(path) for path in paths]
    original = quality_engine.score_messages
    checks, cases = {}, []
    source_threads, close_threads = [], []
    values = [object() for _ in range(19)]

    def source():
        try:
            for value in values:
                source_threads.append(threading.get_ident())
                yield value
        finally:
            close_threads.append(threading.get_ident())

    with OwnedRawPrefetch(source()) as producer:
        output = list(producer)
    checks["same_all19_objects_and_order_single_source_owner"] = all(a is b for a, b in zip(output, values)) and len(output) == 19
    checks["next_and_close_owned_by_same_separate_reader"] = set(source_threads) == set(close_threads) == {producer.trace["reader_thread"]}
    checks["normal_full_reader_queue_and_references_closed"] = producer.trace["closed"] and not producer.trace["reader_alive_after_join"] and producer.trace["queued_references_after_close"] == 0
    cases.append(producer.trace)
    with OwnedRawPrefetch(source()) as producer:
        iterator = iter(producer)
        next(iterator)
        for _ in range(100):
            if producer.trace["peak_queue_results"] == 2:
                break
            threading.Event().wait(.001)
        iterator.close()
    checks["early_close_full_queue_cancels_and_joins_source"] = producer.trace["peak_queue_results"] == 2 and producer.trace["closed"] and not producer.trace["reader_alive_after_join"]
    cases.append(producer.trace)
    for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
        message = "original source " + error_type.__name__

        def failed_source():
            yield values[0]
            raise error_type(message)

        got = None
        with OwnedRawPrefetch(failed_source()) as producer:
            try:
                list(producer)
            except BaseException as error:
                got = type(error), str(error)
        checks[error_type.__name__ + "_source_error_exact_and_reader_joined"] = got == (error_type, message) and not producer.trace["reader_alive_after_join"]
        cases.append(producer.trace)
        got = None
        try:
            with raw_reader_prefetch_execution() as trace:
                raise error_type("original consumer " + error_type.__name__)
        except BaseException as error:
            got = type(error), str(error)
        checks[error_type.__name__ + "_consumer_error_exact_and_aliases_restored"] = got == (error_type, "original consumer " + error_type.__name__) and trace["binding_restored"]
    rng = np.random.default_rng(143)
    frames = [(index, {"payload": rng.integers(0, 256, (64, 80), dtype=np.uint8).tobytes()}) for index in range(29)]

    def decode(payload):
        return len(payload), "camera_optical", np.frombuffer(payload, np.uint8).reshape(64, 80)

    def demosaic(raw):
        return np.repeat(raw[:, :, None], 3, axis=2)

    def run(enabled):
        with raw_reader_prefetch_execution(enabled) as trace:
            rows, detail = run_initial_pipeline.score_messages(iter(frames), decode, demosaic, quality_engine.compact_quality,
                workers=4, capacity=16, maximum_seconds=30)
        clean = [{key: value for key, value in row.items() if not key.endswith("seconds")} for row in rows]
        return clean, detail, trace

    a, before, _ = run(False)
    b, after, trace = run(True)
    checks["all29_original4worker16queue_real_quality_metadata_exact"] = a == b and before["submitted"] == after["submitted"] == 29 and before["queue_capacity"] == after["queue_capacity"] == 16
    checks["complete_scoring_actual_reader_closed"] = len(trace["calls"]) == 1 and trace["calls"][0]["delivered"] == 29 and trace["calls"][0]["source_closed_on_reader"] and not trace["calls"][0]["reader_alive_after_join"]
    checks["native_aliases_restored_and_sources_unchanged"] = quality_engine.score_messages is run_initial_pipeline.score_messages is original and sources == [data_identity(path) for path in paths]
    checks["no_owned_RAW_reader_threads_left"] = not any(thread.name == "owned-RAW-reader" for thread in threading.enumerate())
    result = {"status": "generic_original_scorer_passed_actual_RAW_pilot_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "cases": cases, "actual29_scoring_trace": trace,
        "own_process": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "elapsed_seconds": time.perf_counter() - began, "new_optimization_count": 0}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A private RAW reader contract failed")


if __name__ == "__main__":
    main()
