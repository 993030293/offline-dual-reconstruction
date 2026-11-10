"""Exercise repeated keys, blocked submission and simultaneous real failure cleanup."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from check_quality_persistence import scores
from quality_persistence_pipeline import QualityWriter, quality_persistence_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("quality_persistence_pipeline.py"),
        Path(__file__).with_name("check_quality_persistence.py"), V1/"quality_engine.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "faults_are_not_real_data": True,
        "all_threads_drained_on_both_primary_and_writer_errors": True,
        "repeated_keys_preserve_committed_bytes_not_transient_hit_timing": True, "counted": False})
    original_write, original_score = native.ScoreCache.write, native.score_messages
    caches = [native.ScoreCache(args.output/name, {"contract": "repeated_key_bytes"})
        for name in ("duplicate_native", "duplicate_pipeline")]
    key, value = caches[0].key(b"shared_image_payload"), scores(71)
    for _ in range(257):
        original_write(caches[0], key, value)
    writer = QualityWriter(4)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: writer.submit(original_write, caches[1], key, value), range(257)))
    finally:
        writer.close()
    repeated = (caches[0].read(key) == caches[1].read(key) == value
        and (caches[0].directory/(key+".json")).read_bytes() == (caches[1].directory/(key+".json")).read_bytes()
        and len(list(caches[1].directory.iterdir())) == 1 and writer.trace["committed"] == 257)

    cache = native.ScoreCache(args.output/"blocked", {"contract": "submit_close_race"})
    gate, entered, closing, waiting = (threading.Event() for _ in range(4))
    def gated_write(store, identity, quality):
        entered.set()
        if not gate.wait(5):
            raise TimeoutError("The bounded race fixture did not release its native writer")
        return original_write(store, identity, quality)
    blocked_writer = QualityWriter(1)
    blocked_writer.submit(gated_write, cache, "c"*64, scores(3))
    if not entered.wait(5):
        gate.set()
        blocked_writer.close()
        raise TimeoutError("The original writer did not enter its bounded race fixture")
    original_acquire = blocked_writer.slots.acquire
    def tracked_acquire(*values, **named):
        waiting.set()
        return original_acquire(*values, **named)
    blocked_writer.slots.acquire = tracked_acquire
    original_shutdown = blocked_writer.pool.shutdown
    def tracked_shutdown(*values, **named):
        closing.set()
        return original_shutdown(*values, **named)
    blocked_writer.pool.shutdown = tracked_shutdown
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            submit = pool.submit(blocked_writer.submit, original_write, cache, "d"*64, scores(4))
            if not waiting.wait(5):
                raise TimeoutError("The second submission did not reach the bounded queue")
            close = pool.submit(blocked_writer.close)
            if not closing.wait(5):
                raise TimeoutError("The bounded close fixture did not begin")
            gate.set()
            close.result(timeout=5)
            try:
                submit.result(timeout=5)
            except RuntimeError:
                race = blocked_writer.trace["closed"] and blocked_writer.trace["submitted"] == 1
            else:
                race = False
    finally:
        gate.set()
        blocked_writer.close()
    race = race and cache.read("c"*64) == scores(3) and not (cache.directory/("d"*64+".json")).exists()

    error_cache = native.ScoreCache(args.output/"both_errors", {"contract": "primary_and_real_JSON_error"})
    release = threading.Event()
    def pending_error(store, identity, quality):
        if not release.wait(5):
            raise TimeoutError("The bounded cleanup fixture did not release its original JSON write")
        return original_write(store, identity, quality)
    native.ScoreCache.write = pending_error
    def decode(payload):
        return 123, "generic_optical", np.zeros((8, 8), np.uint8)
    def nonfinite_scorer(image):
        value = scores(0)
        value["quality_score"] = float("nan")
        return value
    def failed_progress(count):
        release.set()
        raise ValueError("primary_consumer_failure")
    try:
        try:
            with quality_persistence_execution(True, error_cache.directory) as trace:
                entry.score_messages([(0, {"payload": b"input"})], decode, lambda raw: raw,
                    nonfinite_scorer, workers=1, capacity=1, cache=error_cache, progress=failed_progress)
        except ValueError as error:
            both = (str(error) == "primary_consumer_failure"
                and any("Owned quality persistence also failed:" in note for note in getattr(error, "__notes__", ()))
                and trace["calls"][0]["closed"] and trace["calls"][0]["committed"] == 0
                and trace["original_functions_restored"])
        else:
            both = False
    finally:
        release.set()
        native.ScoreCache.write = original_write
    checks = {"all257_repeated_keys_original_committed_bytes_and_readback_exact": repeated,
        "blocked_submit_cannot_commit_after_owned_close_and_no_deadlock": race,
        "original_primary_error_preserved_with_real_JSON_cleanup_error_note": both,
        "original_functions_restored_and_no_owned_writer_threads": native.ScoreCache.write is original_write
            and native.score_messages is original_score and entry.score_messages is original_score
            and not any(thread.name.startswith("owned-quality-write") for thread in threading.enumerate()),
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "sources": sources, "duplicate_trace": writer.trace, "race_trace": blocked_writer.trace,
        "dual_failure_trace": trace, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks}, flush=True)
    if not all(checks.values()):
        raise ValueError("An owned original-write edge contract failed")


if __name__ == "__main__":
    main()
