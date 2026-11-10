"""Compare exact feature access with independent eager and bounded-LRU processes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

from runtime import atomic_json, data_identity
from sift_feature_store import SiftFeatureStore


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    trace = json.loads(args.trace.read_text())
    database_id = data_identity(args.database)
    if database_id["sha256"] != trace["database"]["sha256"] or trace["query_count"] != 24:
        raise ValueError("The complete real localization trace and feature database differ")
    start = time.monotonic()
    store = SiftFeatureStore(args.database, set(trace["registered_names"]), args.worker, 32)
    load_seconds = time.monotonic()-start
    process = psutil.Process()
    rss_after_load = process.memory_info().rss
    rows, access_seconds = [], 0.
    for image_id in trace["access_image_ids"]:
        before = time.monotonic()
        name, keypoints, descriptors = store[image_id]
        access_seconds += time.monotonic()-before
        row = {"image_id": image_id, "name": name}
        for label, array in (("keypoints", keypoints), ("descriptors", descriptors)):
            row[label] = {"shape": list(array.shape), "strides": list(array.strides), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        rows.append(row)
    rss_after_access = process.memory_info().rss
    memory = process.memory_info()
    peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss))
    storage = store.receipt()
    store.close()
    elapsed = time.monotonic()-start
    if elapsed > 300 or peak > 32*1024**3 or database_id != data_identity(args.database):
        raise ValueError("The feature benchmark exceeded budgets or changed its immutable database")
    result = {"status": "all_actual_requests_completed", "mode": args.worker,
        "database": database_id, "trace": data_identity(args.trace), "rows": rows, "store": storage,
        "load_seconds": load_seconds, "access_seconds": access_seconds,
        "load_and_access_seconds": load_seconds+access_seconds, "elapsed_seconds": elapsed,
        "RSS_after_load_bytes": rss_after_load, "RSS_after_access_bytes": rss_after_access, "peak_RSS_bytes": peak,
        "array_hashing_in_timed_access": False, "native_matching_in_this_microbenchmark": False}
    atomic_json(args.output/"RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "trace", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--worker", choices=["eager", "lazy"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    for name in ("benchmark_sift_store.py", "sift_feature_store.py", "runtime.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshot/name)
    state = {"status": "running", "candidate": "P27_LAZY_SIFT_FEATURES", "pairs": [],
        "orders": [["eager", "lazy"], ["lazy", "eager"], ["eager", "lazy"]], "counted": False,
        "sources": [data_identity(p) for p in sorted(snapshot.glob("*.py"))]}
    atomic_json(args.output/"STATE.json", state)
    try:
        for index, order in enumerate(state["orders"]):
            pair = {}
            for mode in order:
                output = args.output/f"trial{index}_{mode}"
                command = [sys.executable, str(snapshot/Path(__file__).name), "--database", str(args.database),
                    "--trace", str(args.trace), "--output", str(output), "--worker", mode]
                with (args.output/f"trial{index}_{mode}.log").open("x") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                pair[mode] = json.loads((output/"RESULT.json").read_text())
            pair["all_actual_request_arrays_exact"] = pair["eager"]["rows"] == pair["lazy"]["rows"]
            if not pair["all_actual_request_arrays_exact"] or pair["lazy"]["store"]["maximum_cached_images"] > 32:
                raise ValueError("Exact-array or bounded-cache guard failed")
            state["pairs"].append(pair)
            atomic_json(args.output/"STATE.json", state)
        rss = {mode: statistics.median(p[mode]["RSS_after_access_bytes"] for p in state["pairs"]) for mode in ("eager", "lazy")}
        times = {mode: statistics.median(p[mode]["load_and_access_seconds"] for p in state["pairs"]) for mode in rss}
        reduction, ratio = 1-rss["lazy"]/rss["eager"], times["lazy"]/times["eager"]
        state.update(status="resource_and_equivalence_passed_real_matching_pending" if reduction >= .2 and ratio <= 1.10 else "rejected_resource_guard",
            median_RSS_after_access_bytes=rss, RSS_reduction=reduction,
            median_load_and_access_seconds=times, median_time_ratio=ratio,
            real_matching_comparison_completed=False, total_SfM_speed_claim=False)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({k: state[k] for k in ("status", "RSS_reduction", "median_time_ratio")}))
    except BaseException as error:
        state.update(status="failed_benchmark", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise


if __name__ == "__main__":
    main()
