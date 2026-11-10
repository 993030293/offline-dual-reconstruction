"""Observe one unchanged native SIFT worker at its actual Linux PID."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    args = parser.parse_args()
    expected = Path("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/fused_integer_match_worker.py")
    if args.native.resolve() != expected.resolve():
        raise ValueError("Only the current pinned exact integer worker is permitted")
    process = psutil.Process()
    started = time.monotonic()
    stopped = threading.Event()
    state = {"pid": process.pid, "creation_time": process.create_time(), "native_source": str(args.native),
             "wrapper_source": str(Path(__file__).resolve()), "maximum_RSS_bytes": 32*1024**3,
             "maximum_seconds": 1750, "peak_RSS_bytes": 0, "samples": 0,
             "status": "running", "closed": False, "counted": False}
    args.resource.parent.mkdir(parents=True, exist_ok=True)

    def sample():
        state["samples"] += 1
        state["peak_RSS_bytes"] = max(state["peak_RSS_bytes"], process.memory_info().rss)
        state["elapsed_seconds"] = time.monotonic()-started
        if state["peak_RSS_bytes"] > state["maximum_RSS_bytes"] or state["elapsed_seconds"] > 1750:
            state["status"] = "failed_original_resource_budget"
        temporary = args.resource.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, args.resource)
        if state["status"] == "failed_original_resource_budget":
            os._exit(70)

    def monitor():
        while not stopped.wait(.2):
            try:
                sample()
            except BaseException as error:
                print("Owned Linux RSS observation failed: "+repr(error), file=sys.stderr, flush=True)
                os._exit(71)

    sample()
    watcher = threading.Thread(target=monitor, name="owned-native-matching-rss", daemon=True)
    watcher.start()
    sys.argv = [str(args.native)]
    try:
        runpy.run_path(str(args.native), run_name="__main__")
        state["status"] = "native_worker_returned"
    except BaseException as error:
        state.update(status="native_worker_failed", error=repr(error))
        raise
    finally:
        stopped.set()
        watcher.join()
        state["closed"] = True
        sample()


if __name__ == "__main__":
    main()
