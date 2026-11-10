"""Copy and independently read two owned NTFS files in one bounded host process."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    allowed = (ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001").resolve()
    for path in (args.source, args.target, args.receipt):
        path.resolve().relative_to(allowed)
    if os.name != "nt" or args.source.is_symlink() or not args.source.is_file():
        raise ValueError("This host worker only accepts owned ordinary NTFS files")
    if args.source.stat().st_size > 256 * 1024**2 or args.target.exists() or args.receipt.exists():
        raise ValueError("The original byte budget and new-file ownership are required")
    process = psutil.Process()
    row = {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time(),
        "status": "running", "maximum_seconds": 120, "maximum_RSS_bytes": 8 * 1024**3,
        "peak_RSS_bytes": 0, "counted": False}
    stop = threading.Event()

    def sample():
        memory = process.memory_info()
        row["peak_RSS_bytes"] = max(row["peak_RSS_bytes"], memory.rss, getattr(memory, "peak_wset", memory.rss))
        row["elapsed_seconds"] = time.perf_counter() - started
        if row["peak_RSS_bytes"] > row["maximum_RSS_bytes"] or row["elapsed_seconds"] > 120:
            raise RuntimeError("The original host worker resource budget was exceeded")

    def watch():
        try:
            while not stop.wait(.02):
                sample()
        except BaseException as error:
            row.update(status="failed_resource_guard", error=repr(error))
            atomic_json(args.receipt, row)
            os._exit(73)

    sample()
    atomic_json(args.receipt, row)
    observer = threading.Thread(target=watch, name="owned-host-file-RSS", daemon=True)
    observer.start()
    try:
        copy_started = time.perf_counter()
        shutil.copyfile(args.source, args.target)
        row["copy_seconds"] = time.perf_counter() - copy_started
        read_started = time.perf_counter()
        actual, expected = data_identity(args.source), data_identity(args.target)
        row["two_complete_identity_seconds"] = time.perf_counter() - read_started
        row.update(status="completed", independent_file=not os.path.samefile(args.source, args.target),
            source_identity=actual, target_identity=expected)
    except BaseException as error:
        row.update(status="failed_operation", error_type=type(error).__name__, error=str(error),
            errno=getattr(error, "errno", None), winerror=getattr(error, "winerror", None))
        raise
    finally:
        stop.set()
        observer.join(timeout=10)
        if observer.is_alive():
            raise RuntimeError("The owned host resource observer did not join")
        sample()
        atomic_json(args.receipt, row)
    print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
