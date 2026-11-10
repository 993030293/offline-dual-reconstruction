"""Retain original resource checks while publishing immutable status snapshots."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity

_JOURNALS = {}


def atomic_json(path, payload):
    path = Path(path)
    body = json.dumps(payload, indent=2, allow_nan=False)+"\n"
    state = _JOURNALS.setdefault(path, {"next": 0, "latest": None})
    history = path.parent/(path.stem+"_history")
    history.mkdir(exist_ok=True)
    snapshot = history/(f"sample_{state['next']:06d}.json")
    with snapshot.open("x", encoding="utf-8") as stream:
        stream.write(body)
    state.update(next=state["next"]+1, latest=body)


def finalize_journals():
    for path, state in _JOURNALS.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(state["latest"])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, required=True)
    parser.add_argument("--maximum-rss-gib", type=float, default=32.)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.receipt.exists() or args.log.exists() or args.maximum_seconds <= 0:
        raise ValueError("A bounded task requires a command, positive budget and unique receipts")
    started = time.monotonic()
    tracked, historical_peaks = {}, {}
    state = {"status": "starting", "command": command, "source": data_identity(Path(__file__)),
        "maximum_seconds": args.maximum_seconds, "maximum_RSS_bytes": int(args.maximum_rss_gib*1024**3),
        "peak_tree_RSS_bytes": 0, "counted": False}
    atomic_json(args.receipt, state)
    with args.log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", child_pid=process.pid)
        atomic_json(args.receipt, state)
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    for item in [parent]+parent.children(recursive=True):
                        tracked[item.pid] = item.create_time()
                except psutil.NoSuchProcess:
                    pass
                current = 0
                for pid, born in tracked.items():
                    try:
                        item = psutil.Process(pid)
                        if item.create_time() != born:
                            continue
                        memory = item.memory_info()
                        current += memory.rss
                        historical_peaks[pid] = max(historical_peaks.get(pid, 0), memory.rss,
                                                    getattr(memory, "peak_wset", memory.rss))
                    except psutil.NoSuchProcess:
                        continue
                state.update(peak_tree_RSS_bytes=max(state["peak_tree_RSS_bytes"], current),
                    historical_per_process_peak_RSS_bytes=historical_peaks,
                    elapsed_seconds=time.monotonic()-started,
                    owned_process_identities=[{"pid": pid, "creation_time": born} for pid, born in tracked.items()])
                atomic_json(args.receipt, state)
                if state["elapsed_seconds"] > args.maximum_seconds or current > state["maximum_RSS_bytes"]:
                    raise RuntimeError("The registered task time or process-tree RSS budget was exceeded")
                time.sleep(.5)
            state.update(status="completed" if process.returncode == 0 else "failed_child",
                returncode=process.returncode, elapsed_seconds=time.monotonic()-started)
            atomic_json(args.receipt, state)
        except BaseException as error:
            owned = []
            for pid, born in reversed(list(tracked.items())):
                try:
                    item = psutil.Process(pid)
                    if item.create_time() == born:
                        item.kill()
                        owned.append(item)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(owned, timeout=10)
            if process.poll() is None:
                process.kill()
            process.wait()
            state.update(status="failed_controller", error=repr(error), returncode=process.returncode,
                         elapsed_seconds=time.monotonic()-started)
            atomic_json(args.receipt, state)
            raise
    print(json.dumps({key: state[key] for key in ("status", "returncode", "elapsed_seconds", "peak_tree_RSS_bytes")}))
    if process.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_journals()
