"""Stage completed fresh models without touching live training or existing assets."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from atomic_json_windows import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--maximum-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = args.campaign / ("viewer_build_watch_" + str(time.time_ns()))
    root.mkdir(exist_ok=False)
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
             "branches": {}, "errors": {}, "native_browser_packing_and_visual_review_separately_required": True}
    sides = [(row["case"], mode) for row in read_json(args.campaign / "CASES.json")["cases"]
             for mode in ("baseline", "optimized")]
    started = time.monotonic()
    while time.monotonic() - started < args.maximum_seconds:
        for case, mode in sides:
            branch = case + "-" + mode
            if branch in state["branches"] or branch in state["errors"]:
                continue
            side = args.campaign / case / mode
            staged = args.campaign / "viewer" / branch / "STAGING_RECEIPT.json"
            if staged.exists():
                existing = read_json(staged)
                state["branches"][branch] = {"status": "already_staged_this_campaign", "receipt": str(staged),
                                            "source_model": existing["metadata"]["source_model"]}
                continue
            receipt, model = side / "training_process.json", side / "training/STATE.json"
            if not receipt.exists() or not model.exists():
                continue
            measured, trained = read_json(receipt), read_json(model)
            if measured["status"] != "completed" or trained["status"] != "completed":
                continue
            command = [sys.executable, str(Path(__file__).with_name("run_owned_task.py")),
                "--receipt", str(root / (branch + ".json")), "--log", str(root / (branch + ".log")),
                "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--", sys.executable,
                str(Path(__file__).with_name("build_raw_viewer.py")), "--campaign", str(args.campaign),
                "--case", case, "--mode", mode]
            result = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            if result.returncode == 0 and staged.exists():
                state["branches"][branch] = {"status": "staged_native_browser_packing_pending", "receipt": str(staged)}
            else:
                state["errors"][branch] = {"returncode": result.returncode, "stderr": result.stderr[-4096:],
                                            "partial_assets_retained": True}
        state["elapsed_seconds"] = time.monotonic() - started
        if len(state["branches"]) + len(state["errors"]) == len(sides):
            state["status"] = "completed" if not state["errors"] else "completed_with_failures"
            atomic_json(root / "STATE.json", state)
            return
        atomic_json(root / "STATE.json", state)
        time.sleep(10)
    state["status"] = "wait_budget_exhausted"
    atomic_json(root / "STATE.json", state)


if __name__ == "__main__":
    main()
