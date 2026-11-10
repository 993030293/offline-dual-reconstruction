"""Retry two archived failed training stages while retaining completed raw inputs."""

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
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
             "case": args.case, "rows": [], "original_failures_retained": True}
    atomic_json(args.output / "STATE.json", state)
    for mode in ("baseline", "optimized"):
        side = args.campaign / args.case / mode
        previous = read_json(side / "PIPELINE_STATE.json")
        if previous["status"] != "failed_pipeline_stage" or previous["current_stage"] != "training":
            raise ValueError("Only the explicit training-stage failure can be retried here")
        try:
            if psutil.Process(previous["pid"]).create_time() == previous["process_birth"]:
                raise RuntimeError("The previous side controller is still alive")
        except psutil.NoSuchProcess:
            pass
        if (side / "training_process.json").exists() or (side / "training").exists():
            raise ValueError("Archive the verified failed training outputs before retry")
        atomic_json(args.output / (mode + "_ORIGINAL_PIPELINE_FAILURE.json"), previous)
        started = time.monotonic()
        state["current_mode"] = mode
        atomic_json(args.output / "STATE.json", state)
        command = [sys.executable, str(Path(__file__).with_name("run_side.py")), "--campaign", str(args.campaign),
                   "--case", args.case, "--mode", mode, "--render-near-recovery"]
        with (args.output / (mode + ".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        completed = read_json(side / "PIPELINE_STATE.json")
        state["rows"].append({"mode": mode, "returncode": result.returncode, "status": completed["status"],
                             "elapsed_seconds": time.monotonic() - started})
        atomic_json(args.output / "STATE.json", state)
    state["status"] = "completed" if all(row["returncode"] == 0 for row in state["rows"]) else "completed_with_failures"
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
