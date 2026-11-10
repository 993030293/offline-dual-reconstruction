"""Run the current and candidate complete Box1 pipelines strictly sequentially."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--playwright-package", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "PAIR_STATE.json"
    if path.exists():
        raise FileExistsError("A pair campaign requires a new independent state path")
    state = {"status": "running", "pid": os.getpid(), "birth": psutil.Process().create_time(),
             "sides": {}, "strictly_sequential": True, "start_unix": time.time()}
    started = time.monotonic()

    def publish():
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(path, state)

    try:
        for route, port in [("reference", 8224), ("sparse", 8225)]:
            destination = args.output / ("formal_"+route+"_001")
            receipt = args.output / ("formal_"+route+"_001_process.json")
            state["current_route"] = route
            publish()
            command = [sys.executable, str(OLD / "run_owned_task.py"), "--receipt", str(receipt),
                "--log", str(args.output / ("formal_"+route+"_001.log")), "--maximum-seconds", "15500",
                "--maximum-rss-gib", "32", "--", sys.executable, str(CODE / "run_full_case.py"),
                "--campaign", str(destination), "--route", route, "--port", str(port),
                "--node", str(args.node), "--playwright-package", str(args.playwright_package)]
            process = subprocess.Popen(command, env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            state["active_child"] = {"pid": process.pid, "birth": psutil.Process(process.pid).create_time(), "command": command}
            publish()
            code = process.wait()
            state["sides"][route] = {"returncode": code, "output": str(destination), "receipt": str(receipt)}
            publish()
            if code or not (destination / "RESULT.json").exists():
                raise RuntimeError("A full side did not finish; preserve all costs and inspect "+route)
        state["status"] = "both_full_raw_runs_complete_quality_and_time_acceptance_pending"
        state.pop("active_child", None)
        publish()
    except BaseException:
        state.update(status="failed_pair_campaign", traceback=traceback.format_exc())
        publish()
        raise


if __name__ == "__main__":
    main()
