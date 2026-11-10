"""Wait for the CPU diagnostic to drain, repeat mapping, then continue one raw side."""

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
    parser.add_argument("--mode", required=True, choices=("baseline", "optimized"))
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--wait-for", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    side = args.campaign / args.case / args.mode
    code = Path(__file__).resolve().parent
    state = {"status": "waiting_for_CPU_diagnostic_drain", "pid": os.getpid(),
             "process_birth": psutil.Process().create_time(), "wait_for": str(args.wait_for), "CPU_threads": 8}
    atomic_json(args.output / "STATE.json", state)
    started = time.monotonic()
    while not any((args.wait_for / name).exists() for name in ("RESULT.json", "FAILURE.json")):
        if time.monotonic() - started > 3600:
            raise TimeoutError("The independent CPU diagnostic has not reached its bounded terminal result")
        time.sleep(2)
    for path in args.wait_for.rglob("PROCESS.json"):
        receipt = read_json(path)
        if receipt["status"] in ("starting", "running"):
            raise RuntimeError("A CPU diagnostic child is still active")
        for row in receipt.get("owned_process_identities", []):
            try:
                if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                    raise RuntimeError("An exact CPU diagnostic child remains alive after its terminal marker")
            except psutil.NoSuchProcess:
                pass
    state.update(status="repeating_native_mapping", queue_wait_seconds=time.monotonic() - started)
    atomic_json(args.output / "STATE.json", state)
    command = [sys.executable, str(code / "run_owned_task.py"), "--receipt", str(side / "sfm_process.json"),
        "--log", str(side / "sfm.log"), "--maximum-seconds", "4200", "--maximum-rss-gib", "32", "--",
        sys.executable, str(code / "resume_sfm_after_monitor.py"), "--initial", str(side / "initial"),
        "--prior", str(args.prior), "--output", str(side / "sfm"), "--mode", args.mode]
    with (args.output / "mapping_controller.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    state.update(mapping_returncode=result.returncode)
    if result.returncode:
        state["status"] = "failed_repeated_mapping"
        atomic_json(args.output / "STATE.json", state)
        return
    state["status"] = "continuing_own_raw_pipeline"
    atomic_json(args.output / "STATE.json", state)
    command = [sys.executable, str(code / "run_side.py"), "--case", args.case, "--mode", args.mode,
               "--campaign", str(args.campaign)]
    with (args.output / "pipeline_controller.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    state.update(status="completed" if result.returncode == 0 else "failed_pipeline_continuation",
                 pipeline_returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
