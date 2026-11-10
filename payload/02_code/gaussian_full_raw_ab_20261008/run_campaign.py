"""Run all raw comparison sides sequentially without shared output or GPU contention."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

V1 = Path(__file__).resolve().parents[2] / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["box2", "box1", "c1", "c0_july"])
    args = parser.parse_args()
    launch = args.campaign / ("campaign_controller_" + str(time.time_ns()))
    launch.mkdir()
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
             "source": data_identity(Path(__file__)), "raw_case_order": args.cases,
             "resource_policy": "one pipeline side at a time; unrelated system activity recorded separately",
             "sides": [], "evaluation_and_module_audit_still_required": True}
    state_path = args.campaign / "CAMPAIGN_STATE.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old["status"] == "running":
            try:
                previous = psutil.Process(old["pid"])
                if previous.create_time() == old["process_birth"]:
                    raise RuntimeError("An active comparison campaign must not be duplicated")
            except psutil.NoSuchProcess:
                pass
    atomic_json(launch / "CONTRACT.json", state)
    started = time.monotonic()
    for case in args.cases:
        for mode in ("baseline", "optimized"):
            state.update(current_case=case, current_mode=mode, elapsed_seconds=time.monotonic()-started)
            atomic_json(state_path, state)
            command = [sys.executable, str(Path(__file__).with_name("run_side.py")), "--case", case,
                       "--mode", mode, "--campaign", str(args.campaign)]
            log_path = launch / (case + "_" + mode + ".log")
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            receipt_path = args.campaign / case / mode / "PIPELINE_STATE.json"
            side = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            state["sides"].append({"case": case, "mode": mode, "returncode": result.returncode,
                "status": side["status"] if side else "failed_before_pipeline_state",
                "receipt": str(receipt_path), "controller_log": str(log_path)})
            atomic_json(state_path, state)
            print(json.dumps(state["sides"][-1]), flush=True)
    state.update(status="training_sides_resolved_evaluation_and_audits_pending", elapsed_seconds=time.monotonic()-started,
                 sides_completed=sum(row["returncode"] == 0 for row in state["sides"]),
                 sides_failed=sum(row["returncode"] != 0 for row in state["sides"]))
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
