"""Serialize bounded CPU diagnostics after the owned GPU timing campaign exits."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
OUT1 = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005"
OUT2 = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "waiting_for_owned_training", "controller_pid": os.getpid(),
        "maximum_wait_seconds": 1200, "training_state": str(OUT2/"q31_training_001/STATE.json"),
        "source": data_identity(Path(__file__)), "runs": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    def run(command, name, seconds, memory):
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(name+"_process.json")),
            "--log", str(args.output/(name+".log")), "--maximum-seconds", str(seconds),
            "--maximum-rss-gib", str(memory), "--", *command]
        row = {"name": name, "command": bounded, "status": "running"}
        state["runs"].append(row)
        state["status"] = name
        with (args.output/(name+"_controller.log")).open("x") as log:
            process = subprocess.Popen(bounded, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            code = process.wait()
        row.update(status="completed" if code == 0 else "failed", returncode=code)
        if code:
            raise RuntimeError("A sequential CPU diagnostic failed: "+name)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"name": name, "status": row["status"]}), flush=True)
    try:
        while True:
            training = read(Path(state["training_state"]))
            if training["status"] != "running":
                break
            if time.monotonic()-started > 1200:
                raise TimeoutError("Owned training is still active; no CPU diagnostics were launched")
            time.sleep(5)
        if any(row["status"] == "running" for row in training["runs"]):
            raise ValueError("Training controller ended with an active child; verify ownership before proceeding")
        # Poll the original controller identity without stopping any unrelated process.
        pid = str(int(training["controller_pid"]))
        for _ in range(5):
            check = subprocess.run(["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--", "ps", "-p", pid, "-o", "args="],
                capture_output=True, text=True, check=False)
            if "run_prefetch_training.py" not in check.stdout:
                break
            time.sleep(2)
        else:
            raise RuntimeError("The owned timing controller has not exited")
        state["training_terminal_status"] = training["status"]
        state["training_success_claim"] = False
        run([sys.executable, str(Path(__file__).with_name("audit_target_observations.py")),
            "--dataset", str(OUT1/"new_map_export_003/dataset"),
            "--roi", str(OUT1/"new_map_display_roi_001/vehicle_roi_receipt.json"),
            "--exclusions", str(OUT1/"p18_raw_batched_pose_001/EXCLUDED_EVALUATION_IDS.json"),
            "--output", str(OUT2/"q32_observation_audit_001")], "target_observation_audit", 315, 4)
        run([sys.executable, str(Path(__file__).with_name("run_jpeg_cache_campaign.py")),
            "--quality-cache", str(OUT1/"p1_shared_quality_cache"),
            "--validation", str(OUT2/"q33_lifecycle_001/RESULT.json"),
            "--output", str(OUT2/"q33_campaign_001")], "jpeg_cache_campaign", 1500, 12)
        state.update(status="diagnostics_completed_assessment_pending", elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
