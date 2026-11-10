"""Run one seed of full-intrinsics reference/fused training in a single GPU queue."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--numeric-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--training-script", type=Path, default=Path(__file__).with_name("train_case.py"))
    args = parser.parse_args()
    validated = json.loads(args.numeric_receipt.read_text())
    if validated["status"] != "numeric_and_kernel_timing_completed_training_pending":
        raise ValueError("The isolated extension has not passed its registered numerical preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "same_full_intrinsics": True, "iterations_per_run": 10000, "input": str(args.dataset),
        "source": data_identity(Path(__file__)), "extension_numeric_receipt": str(args.numeric_receipt),
        "training_script": data_identity(args.training_script),
        "all_3_paired_seeds_required_for_acceleration_acceptance": True,
        "acceptance": "median training wall reduction >=10 percent; each seed full/target PSNR delta >=-0.1 dB and SSIM delta >=-0.002; protected views checked"}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    for label in (["reference", "fused"] if args.seed != 1 else ["fused", "reference"]):
        command = [sys.executable, str(args.training_script), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full", "--seed", str(args.seed),
            "--ssim-backend", label]
        if label == "fused":
            command += ["--ssim-source", str(args.source), "--ssim-library", str(args.library)]
        row = {"variant": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        with (args.output/(label+".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(status="running", pid=process.pid)
            atomic_json(args.output/"PAIR_STATE.json", state)
            try:
                code = process.wait(timeout=1900)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                row.update(status="watchdog_timeout")
                state["status"] = "failed"
                atomic_json(args.output/"PAIR_STATE.json", state)
                raise
        row.update(status="completed" if code == 0 else "failed", returncode=code)
        atomic_json(args.output/"PAIR_STATE.json", state)
        if code:
            state["status"] = "failed"
            atomic_json(args.output/"PAIR_STATE.json", state)
            raise RuntimeError("Training case failed; paired comparison is incomplete")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({"status": state["status"], "seed": args.seed, "elapsed_seconds": state["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
