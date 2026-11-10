"""Run the legacy/full-intrinsics pair sequentially with persistent child logs."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    contract = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed,
                "iterations": 10000, "input_dataset": str(args.dataset), "runs": [],
                "primary_evaluation": "frozen_C212_H_cameras_full_intrinsics_all_24_development_frames",
                "acceptance": "projection contract and finite training; image metrics reported before promotion",
                "scope": "P0_pair_not_15_optimizations", "source": data_identity(__file__)}
    atomic_json(args.output/"PAIR_STATE.json", contract)
    for projection in ["legacy", "full"]:
        output = args.output / projection
        command = [sys.executable, str(Path(__file__).with_name("train_case.py")),
                   "--runtime", str(args.runtime), "--dataset", str(args.dataset),
                   "--output", str(output), "--projection", projection, "--seed", str(args.seed)]
        record = {"projection": projection, "command": command, "output": str(output), "status": "starting"}
        contract["runs"].append(record)
        with (args.output/f"{projection}.log").open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            record.update(pid=child.pid, status="running")
            atomic_json(args.output/"PAIR_STATE.json", contract)
            print(json.dumps(record), flush=True)
            try:
                code = child.wait(timeout=1900)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                record.update(status="timed_out", elapsed_seconds=time.monotonic()-started)
                contract["status"] = "failed"
                atomic_json(args.output/"PAIR_STATE.json", contract)
                raise
        state = json.loads((output/"STATE.json").read_text()) if (output/"STATE.json").exists() else {}
        record.update(returncode=code, status=state.get("status", "missing_state"))
        if code or record["status"] != "completed":
            contract["status"] = "failed"
            atomic_json(args.output/"PAIR_STATE.json", contract)
            raise RuntimeError(f"Training {projection} failed; preserve the output and diagnose")
        atomic_json(args.output/"PAIR_STATE.json", contract)
    contract.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output/"PAIR_STATE.json", contract)
    print(json.dumps({"status": contract["status"], "elapsed_seconds": contract["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
