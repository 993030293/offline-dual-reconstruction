"""Run explicit renderer-interface recovery sequentially after preserved failures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    output = args.campaign / args.case / "renderer_interface_repair_pair"
    output.mkdir(exist_ok=False)
    state = {"status": "running", "rows": [], "pid": os.getpid(), "birth": psutil.Process().create_time(),
        "source": data_identity(Path(__file__)), "original_failure_acceptance_unchanged": True,
        "new_renderer_interface_policy_not_original75_pass": True}
    atomic_json(output / "STATE.json", state)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    for mode in ("baseline", "optimized"):
        side = args.campaign / args.case / mode
        previous = read(side / "PIPELINE_STATE.json")
        if previous["status"] != "failed_pipeline_stage" or previous["current_stage"] != "metric_sim3_track_recovery":
            raise ValueError("Only the retained original metric-track failure enters this explicit pair")
        try:
            if psutil.Process(previous["pid"]).create_time() == previous["process_birth"]:
                raise RuntimeError("The original side controller still owns its pipeline")
        except psutil.NoSuchProcess:
            pass
        atomic_json(output / (mode+"_ORIGINAL_PIPELINE_FAILURE.json"), previous)
        before = time.perf_counter()
        with (output / (mode+".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(CODE / "run_side.py"), "--case", args.case,
                "--mode", mode, "--campaign", str(args.campaign), "--render-near-recovery"],
                env=environment, stdout=log, stderr=subprocess.STDOUT)
        current = read(side / "PIPELINE_STATE.json")
        state["rows"].append({"mode": mode, "returncode": result.returncode,
            "status": current["status"], "elapsed_seconds": time.perf_counter()-before,
            "receipt": str(side / "PIPELINE_STATE.json")})
        atomic_json(output / "STATE.json", state)
    state["status"] = "both_explicit_renderer_repair_outcomes_recorded"
    atomic_json(output / "RESULT.json", state)
    atomic_json(output / "STATE.json", state)


if __name__ == "__main__":
    main()
