"""Prepare two independent raw-only July motion stages in a bounded CPU lane."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.campaign / "JULY_COARSE_PAIR_STATE.json"
    if state_path.exists():
        raise FileExistsError("A raw-only coarse pair needs a new controller receipt")
    cases = json.loads((args.campaign / "CASES.json").read_text())["cases"]
    raw = next(row["raw"] for row in cases if row["case"] == "c0_july")
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
        "source": data_identity(Path(__file__)), "results": [], "raw": raw,
        "two_independent_raw_solutions": True, "CPU_lane": "one coarse side at a time, BLAS and OpenMP one thread",
        "overlap_with_main_pipeline_is_not_isolated_speed_evidence": True}
    started = time.monotonic()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        for mode in ("baseline", "optimized"):
            side = args.campaign / "c0_july" / mode
            side.mkdir(parents=True, exist_ok=True)
            receipt, log = side / "coarse_process.json", side / "coarse.log"
            state["mode"] = mode
            atomic_json(state_path, state)
            command = [sys.executable, V1 / "run_bounded_task.py", "--receipt", receipt, "--log", log,
                       "--maximum-seconds", "3600", "--maximum-rss-gib", "12", "--", sys.executable,
                       Path(__file__).with_name("run_coarse_c0.py"), "--raw", raw, "--output", side / "coarse"]
            result = subprocess.run(list(map(str, command)), env=env)
            measured = json.loads(receipt.read_text())
            state["results"].append({"mode": mode, "returncode": result.returncode, "receipt": str(receipt),
                "status": measured["status"], "elapsed_seconds": measured["elapsed_seconds"]})
            atomic_json(state_path, state)
            if result.returncode:
                raise RuntimeError("The complete raw-only motion stage failed; no visual endpoint substitute is allowed")
        state.update(status="two_fresh_raw_coarse_stages_completed_geometry_unverified", elapsed_seconds=time.monotonic()-started)
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status="failed_raw_coarse_pair", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
