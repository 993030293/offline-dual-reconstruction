"""Prepare remaining raw initial stages sequentially while geometry is running."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.campaign / "REMAINING_INITIAL_PREPARATION.json"
    if state_path.exists():
        raise ValueError("A raw preparation controller already has an owned receipt")
    state = {"status": "running", "source": data_identity(Path(__file__)), "rows": [],
        "one_initial_stage_at_a_time": True, "maximum_native_quality_workers": 4,
        "new_raw_inputs_only": True, "source_primary_pipeline_unchanged": True}
    atomic_json(state_path, state)
    entries = read(args.campaign / "CASES.json")
    cases = entries["cases"] if isinstance(entries, dict) else entries
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    for name in ("c1", "c0_july"):
        raw = next(row["raw"] for row in cases if row["case"] == name)
        for mode in ("baseline", "optimized"):
            side = args.campaign / name / mode
            side.mkdir(parents=True, exist_ok=True)
            receipt = side / "initial_process.json"
            if receipt.exists():
                state["rows"].append({"case": name, "mode": mode, "status": "existing_owned_stage_not_duplicated",
                    "stage_status": read(receipt)["status"]})
                atomic_json(state_path, state)
                continue
            command = [sys.executable, CODE / "run_initial.py", "--raw", raw, "--case", name,
                "--mode", mode, "--output", side / "initial"]
            if name == "c0_july":
                coarse = side / "coarse/coarse_body_odometry.npz"
                if not coarse.exists() or not read(side / "coarse_process.json")["status"] == "completed":
                    raise ValueError("Current complete raw July motion must precede raw image selection")
                command.extend(["--coarse-trajectory", coarse])
            guarded = [sys.executable, V1 / "run_bounded_task.py", "--receipt", receipt,
                "--log", side / "initial.log", "--maximum-seconds", "1200", "--maximum-rss-gib", "32", "--", *command]
            before = time.perf_counter()
            completed = subprocess.run(list(map(str, guarded)), env=environment)
            state["rows"].append({"case": name, "mode": mode, "returncode": completed.returncode,
                "wall_seconds": time.perf_counter()-before, "receipt": str(receipt), "status": read(receipt)["status"]})
            atomic_json(state_path, state)
    state["status"] = "all_four_remaining_initial_stage_outcomes_recorded"
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
