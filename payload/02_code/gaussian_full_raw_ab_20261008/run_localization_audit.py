"""Keep complete query failures while comparing cold current-input localization variants."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity
from atomic_json_windows import atomic_json, read_json


def read(path):
    return read_json(Path(path))


def semantic(value):
    if isinstance(value, dict):
        return {key: semantic(child) for key, child in value.items() if key not in ("elapsed_seconds", "image_reference")}
    if isinstance(value, list):
        return [semantic(child) for child in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", default="box2")
    parser.add_argument("--wait-for-primary", action="store_true")
    args = parser.parse_args()
    output = args.campaign / args.case / "localization_modules"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    state = {"status": "waiting_for_primary_if_requested", "rows": [], "source": data_identity(Path(__file__)),
        "primary_outputs_unchanged": True, "CPU_process_trials_are_isolated_from_primary_compute": args.wait_for_primary}
    atomic_json(output / "STATE.json", state)
    if args.wait_for_primary:
        while True:
            main_running = read(args.campaign / "CAMPAIGN_STATE.json")["status"] == "running"
            side_running = any(read(path)["status"] == "running" for path in args.campaign.glob("*/*/PIPELINE_STATE.json"))
            checkpoint_running = read(args.campaign / "box2/checkpoint_modules/STATE.json")["status"] == "running" and not (args.campaign / "box2/checkpoint_modules/RESULT.json").exists() and not (args.campaign / "box2/checkpoint_modules/FAILURE.json").exists()
            evaluation_running = read(args.campaign / "EVALUATION_CAMPAIGN_STATE.json")["status"] == "running"
            recovery_running = any(read(path)["status"] in ("waiting_for_existing_side_outcome", "running_declared_renderer_interface_recovery")
                for path in args.campaign.glob("*/*_projection_observer_*/STATE.json"))
            if not main_running and not side_running and not checkpoint_running and not evaluation_running and not recovery_running:
                break
            if time.perf_counter()-started > 21600:
                raise TimeoutError("Primary wait budget reached; existing healthy jobs are not changed")
            time.sleep(5)
    state["status"] = "running"
    atomic_json(output / "STATE.json", state)
    choices = ("eager_original", "eager_metadata", "lazy_original", "lazy_metadata", "bytes", "indexed",
               "integer", "prepared", "process", "strict", "ordered")
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for trial in range(3):
        order = choices if trial != 1 else tuple(reversed(choices))
        for mechanism in order:
            name = str(trial)+"_"+mechanism
            folder = output / name
            command = [sys.executable, CODE / "localization_variant.py", "--side", args.campaign / args.case / "optimized",
                       "--output", folder, "--mechanism", mechanism]
            if mechanism in ("integer", "prepared", "process", "strict", "ordered"):
                command = [sys.executable, CODE / "gpu_lease.py", "--receipt", output / (name+"_GPU_LEASE.json"),
                           "--label", "localization_audit_"+name, "--", *command]
            guarded = [sys.executable, CODE / "run_owned_task.py", "--receipt", output / (name+"_process.json"),
                       "--log", output / (name+".log"), "--maximum-seconds", "3300", "--maximum-rss-gib", "8", "--", *command]
            before = time.perf_counter()
            child = subprocess.run(list(map(str, guarded)), env=environment)
            row = {"trial": trial, "mechanism": mechanism, "returncode": child.returncode,
                   "whole_stage_seconds": time.perf_counter()-before, "folder": str(folder)}
            if child.returncode == 0:
                row["result"] = read(folder / "RESULT.json")
            else:
                row["failure"] = read(folder / "FAILURE.json") if (folder / "FAILURE.json").exists() else {"guard": read(output / (name+"_process.json"))}
            state["rows"].append(row)
            atomic_json(output / "STATE.json", state)
    reference = next((row for row in state["rows"] if row["mechanism"] == "lazy_metadata" and row["returncode"] == 0), None)
    if reference is None:
        raise RuntimeError("The complete current native localization reference failed")
    original = read(Path(reference["folder"]) / "localization/LOCALIZATION.json")
    original_rows = semantic(original["rows"])
    outcomes = {}
    for mechanism in choices:
        trials = [row for row in state["rows"] if row["mechanism"] == mechanism]
        exact = []
        for row in trials:
            exact.append(row["returncode"] == 0 and semantic(read(Path(row["folder"]) / "localization/LOCALIZATION.json")["rows"]) == original_rows)
        outcomes[mechanism] = {"successful_trials": sum(row["returncode"] == 0 for row in trials),
            "complete24_semantic_rows_exact_per_trial": exact,
            "median_whole_stage_seconds": statistics.median(row["whole_stage_seconds"] for row in trials),
            "all_failed_evaluation_rows_retained": all(exact)}
    state.update(status="complete_mechanism_outcomes_recorded", outcomes=outcomes,
        elapsed_seconds=time.perf_counter()-started, original_evaluation_summary=original["summary"],
        no_localization_failures_removed=True, absolute_pose_accuracy_not_established=True)
    atomic_json(output / "RESULT.json", state)
    atomic_json(output / "STATE.json", state)


if __name__ == "__main__":
    main()
