"""Retest complete localization compositions after recording unchanged legacy failures."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

from atomic_json_windows import atomic_json, read_json
from run_localization_audit import semantic

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity

CHOICES = ("eager_original", "eager_metadata", "lazy_original", "lazy_metadata", "bytes", "indexed",
           "integer", "prepared", "process", "strict", "ordered")


def features_exact(reference, candidate):
    left = {path.name: path for path in reference.glob("*.npz")}
    right = {path.name: path for path in candidate.glob("*.npz")}
    if set(left) != set(right):
        return False
    for name in left:
        with np.load(left[name]) as first, np.load(right[name]) as second:
            if set(first.files) != set(second.files):
                return False
            if not all(first[field].dtype == second[field].dtype and
                    np.array_equal(first[field], second[field]) for field in first.files):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    campaign.relative_to((ROOT / "04_outputs/gaussian_full_raw_ab_20261008").resolve())
    args.output.resolve().relative_to(campaign)
    legacy = read_json(campaign / "box2/localization_modules/RESULT.json")
    isolated = read_json(campaign / "box2/matching_modules_isolated/RESULT.json")
    if legacy["status"] != "complete_mechanism_outcomes_recorded" or not str(isolated["status"]).startswith("complete"):
        raise ValueError("Both earlier native audits must resolve before compatibility retesting")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
        "started_UTC": datetime.now(timezone.utc).isoformat(), "rows": [],
        "source": data_identity(Path(__file__)), "legacy_failures_preserved": True,
        "initial_selection_or_training_unchanged": True, "full_raw_denominator": 24,
        "no_threshold_relaxation": True, "matching_audit_terminal_before_launch": True}
    atomic_json(args.output / "STATE.json", state)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        for trial in range(3):
            order = CHOICES if trial != 1 else tuple(reversed(CHOICES))
            for mechanism in order:
                name = str(trial)+"_"+mechanism
                folder = args.output / name
                command = [sys.executable, CODE / "localization_variant.py", "--side", campaign / "box2/optimized",
                    "--output", folder, "--mechanism", mechanism, "--compat-adapter"]
                if mechanism in ("integer", "prepared", "process", "strict", "ordered"):
                    command = [sys.executable, CODE / "gpu_lease.py", "--receipt", args.output / (name+"_GPU_LEASE.json"),
                        "--label", "compatibility_localization_"+name, "--", *command]
                guarded = [sys.executable, CODE / "run_owned_task.py", "--receipt", args.output / (name+"_process.json"),
                    "--log", args.output / (name+".log"), "--maximum-seconds", "3300", "--maximum-rss-gib", "8", "--", *command]
                before = time.perf_counter()
                child = subprocess.run(list(map(str, guarded)), env=environment)
                row = {"trial": trial, "mechanism": mechanism, "returncode": child.returncode,
                    "whole_stage_seconds": time.perf_counter()-before, "folder": str(folder)}
                result = folder / ("RESULT.json" if child.returncode == 0 else "FAILURE.json")
                row["result" if child.returncode == 0 else "failure"] = read_json(result) if result.is_file() else {
                    "guard": read_json(args.output / (name+"_process.json"))}
                state["rows"].append(row)
                atomic_json(args.output / "STATE.json", state)
        reference = next(row for row in state["rows"] if row["mechanism"] == "lazy_metadata" and row["returncode"] == 0)
        reference_path = Path(reference["folder"]) / "localization"
        original = read_json(reference_path / "LOCALIZATION.json")
        original_rows = semantic(original["rows"])
        outcomes = {}
        for mechanism in CHOICES:
            trials = [row for row in state["rows"] if row["mechanism"] == mechanism]
            exact, feature_values = [], []
            for row in trials:
                success = row["returncode"] == 0
                candidate = Path(row["folder"]) / "localization"
                exact.append(success and semantic(read_json(candidate / "LOCALIZATION.json")["rows"]) == original_rows)
                feature_values.append(success and features_exact(reference_path / "feature_cache", candidate / "feature_cache"))
            outcomes[mechanism] = {"successful_trials": sum(row["returncode"] == 0 for row in trials),
                "complete24_semantic_rows_exact_per_trial": exact, "complete_query_feature_arrays_exact_per_trial": feature_values,
                "median_whole_stage_seconds": statistics.median(row["whole_stage_seconds"] for row in trials),
                "all_failed_evaluation_rows_retained": all(exact)}
        state.update(status="complete_compatibility_outcomes_recorded", outcomes=outcomes,
            elapsed_seconds=time.perf_counter()-started, original_evaluation_summary=original["summary"],
            no_localization_failures_removed=True, absolute_pose_accuracy_not_established=True,
            all_mechanisms_complete_exact=all(item["successful_trials"] == 3 and
                all(item["complete24_semantic_rows_exact_per_trial"]) and
                all(item["complete_query_feature_arrays_exact_per_trial"]) for item in outcomes.values()))
        atomic_json(args.output / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_compatibility_audit", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output / "STATE.json", state)
        raise
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
