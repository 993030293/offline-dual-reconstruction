"""Retain failed option receipts and repeat only the complete process-localization trials."""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

from atomic_json_windows import atomic_json, read_json
from run_localization_audit import semantic
from run_localization_compatibility_audit import features_exact
from runtime import data_identity

CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    args.output.resolve().relative_to(campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__), CODE / "localization_process_receipt_retry.py",
        CODE / "localization_variant.py", CODE / "localization_contract_adapter.py"]
    identities = [data_identity(path) for path in sources]
    state = {"status": "waiting_for_current_compatibility_outcomes", "pid": os.getpid(),
        "process_birth": psutil.Process().create_time(), "sources": identities,
        "rows": [], "all_original_failures_preserved": True, "full_query_denominator": 24}
    atomic_json(args.output / "STATE.json", state)
    try:
        parent_path = campaign / "box2/localization_modules_compatibility_001/RESULT.json"
        while not parent_path.is_file():
            if read_json(parent_path.with_name("STATE.json"))["status"].startswith("failed"):
                raise RuntimeError("The preceding complete audit failed before source outcomes could be compared")
            if time.perf_counter()-started > 1200:
                raise TimeoutError("The current audit did not resolve; healthy processes are unchanged")
            time.sleep(5)
        parent = read_json(parent_path)
        if parent["status"] != "complete_compatibility_outcomes_recorded":
            raise ValueError("The preceding audit has no complete mechanism outcomes")
        if identities != [data_identity(path) for path in sources]:
            raise ValueError("The queued retry source changed before execution")
        reference = next(row for row in parent["rows"] if row["trial"] == 0 and
            row["mechanism"] == "lazy_metadata" and row["returncode"] == 0)
        reference_path = Path(reference["folder"]) / "localization"
        reference_rows = semantic(read_json(reference_path / "LOCALIZATION.json")["rows"])
        state.update(status="running_complete_process_receipt_retries", previous_full_audit=str(parent_path))
        atomic_json(args.output / "STATE.json", state)
        for trial in range(3):
            folder = args.output / (str(trial)+"_process")
            command = [sys.executable, CODE / "gpu_lease.py", "--receipt", args.output / (str(trial)+"_GPU_LEASE.json"),
                "--label", "process_receipt_retry_"+str(trial), "--", sys.executable,
                CODE / "localization_process_receipt_retry.py", "--side", campaign / "box2/optimized", "--output", folder]
            guarded = [sys.executable, CODE / "run_owned_task.py", "--receipt", args.output / (str(trial)+"_process.json"),
                "--log", args.output / (str(trial)+".log"), "--maximum-seconds", "3300", "--maximum-rss-gib", "8", "--", *command]
            before = time.perf_counter()
            child = subprocess.run(list(map(str, guarded)), env={**os.environ, "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
            row = {"trial": trial, "returncode": child.returncode,
                "whole_stage_seconds": time.perf_counter()-before, "folder": str(folder)}
            success = child.returncode == 0
            row["complete24_semantic_rows_exact"] = success and semantic(read_json(folder / "localization/LOCALIZATION.json")["rows"]) == reference_rows
            row["complete_query_feature_arrays_exact"] = success and features_exact(reference_path / "feature_cache", folder / "localization/feature_cache")
            row["actual_process_result"] = read_json(folder / "RESULT.json") if success else {"guard": read_json(args.output / (str(trial)+"_process.json"))}
            state["rows"].append(row)
            atomic_json(args.output / "STATE.json", state)
        complete = all(row["returncode"] == 0 and row["complete24_semantic_rows_exact"] and
            row["complete_query_feature_arrays_exact"] for row in state["rows"])
        candidate = {"successful_trials": sum(row["returncode"] == 0 for row in state["rows"]),
            "complete24_semantic_rows_exact_per_trial": [row["complete24_semantic_rows_exact"] for row in state["rows"]],
            "complete_query_feature_arrays_exact_per_trial": [row["complete_query_feature_arrays_exact"] for row in state["rows"]],
            "median_whole_stage_seconds": statistics.median(row["whole_stage_seconds"] for row in state["rows"]),
            "all_failed_evaluation_rows_retained": complete}
        state.update(status="complete_process_receipt_retry_outcomes_recorded", complete_all_oracles=complete,
            outcomes={"prepared": parent["outcomes"]["prepared"], "process": candidate},
            no_localization_failures_removed=True, elapsed_seconds=time.perf_counter()-started,
            source_unchanged=identities == [data_identity(path) for path in sources])
        if not state["source_unchanged"]:
            raise ValueError("The complete retry source changed during execution")
        atomic_json(args.output / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_process_receipt_retry_audit", error=repr(error))
        atomic_json(args.output / "STATE.json", state)
        raise
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
