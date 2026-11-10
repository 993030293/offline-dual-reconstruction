"""Close an exited fusion review and apply the registered full-node speed gate."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    if (OUT / "COMBINATION_ACCEPTANCE.json").exists():
        raise FileExistsError("Never overwrite a combination decision")
    own = data_identity(Path(__file__))
    shutil.copyfile(Path(__file__), OUT / "CLOSING_SOURCE.py")
    review = read(OUT / "a0/REVIEW.json")
    lease = read(OUT / "REVIEW_LEASE_001.json")
    guard = read(OUT / "REVIEW_GUARD_001.json")
    goal = read(OUT / "NATIVE_GOAL_SOURCE_GET.json")["goal"]
    campaign = read(OUT / "c1/RESULT.json")
    rows = review["Windows"] + guard["owned_process_identities"]
    rows += [{"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]}
             for k in ["guard", "child"]]
    unique = {(r["pid"], r["creation_time"]): dict(pid=r["pid"], creation_time=r["creation_time"]) for r in rows}
    windows = [dict(r, gone=gone(r)) for r in unique.values()]
    checks = {
        "all_thirteen_original_self_review_gates_passed": len(review["checks"]) == 13 and all(review["checks"].values()),
        "exited_successful_review_guard_and_lease_closed": guard["status"] == "completed"
            and lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
        "all_prior_and_review_Windows_instances_gone": all(r["gone"] for r in windows),
        "recorded_actual_prior_Linux_closure_preserved": all(not r["alive"] for r in review["Linux"]),
        "all_group_leases_released": all(bool(read(p).get("lease_released_utc")) for p in OUT.glob("*_LEASE_*.json")),
        "native_same_parent_goal_active": goal["status"] == "active"
            and goal["threadId"] == "01a079ba-7973-7780-bf60-a034f8b50aa3",
        "combination_new_independent_mechanism_count_zero": review["new_optimization_count"] == 0
            and campaign["new_optimization_count"] == 0,
        "closing_source_unchanged": own == data_identity(Path(__file__)),
    }
    close_seconds = time.perf_counter() - STARTED
    fees = review["all_prior_nonoverlapping_shared_fees_seconds"] + lease["child_elapsed_seconds"] + close_seconds
    raw = {mode: statistics.median(p["raw_complete_node_and_whole_seconds"][mode] for p in campaign["pairs"])
           for mode in ["incumbent", "process"]}
    medians = {mode: value + fees / 6 for mode, value in raw.items()}
    ratios = [(p["raw_complete_node_and_whole_seconds"]["process"] + fees / 6) /
              (p["raw_complete_node_and_whole_seconds"]["incumbent"] + fees / 6) for p in campaign["pairs"]]
    reduction = 1 - medians["process"] / medians["incumbent"]
    checks["registered_complete_node5percent_and_whole_gates_after_all_measured_fees"] = (
        reduction >= .05 and medians["process"] <= medians["incumbent"] and all(r <= 1.1 for r in ratios))
    result = {
        "status": "accepted_scoped_engineering_combination" if all(checks.values()) else "failed_close",
        "combination": "F001_R120_R124_GEOMETRY_SIFT_COMBINATION",
        "new_optimization_count": 0, "checks": checks,
        "raw_complete_localization_node_medians_seconds": raw,
        "complete_node_and_whole_medians_with_all_measured_fees_seconds": medians,
        "complete_node_and_whole_reduction_with_all_measured_fees": reduction,
        "whole_pair_ratios_with_all_measured_fees": ratios,
        "all_nonoverlapping_measured_shared_fees_seconds": fees,
        "per_side_surcharge_seconds": fees / 6,
        "old_two_mechanisms_first_fees_seconds": review["original_two_mechanism_first_fees_seconds"],
        "exited_self_review_seconds": lease["child_elapsed_seconds"],
        "closure_before_publication_seconds": close_seconds,
        "current_source_and_input_identities": len(review["current_source_input_identities"]),
        "saved_source_copies_exact_recorded_revisions": len(review["saved_source_copies"]),
        "prior_and_review_Windows_instances_closed": len(windows),
        "prior_Linux_instances_closed": len(review["Linux"]),
        "group_leases_released": len(list(OUT.glob("*_LEASE_*.json"))),
        "root_single_chat_self_review": True, "no_subagent": True,
        "scope": "current_immutable1016_map_cold24_queries48_matches_R120_control_add_original_R124_two_CPU_processes_capacity2",
        "stage_and_whole_same_actual_complete_localization_node_window": True,
        "external_other_objects_full_pipeline_independent_GT_new_training_preview_production": "not_run",
        "cancellation_evidence": "actual_KeyboardInterrupt_exception_injection_after_two_queries_not_OS_SIGINT",
        "closing_source": own,
        "small_final_metadata_and_text_publication_after_close_timer_not_claimed_included": True,
        "own_closing_script_exit_requires_outer_caller_check": True,
    }
    atomic_json(OUT / "FINAL_COST_001.json", result)
    atomic_json(OUT / "FINAL_CLOSURE_001.json", {
        "status": "closed_scoped_combination_ready" if all(checks.values()) else "failed_close",
        "checks": checks, "Windows": windows, "new_optimization_count": 0,
        "closing_script_own_pid": psutil.Process().pid,
        "closing_script_own_creation_time": psutil.Process().create_time(),
    })
    if not all(checks.values()):
        raise AssertionError("A final original source, process, cost or goal gate failed")
    atomic_json(OUT / "COMBINATION_ACCEPTANCE.json", result)
    print(json.dumps({k: result[k] for k in ["status", "checks", "all_nonoverlapping_measured_shared_fees_seconds",
        "complete_node_and_whole_medians_with_all_measured_fees_seconds", "complete_node_and_whole_reduction_with_all_measured_fees"]}), flush=True)


if __name__ == "__main__":
    main()
