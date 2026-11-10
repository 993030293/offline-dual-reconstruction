"""Close the cost-negative V4 filter audit without promoting the mechanism."""

import time

STARTED = time.perf_counter()

import json
import msvcrt
import os
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/point_filter_predicate_001"
SHARED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    target = OWN / "V4_NEGATIVE_CLOSURE_001.json"
    if target.exists():
        raise FileExistsError("The final negative closure already exists")
    review = read(OWN / "r0/RESULT.json")
    if not all(review["checks"].values()):
        raise AssertionError("A semantic, source, complete-output or resource review failed")
    processes = {(row["pid"], row["creation_time"]): row for row in review["closed_process_records"]}
    leases = []
    for label in ("g0", "e0", "e1", "r0"):
        lease = read(OWN / (label + "_lease.json"))
        guard = read(OWN / (label + "_guard.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("A V4 component lease did not close")
        for role in ("guard", "child"):
            row = {"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]}
            processes[(row["pid"], row["creation_time"])] = row
        for row in guard.get("owned_process_identities", []):
            processes[(row["pid"], row["creation_time"])] = row
        leases.append({"label": label, "child_seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
    if not all(closed(row) for row in processes.values()):
        raise AssertionError("An actual owned process instance is still running")
    owner = read(SHARED / "LOCK_OWNER.json")
    if owner["status"] != "released" or not owner.get("lease_released_utc"):
        raise AssertionError("The shared benchmark owner did not record release")
    with (SHARED / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    old = read(OLD / "campaign_001/RESULT.json")
    extra_fee = sum(read(OLD / name)["child_elapsed_seconds"] for name in ("generic_001_LEASE.json", "generic_002_LEASE.json")) + sum(row["child_seconds"] for row in leases)
    elapsed = time.perf_counter() - STARTED
    known_fee = extra_fee + elapsed
    phase = {label: old["medians"][label]["phase_seconds"] + known_fee / 6 for label in ("native", "columns")}
    whole = {label: old["medians"][label]["whole_seconds"] + known_fee / 6 for label in ("native", "columns")}
    reduction = 1 - phase["columns"] / phase["native"]
    if reduction >= .05:
        raise AssertionError("This negative-only closer cannot publish an acceptance")
    output_bytes = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file())
    if output_bytes > 30 * 1024**3 or shutil.disk_usage(OWN).free < 20 * 1024**3:
        raise AssertionError("The final independent output or disk budget failed")
    result = {"status": "closed_negative_V4_complete_known_cost_below5percent", "candidate": "R113_V4",
              "new_optimization_count": 0, "parent_additional_accepted_count": 24, "parent_goal_remains_active": True,
              "semantic_source_complete_output_and_resource_checks": "25_separate_root_self_review_checks_passed",
              "complete_phase_medians_seconds": phase, "complete_phase_reduction_upper_bound": reduction,
              "complete_whole_medians_seconds": whole, "complete_whole_ratio": whole["columns"] / whole["native"],
              "all_metered_fee_before_publisher_exit_seconds": known_fee,
              "old_and_new_leases": leases, "publication_internal_seconds": elapsed,
              "publisher_exit_and_later_fees_pending_not_needed_to_prove_negative_cost": True,
              "additional_fees_only_lower_phase_reduction_no_acceptance_possible": True,
              "old_formal_first_compare_and_close_already_in_side_costs_not_double_charged": True,
              "unmetered_text_research_review_overhead": "unquantified_not_assumed_zero_no_total_wall_claim",
              "forced_delete_three_actual_native_cases": str(OLD / "v4_delete_001/RESULT.json"),
              "complete_public360_component_and7plus1_real_frame_deregister": str(OWN / "e1/RESULT.json"),
              "new_external_paired_cost_not_three_pair_performance_claim": True,
              "e0_actual_property_method_validator_failure_and_fee_preserved": True,
              "all_saved_owned_process_instances_exited": list(processes.values()),
              "all_seven_old_and_new_leases_released_and_OS_lock_observed_free": True,
              "current_source_identity": data_identity(Path(__file__)), "output_bytes_before_publication": output_bytes,
              "publisher_pid": os.getpid(), "publisher_creation_time": psutil.Process().create_time(),
              "production_deployed": False, "old_strict_failure_overwritten": False,
              "no_Gaussian_training_latest_combination_GT_or_unseen_object_full_chain": True,
              "original_full_tree_peak_RSS_history_missing_not_filled_by_new_components": True}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "known_fee": known_fee, "phase_upper_bound": reduction,
                      "whole_ratio": result["complete_whole_ratio"], "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
