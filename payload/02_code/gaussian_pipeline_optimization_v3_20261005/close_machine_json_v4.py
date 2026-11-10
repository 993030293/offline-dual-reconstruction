"""Publish the actual-cost lower-bound rejection and resource closure."""

import time

STARTED = time.perf_counter()

import json
import msvcrt
import os
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


PARENT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OLD = PARENT / "machine_json_001"
OWN = PARENT / "mj_v4_001"


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
        raise FileExistsError("The negative JSON closure already exists")
    audit = read(OWN / "a1/RESULT.json")
    if not all(audit["checks"].values()):
        raise AssertionError("The saved full-source/output and cost audit did not pass")
    original = read(OLD / "campaign_001/RESULT.json")
    fee = audit["original_additional_fee_lower_bound_seconds"]
    processes, new_leases = [], []
    for name in ("machine_json_001_oracle_lease.json", "machine_json_001_profile_lease.json"):
        row = read(PARENT / name)
        for role in ("guard", "child"):
            processes.append({"pid": row[role + "_pid"], "creation_time": row[role + "_creation_time"]})
    for label, expected in (("a0", 1), ("a1", 0)):
        lease = read(OWN / (label + "_lease.json"))
        guard = read(OWN / (label + "_guard.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != expected:
            raise AssertionError("An actual JSON audit lease did not close as recorded")
        fee += lease["child_elapsed_seconds"]
        for role in ("guard", "child"):
            processes.append({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]})
        processes.extend(guard.get("owned_process_identities", []))
        new_leases.append({"label": label, "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
        if guard["peak_tree_RSS_bytes"] > guard["maximum_RSS_bytes"] or guard["elapsed_seconds"] > guard["maximum_seconds"]:
            raise AssertionError("An actual bounded audit exceeded its time or memory limit")
    for name in ("machine_json_001_oracle_process.json", "machine_json_001_profile_process.json"):
        processes.extend(read(PARENT / name).get("owned_process_identities", []))
    unique = {(row["pid"], row["creation_time"]): row for row in processes}
    if not all(closed(row) for row in unique.values()):
        raise AssertionError("An actual old or new owned process instance still runs")
    owner = read(PARENT / "coordination_001/LOCK_OWNER.json")
    if owner["status"] != "released" or not owner.get("lease_released_utc"):
        raise AssertionError("The last shared resource owner did not release")
    with (PARENT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    elapsed = time.perf_counter() - STARTED
    fee += elapsed
    phases = {label: original["medians"][label]["phase_seconds"] + fee / 6 for label in ("native", "compact")}
    wholes = {label: original["medians"][label]["whole_seconds"] + fee / 6 for label in ("native", "compact")}
    reduction = 1 - phases["compact"] / phases["native"]
    if reduction >= .05:
        raise AssertionError("A lower-bound failure is not established; acceptance is prohibited here")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if state["additional_accepted_count"] != 24 or any(name.startswith("R110") for name in state["accepted_mechanisms"]):
        raise AssertionError("The parent frontier unexpectedly changed")
    failure = read(OWN / "a0/FAILURE.json")
    result = {"status": "closed_negative_V4_known_fee_lower_bound_stage_upper_below5percent", "candidate": "R110_V4",
              "new_optimization_count": 0, "parent_count": 24, "parent_goal_remains_active": True,
              "complete_stage_reduction_upper_bound": reduction, "complete_phase_medians_lower_bound_seconds": phases,
              "complete_whole_medians_lower_bound_seconds": wholes, "whole_ratio_at_known_lower_bound": wholes["compact"] / wholes["native"],
              "additional_known_fee_lower_bound_before_publisher_exit_seconds": fee,
              "actual_formal_unallocated_seconds": audit["actual_formal_unallocated_seconds"],
              "old_generic_lease_seconds": audit["old_oracle_seconds"], "new_audit_leases": new_leases,
              "publication_internal_seconds": elapsed, "six_source_output_arithmetic_checks_passed": True,
              "old_first_and_comparison_fees_not_double_charged": True, "old_formal_window_reused_not_double_charged": True,
              "actual_new_list_schema_reader_setup_failure_preserved": failure,
              "new_schema_fixed_and_all_three_complete1016_type_order_floatbit_paths_rechecked": True,
              "old_strict_cost_failure_preserved": True,
              "browser_JSON_parse24_drone_path_free_flight_and_recovery": "not_run_after_complete_cost_lower_bound_failure",
              "unmetered_old_reviews_text_research_and_later_publication": "not_quantified_only_reduce_the_upper_bound",
              "no_new_train_GT_unseen_object_or_production_default": True,
              "all_saved_actual_Windows_instances_closed": list(unique.values()),
              "four_old_and_new_leases_released_and_OS_lock_observed_free": True,
              "source_identity": data_identity(Path(__file__)), "publisher_pid": os.getpid(),
              "publisher_creation_time": psutil.Process().create_time(), "publisher_actual_exit_pending": True}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "known_fee_lower_bound": fee,
                      "stage_reduction_upper_bound": reduction, "whole_ratio": result["whole_ratio_at_known_lower_bound"], "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
