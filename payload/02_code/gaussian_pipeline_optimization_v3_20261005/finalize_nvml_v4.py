"""Accept one optional monitor component after real exits and complete known costs."""

import json
import msvcrt
import os
from pathlib import Path
import statistics
import sys
import time

import psutil


STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm_v4_001"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm001"
SHARED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    target = OWN / "ACCEPTANCE_V4_001.json"
    if target.exists():
        raise FileExistsError("The V4 acceptance must not replace an earlier receipt")
    review = read(OWN / "r0/RESULT.json")
    old = read(OLD / "c1/RESULT.json")
    checks = {"all23_separate_root_review_checks_passed": len(review["checks"]) == 23 and all(review["checks"].values()),
              "original_three_pairs_reused_not_fake_new_fresh_runs": review["formal_three_pairs_reused"] and not review["fresh_three_performance_pairs_this_V4"],
              "old_strict_failure_and_actual_f0_cleanup_failure_preserved": read(OLD / "NEGATIVE_CLOSURE_001.json")["new_optimization_count"] == 0
              and read(OWN / "f0/node/NODE.json")["status"] == "failed"}
    process_records = {(row["pid"], row["creation_time"]): row for row in review["owned_Windows_process_records"]}
    process_records[(review["review_pid"], review["review_creation_time"])] = {
        "pid": review["review_pid"], "creation_time": review["review_creation_time"]}
    for name in ("GENERIC_PROCESS_001.json", "GENERIC_PROCESS_002.json"):
        for row in read(OLD / name)["owned_process_identities"]:
            process_records[(row["pid"], row["creation_time"])] = row
    for label in ("g0", "f0", "f1", "c0", "r0"):
        lease, process = read(OWN / (label + "_lease.json")), read(OWN / (label + "_process.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != process["returncode"]:
            raise AssertionError("A complete actual V4 child or lease remains open")
        for role in ("guard", "child"):
            row = {"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]}
            process_records[(row["pid"], row["creation_time"])] = row
        for row in process["owned_process_identities"]:
            process_records[(row["pid"], row["creation_time"])] = row
    for row in process_records.values():
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                raise AssertionError("A saved matching owned Windows instance remains live")
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_old_new_and_review_actual_Windows_instances_exited"] = True
    checks["all10_actual_Linux_instances_checked_exited"] = len(review["actual_Linux_post_exit"]) == 10 and all(not row["alive"] for row in review["actual_Linux_post_exit"])
    for row in review["sources_and_inputs_checked"]:
        if data_identity(row["path"]) != row:
            raise AssertionError("A reviewed source or protected original input changed before acceptance")
    checks["all37_reviewed_sources_and_inputs_unchanged_now"] = len(review["sources_and_inputs_checked"]) == 37
    if read(SHARED / "LOCK_OWNER.json")["status"] != "released":
        raise AssertionError("The measurement owner did not release its OS lease")
    with (SHARED / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_free_and_all5_V4_leases_released"] = True
    fee = review["fee_before_this_review"] + read(OWN / "r0_lease.json")["child_elapsed_seconds"]
    publication_seconds = time.perf_counter() - STARTED
    total_fee = fee + publication_seconds
    phases = {mode: statistics.median(row["raw_CPU_phase_seconds"][mode] + total_fee / 6 for row in old["pairs"]) for mode in ("cli", "nvml")}
    wholes = {mode: statistics.median(row["raw_whole_seconds"][mode] + total_fee / 6 for row in old["pairs"]) for mode in ("cli", "nvml")}
    reduction = 1 - phases["nvml"] / phases["cli"]
    checks["complete_known_cost_phase_reduction_at_least5percent"] = reduction >= .05
    checks["V4_whole_median_at_most1percent_more_reported_as_tradeoff"] = wholes["nvml"] <= 1.01 * wholes["cli"]
    checks["each_whole_original1_1_guard_preserved"] = all(row["raw_whole_seconds"]["nvml"] + total_fee / 6 <= 1.1 * (row["raw_whole_seconds"]["cli"] + total_fee / 6) for row in old["pairs"])
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["unique_new_mechanism_parent23_before_append"] = state["additional_accepted_count"] == 23 and len(state["accepted_mechanisms"]) == 23 and not any(name.startswith("R122") for name in state["accepted_mechanisms"])
    if not all(checks.values()):
        raise AssertionError("The complete V4 acceptance did not pass its final gates")
    result = {"status": "accepted_optional_scoped_stage_optimization_whole_tradeoff",
              "candidate": "R122_V4", "new_optimization_count": 1,
              "acceptance_protocol": "ENGINEERING_ACCEPTANCE_V4_speed5_whole_median1percent_original_quality_and_each1_1",
              "scope": review["scope"], "checks": checks,
              "complete_phase_medians_seconds": phases, "complete_phase_reduction": reduction,
              "complete_whole_medians_seconds": wholes, "complete_whole_ratio": wholes["nvml"] / wholes["cli"],
              "whole_increase_seconds": wholes["nvml"] - wholes["cli"],
              "whole_pipeline_faster_claimed": False, "known_fee_before_publication": fee,
              "publication_internal_seconds": publication_seconds,
              "all_known_fee_before_publisher_exit_seconds": total_fee,
              "owned_Windows_instances_exited": list(process_records.values()),
              "actual_Linux_checked_exited": review["actual_Linux_post_exit"],
              "source_input_records": 37, "review": str(OWN / "r0/RESULT.json"),
              "review_kind": "separate_root_self_review_no_subagent", "fresh_GPU_generic": str(OWN / "g0/RESULT.json"),
              "actual_fault_and_recovery": [str(OWN / "f1/RESULT.json"), str(OWN / "c0/RESULT.json")],
              "formal_three_pairs_reused": True, "new_fresh_three_performance_pairs_claimed": False,
              "old_strict_failure_overwritten": False, "production_deployed": False,
              "new_training_GT_unknown_objects_latest_R120_R124_combination": "not_verified_this_scope",
              "publisher_pid": os.getpid(), "publisher_creation_time": psutil.Process().create_time(),
              "publisher_source": data_identity(Path(__file__))}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "phase_reduction": reduction,
                      "whole_ratio": result["complete_whole_ratio"], "new_optimization_count": 1}), flush=True)


if __name__ == "__main__":
    main()
