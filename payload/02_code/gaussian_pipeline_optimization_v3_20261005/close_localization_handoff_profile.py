"""Close the full-entry diagnostic and record the bounded next ranking action."""

import json
import msvcrt
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/olp001"
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from sift_process_node_003 import linux_identity_check


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    target = OWN / "DIAGNOSTIC_CLOSURE_001.json"
    if target.exists():
        raise FileExistsError("The diagnostic closure already exists")
    profile = read(OWN / "p1/RESULT.json")
    original = read(OWN / "p1/c/RESULT.json")
    checks = {"eight_original_current_complete_diagnostic_checks_passed": len(profile["checks"]) == 8 and all(profile["checks"].values()),
              "all_original_complete_entry_checks_passed": all(original["checks"].values()),
              "all_seven_private_timing_scope_and_writer_bindings_restored": all(value for key, value in read(OWN / "p1/BINDING_RESTORATION.json").items() if key != "new_optimization_count")}
    contract = read(OWN / "p1/CONTRACT.json")
    checks["all_nine_executed_sources_still_exact"] = len(contract["sources"]) == 9 and [data_identity(Path(row["path"])) for row in contract["sources"]] == contract["sources"]
    private = read(OWN / "p1/PRIVATE_WRITER_BINDING.json")
    checks["private_writer_only_scope_AST_proof_and_actual_source_preserved"] = private["only_allowed_root_constant_changed_AST_exact"] and data_identity(private["actual_executed_source"]["path"]) == private["actual_executed_source"]
    totals = {label: sum(row["end"] - row["start"] for row in values) for label, values in profile["actual_intervals"].items()}
    checks["all_interval_sums_recomputed_without_double_counting_whole"] = totals == profile["actual_component_seconds"]
    windows, linux, leases = [], [], []
    for label, node_name, expected in (("p0", "n0", 1), ("p1", "n1", 0)):
        lease = read(OWN / (label + "_lease.json"))
        node = read(OWN / node_name / "NODE.json")
        if lease["returncode"] != expected or lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("A full diagnostic lease did not close as recorded")
        if node["live_Windows_after_wait"] or any(row["alive"] for row in node["actual_Linux_after_wait"]):
            raise AssertionError("An original diagnostic node failed its actual drain")
        for role in ("guard", "child"):
            windows.append({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]})
        windows.append(node["owner"])
        windows.extend(node["guard"].get("owned_process_identities", []))
        for row in node["Linux_resource_receipts"]:
            if not row["closed"] or row["peak_RSS_bytes"] > row["maximum_RSS_bytes"]:
                raise AssertionError("The real Linux observer was not bounded and closed")
            actual = linux_identity_check(row)
            if actual["alive"]:
                raise AssertionError("An exact Linux worker identity still runs")
            linux.append(actual)
        peaks = node["resource_peaks"]
        if peaks["Windows_tree_RSS_bytes"] > 8 * 1024**3 or peaks["GPU_total_used_bytes"] > 12 * 1024**3 or peaks["owned_output_bytes"] > 30 * 1024**3:
            raise AssertionError("An actual original resource peak exceeded its limit")
        leases.append({"label": label, "actual_child_seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
    unique = {(row["pid"], row["creation_time"]): row for row in windows}
    checks["all_saved_actual_Windows_instances_closed"] = all(closed(row) for row in unique.values())
    checks["actual_Linux_identity_now_closed"] = len(linux) == 1 and not linux[0]["alive"]
    checks["p0_scope_guard_failure_before_matching_preserved"] = read(OWN / "p0/FAILURE.json")["actual_interval_counts"] == {"GPU_request": 0, "CPU_geometry": 0, "CPU_pose": 0}
    broker = original["trace"]["broker"]
    worker_seconds = sum(row["whole_task_seconds"] for row in broker["worker_results"])
    cpu_seconds = sum(row["CPU_seconds"] for row in broker["worker_results"])
    checks["all24_native_CPU_SIFT_tasks_and_two_actual_workers_closed"] = len(broker["worker_results"]) == 24 and len(broker["saved_workers"]) == 2 and broker["closed"]
    metadata = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/mj_v4_001/V4_NEGATIVE_CLOSURE_001.json"
    publisher = read(metadata)
    checks["R110_negative_publisher_exact_instance_exited"] = closed({"pid": publisher["publisher_pid"], "creation_time": publisher["publisher_creation_time"]})
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["parent24_no_new_mechanism_or_scheduler_count"] = state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(state["acceptance_receipts"]) == 24
    if not all(checks.values()):
        raise AssertionError("A full diagnostic content/source/resource closure check failed")
    with (OWN.parent / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    overlap = totals["CPU_geometry"] + totals["CPU_pose"]
    fraction = overlap / profile["complete_query_loop_and_map_verify_seconds"]
    result = {"status": "diagnostic_complete_cross_query_same_work_overlap_deferred_before_implementation",
              "checks": checks, "actual_component_seconds": totals,
              "complete_query_phase_seconds": profile["complete_query_loop_and_map_verify_seconds"],
              "fixed_work_optimistic_CPU_overlap_seconds": overlap,
              "fixed_work_optimistic_query_phase_fraction_before_fees": fraction,
              "single_measured_trace_estimate_not_universal_parallelism_performance_theorem": True,
              "no_scheduler_implemented_or_paired_speed_result": True, "new_optimization_count": 0,
              "parent24_goal_active": True, "actual_SIFT24_task_wall_sum_seconds": worker_seconds,
              "actual_SIFT24_task_CPU_sum_seconds": cpu_seconds,
              "SIFT_parallel_task_sums_are_not_whole_wall": True,
              "two_diagnostic_lease_seconds": sum(row["actual_child_seconds"] for row in leases), "leases": leases,
              "all_actual_saved_Windows_exited": list(unique.values()), "actual_Linux_after_close": linux,
              "R110_actual_negative_publisher_exited": True,
              "scope_protection_failure_and_private_AST_scope_only_fix_preserved": True,
              "close_internal_seconds": time.perf_counter() - started,
              "next_rank_action": "audit_complete_initial_feature_extraction_consumer_source_cost_and_native_GPU_SIFT_failed_scopes_before_fresh_candidate_no_duplicate_parity_or_neighbor_variant",
              "no_new_training_GT_unseen_object_accuracy_or_production": True,
              "publisher_pid": os.getpid(), "publisher_creation_time": psutil.Process().create_time()}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "CPU_overlap_seconds": overlap,
                      "query_fraction": fraction, "SIFT_task_wall_sum_seconds": worker_seconds, "SIFT_CPU_sum_seconds": cpu_seconds,
                      "known_two_lease_seconds": result["two_diagnostic_lease_seconds"], "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
