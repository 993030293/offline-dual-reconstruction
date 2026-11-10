"""Close the rank lease and retain negative and incomplete consumers separately."""

import json
import msvcrt
from pathlib import Path
import sys
import time

import psutil

from rank_v5_existing_cost import ROOT, PARENT, OWN, read, closed, cost_headroom
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    result = read(OWN / "r0/RESULT.json")
    lease, guard = read(OWN / "r0_lease.json"), read(OWN / "r0_guard.json")
    identities = [result["own_process"],
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]},
        *guard["owned_process_identities"], *result["closed_instances"]]
    checks = {
        "rank6_checks_and_all_input_identities_kept": all(result["checks"].values())
            and result["input_identities"] == [data_identity(row["path"]) for row in result["input_identities"]],
        "rank_actual_exit_and_lease_release": lease["status"] == "child_exited" and lease["returncode"] == 0
            and bool(lease["lease_released_utc"]) and guard["status"] == "completed" and guard["returncode"] == 0,
        "rank_resource_budget_kept": guard["elapsed_seconds"] <= 10 and guard["peak_tree_RSS_bytes"] <= 2 * 1024**3,
        "every_saved_old_and_new_instance_closed": all(closed(row) for row in identities),
        "rank_executed_source_snapshot_and_identity_exact": result["source_identity"] == data_identity(Path(__file__).with_name("rank_v5_existing_cost.py"))
            and Path(result["source_identity"]["path"]).read_bytes() == (OWN / "r0/source_snapshot/rank_v5_existing_cost.py").read_bytes(),
    }
    if not all(checks.values()):
        raise AssertionError("A rank closure contract failed")
    with (PARENT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    rank_per_consumer = lease["child_elapsed_seconds"] / 2
    bounds = {}
    for key, row in result["candidates"].items():
        bounds[key] = cost_headroom(*row["native_candidate_stage_seconds"],
                                   row["known_fee_lower_bound_seconds"] + rank_per_consumer)
    if bounds["R110"]["stage_reduction_upper_bound"] >= .03:
        raise AssertionError("The fixed3percent negative cost bound was not established")
    destination = OWN / "CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError("A rank closure must use a new path")
    atomic_json(destination, {"status": "R110_V5_cost_negative_R123_single_pair_incomplete_new0",
        "checks": checks, "actual_shared_rank_lease_seconds": lease["child_elapsed_seconds"],
        "fee_equal_allocation_seconds_per_consumer": rank_per_consumer,
        "candidates": bounds, "R110_browser_and_recovery": "not_run_after_proven_cost_bound_failure",
        "R123_full1050_three_pairs_R132_R140_metadata_fault_recovery": "not_run_not_accepted",
        "R123_bound_scope": "same_saved_complete1016_single_pair_only_not_new_consumer_prediction",
        "historical_unknown_and_future_cost": "not_zero_only_further_reduce_these_upper_bounds",
        "closed_instances": identities, "actual_OS_lock_acquired_and_released": True,
        "source_identity": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started,
        "root_separate_self_review_not_independent_agent": True,
        "new_optimization_count": 0, "parent_count": 28, "goal_remains_active": True})
    print(json.dumps({"candidates": bounds, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
