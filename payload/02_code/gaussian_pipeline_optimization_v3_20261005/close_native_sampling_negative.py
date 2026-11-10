"""Publish a necessary-cost rejection and verify the completed review and OS lease."""

import json
import msvcrt
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
DOC = ROOT/"research/2026-10-05-gaussian-optimization-round3-50"
SHARED = OWNED.with_name("coordination_001")
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    output = OWNED/"FINAL_COST_NEGATIVE_001.json"
    if output.exists():
        raise FileExistsError("The necessary-cost closure must be a fresh artifact")
    review = read(OWNED/"r0/REVIEW.json")
    checks = {"forty_complete_science_sources_process_and_necessary_cost_review_checks": len(review["checks"]) == 40 and all(review["checks"].values())}
    lease = read(OWNED/"R0_LEASE.json")
    review_identity = read(OWNED/"r0/PROCESS_IDENTITY.json")
    identities = [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                  {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}, review_identity]
    observed = []
    for row in identities:
        alive = False
        try:
            process = psutil.Process(row["pid"])
            alive = process.create_time() == row["creation_time"] and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        observed.append(dict(row, alive=alive))
    checks["root_review_actual_PID_creation_time_instances_exited"] = not any(r["alive"] for r in observed)
    checks["eighth_actual_lease_released_after_successful_review"] = lease["returncode"] == 0 and bool(lease.get("lease_released_utc"))
    owner = read(SHARED/"LOCK_OWNER.json")
    checks["shared_latest_owner_actual_released"] = owner["status"] == "released" and owner["owner"] == "root_R123_native_cost_review_001" and bool(owner.get("lease_released_utc"))
    with (SHARED/"exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["OS_resource_lock_actually_free"] = True
    fees = dict(review["measured_nonoverlapping_fee_lower_bound_seconds"])
    old_failure = fees.pop("old_failed_closure")
    old_final = fees.pop("old_final_closure")
    fees["old_closure_observed_elapsed_once"] = max(old_failure, old_final)
    fees["current_root_review_actual_complete_lease"] = lease["child_elapsed_seconds"]
    fees.pop("current_root_review_before_publication")
    fees["post_review_closure_before_publication"] = time.perf_counter()-started
    allocation = sum(fees.values())/6
    medians = {key: {mode: value+allocation for mode, value in rows.items()} for key, rows in review["raw_medians"].items()}
    upper = {key: 1-row["native_DLL"]/row["native"] for key, row in medians.items()}
    checks["necessary_complete_cost_stage_upper_below5percent"] = upper["complete_reference_preparation_stage_seconds"] < .05
    checks["whole_median_kept_nonincrease"] = upper["whole_seconds"] >= 0
    checks["old_identical_closure_elapsed_not_counted_twice"] = fees["old_closure_observed_elapsed_once"] == old_failure == old_final
    state = read(DOC/"CURRENT_STATE.json")
    checks["parent_active_count22_and_candidate_zero_kept"] = state["additional_accepted_count"] == 22 and state["R123_native_new_optimization_count"] == 0
    payload = {"status": "closed_negative_complete_cost" if all(checks.values()) else "failed_verification",
        "checks": checks, "review_kind": "root_single_chat_self_review_and_post_exit_check", "independent_agent_review": False,
        "raw_medians": review["raw_medians"], "raw_reductions": review["raw_reductions"],
        "measured_nonoverlapping_required_fee_lower_bound_seconds": fees, "each_side_fee_lower_bound_seconds": allocation,
        "medians_with_required_measured_fee_lower_bound": medians, "improvement_upper_bounds_after_required_cost": upper,
        "cost_review_correction": "Old failed/final closure artifacts retain the identical elapsed field without separately measured events; charge that observed lower bound once. Replace review-internal time by its complete lease, preserve R0 and all failed evidence. Unmeasured cost is nonnegative and cannot rescue the stage gate.",
        "root_review_actual_instances": observed, "total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds": sum(fees.values())+sum(r["whole_seconds"][mode] for r in read(OWNED/"c1/RESULT.json")["pairs"] for mode in ("native", "native_DLL")),
        "new_optimization_count": 0, "parent_additional_accepted_count": 22, "parent_goal_complete": False,
        "publisher": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "publication_and_post_exit_cost_not_finally_measured_and_cannot_rescue_negative": True}
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks), "cost_reduction_upper": upper, "new_optimization_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
