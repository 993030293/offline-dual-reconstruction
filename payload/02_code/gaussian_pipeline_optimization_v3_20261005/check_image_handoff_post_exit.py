"""Confirm the actual finalizer exit, one counter change and native text handoff."""

from datetime import datetime, timezone
from pathlib import Path
import msvcrt
import re
import time

import psutil

from run_open_image_handoff_campaign import OUT
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    docs = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    state = read(docs / "CURRENT_STATE.json")
    before = read(OUT / "STATE_BEFORE_PROMOTION_001.json")
    receipt, cost = read(OUT / "ACCEPTANCE_WORK_001.json"), read(OUT / "FINAL_COST_001.json")
    report = (docs / "R156_OPEN_IMAGE_HANDOFF_RESULTS.md").read_text(encoding="utf-8")
    frontiers = [docs / name for name in ("PLAN.md", "CANDIDATE_BOARD.md", "GENERALIZATION_PROTOCOL.md", "OPTIMIZE_CHECKLIST.md")]
    texts = [path.read_text(encoding="utf-8").split("\n", 4)[:4] for path in frontiers]
    own = receipt["acceptance_finalizer_own_process"]
    leases = [OUT.parent / "shf001_preflight_lease.json", *[OUT / (name + "_lease.json") for name in ("g0", "c0_first", "c0_rest", "v1", "v2", "v3", "g1", "r0", "r1")]]
    owners = [owner for row in [read(path) for path in leases] for owner in (
        {"pid": row["guard_pid"], "creation_time": row["guard_creation_time"]},
        {"pid": row["child_pid"], "creation_time": row["child_creation_time"]})]
    source = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with source.open("a+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            lock_free = read(source.with_name("LOCK_OWNER.json"))["status"] == "released"
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    checks = {
        "actual_passing_publisher_and_acceptance_finalizer_exited": windows_closed(receipt["own_process"]) and windows_closed(own),
        "exactly_one_new_unique_mechanism_and_one_receipt_29_to30": before["additional_accepted_count"] == 29 and state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 30 and state["accepted_mechanisms"][:-1] == before["accepted_mechanisms"] and state["acceptance_receipts"][:-1] == before["acceptance_receipts"] and state["accepted_mechanisms"][-1] == receipt["accepted_mechanism"],
        "receipt_work_scope_net10percent_and_speed_failed_preserved": receipt["new_optimization_count"] == state["R156_new_optimization_count"] == 1 and receipt["net_source_read_work_under_registered_six_side_model"] >= .10 and not receipt["speed3percent_gate_passed"] and not state["R156_speed_gate_passed"],
        "all_original_target50_prior35_and_V5_gates_kept": state["additional_target"] == 50 and state["prior_accepted_count"] == 35 and state["V5_minimum_complete_stage_net_reduction"] == .03 and state["V5_maximum_whole_median_ratio"] == 1.01 and state["V5_maximum_each_whole_ratio"] == 1.1,
        "all_native_frontier_heads_current30_and20_remaining": all("30/50" in "\n".join(lines) and "20" in "\n".join(lines) and "R156" in "\n".join(lines) for lines in texts),
        "report_actual_complete_values_errors_limits_and_publisher_failure_retained": all(token in report for token in ("4064", "3048", "10.889214", "0.130851", "110.8", "15342", "16256", "v1", "v2", "r0", "8个本地引用", "0.268376", "30/50")),
        "all_report_actual_local_references_resolve": all(Path(link).is_file() for link in re.findall(r"\]\((C:/[^)]+)\)", report)),
        "all10_original_leases_released_and_native_windows_owners_closed": all(windows_closed(owner) for owner in owners) and all(read(path)["lease_released_utc"] for path in leases),
        "benchmark_lock_reacquired_and_released_no_other_owner": lock_free,
        "next_explore_no_active_candidate_single_chat_no_production": state["primary_optimize_submode"] == "explore" and state["active_candidates"] == [] and state["execution_mode"] == "single_chat_no_subagents_or_side_chat_dispatch" and not state["R156_production_default_deployed"],
        "actual_finalizer_source_identity_still_exact": receipt["finalizer_source"] == data_identity(receipt["finalizer_source"]["path"]),
    }
    tail = {"actual_acceptance_finalizer_5e9eca": .2028761,
            "actual_native_parent_state_copy_7676b2": .1667249,
            "actual_failed_hunk_order_native_source_read_783014": .2249391}
    final_fee = cost["known_fee_plus_registered10_reserve_seconds"] + sum(tail.values())
    raw = receipt["raw_whole_medians"]
    adjusted = {mode: seconds + final_fee / 6 for mode, seconds in raw.items()}
    checks["all_tail_known_fee_kept_whole_ratio_still_bounded_speed_still_failed"] = adjusted["handoff"] / adjusted["control"] <= 1.01 and 1 - adjusted["handoff"] / adjusted["control"] < .03
    cost.update(status="actual_acceptance_finalizer_exited_counter30_and_native_frontier_verified_last_checker_exit_pending",
                finalizer_actual_tool_exit={"chunk_id": "5e9eca", "returncode": 0, "seconds": .2028761},
                actual_nonoverlapping_tail_execution_fees=tail, known_fee_plus_registered10_reserve_seconds=final_fee,
                final_known_fee_adjusted_whole_medians=adjusted, new_parent_count=30)
    atomic_json(OUT / "FINAL_COST_001.json", cost)
    process = psutil.Process()
    result = {"status": "actual_finalizer_and_publishers_exited_one_scoped_work_accepted_parent30_active" if all(checks.values()) else "failed_verification",
        "checks": checks, "new_optimization_count": 1, "parent_count": 30, "additional_target": 50, "remaining": 20,
        "actual_last_checker_own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "original10_leases_closed": True, "goal_remains_active": True, "review_is_independent_agent": False,
        "known_fee_plus_registered10_reserve_seconds_before_last_check_exit": final_fee,
        "report_local_links_verified": len(re.findall(r"\]\((C:/[^)]+)\)", report)),
        "internal_before_result_seconds": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": data_identity(Path(__file__))}
    atomic_json(OUT / "FINAL_POST_EXIT_001.json", result)
    print({"checks": checks, "parent": 30, "new": 1, "remaining": 20, "goal": "active"})
    if not all(checks.values()):
        raise AssertionError("An actual finalizer exit, text or unique counter check failed")


if __name__ == "__main__":
    main()
