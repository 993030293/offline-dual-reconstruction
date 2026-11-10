"""Publish a conservative negative screen including actual text-read costs."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "v6_residual_cost_003"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ended(pid, born):
    try:
        return psutil.Process(pid).create_time() != born
    except psutil.NoSuchProcess:
        return True


def main():
    screen, review = read(OWN / "p0/RESULT.json"), read(OWN / "SELF_REVIEW_V6_001.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if not all(review["checks"].values()) or state["additional_accepted_count"] != 33:
        raise AssertionError("The completed quality review or unchanged parent count differs")
    # Tool receipts supply these observed command durations, not document hashes.
    text_rows = [
        ("6af693", .162116, "shared"), ("44c334", .1675148, "shared"),
        ("ab2cc5", .2140021, "R111"), ("ed5d1a", .1708724, "shared"),
        ("9554bc", .2411082, "R111"), ("122a1b", .2671498, "shared"),
        ("32dc5c", .1786221, "R111"), ("d9104d", .1891324, "R111"),
        ("7a9875", .2425054, "R111"), ("bd45e8", .225903, "R111"),
        ("360790", .4594953, "R111"), ("793fbd", .1898507, "R111"),
        ("a85907", 1.0989535, "R111"), ("05725b", .4389402, "R111"),
        ("d5e677", .1828457, "R111"), ("9cde12", .3238027, "R111"),
        ("5686f0", .2788287, "R111"), ("f392c2", .2357568, "R111"),
        ("f87759", .1835267, "R111"), ("f71929", .1635676, "R111"),
        ("d85b18", .2190386, "R111"),
    ]
    shared_text = sum(seconds for _, seconds, owner in text_rows if owner == "shared")
    sparse_text = sum(seconds for _, seconds, owner in text_rows if owner == "R111")
    p0 = read(OWN / "p0_lease.json")
    common = (p0["child_elapsed_seconds"] + 3 + shared_text) / 3
    stages = {name: row["raw_stage_seconds"] for name, row in screen["candidates"].items()}
    old_common = screen["common_current_audit_so_far_and_registered3tail_seconds_each"]
    fees = {
        "R110": screen["candidates"]["R110"]["known_extra_seconds"] - old_common + common,
        "R111": review["old_extra_complete_known_seconds"] + common + sparse_text,
        "R118": screen["candidates"]["R118"]["known_extra_seconds"] - old_common + common,
    }
    instances = []
    for name in ("sparse", "sparse2", "review"):
        lease, guard = read(OWN / (name + "_lease.json")), read(OWN / (name + "_GUARD.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("A measured completion node has not released")
        fees["R111"] += lease["child_elapsed_seconds"]
        for pid, born in [(lease["guard_pid"], lease["guard_creation_time"]),
                          (lease["child_pid"], lease["child_creation_time"])]:
            instances.append({"pid": pid, "creation_time": born, "ended": ended(pid, born)})
        for row in guard["owned_process_identities"]:
            instances.append({**row, "ended": ended(row["pid"], row["creation_time"])})
    bounds = {name: {"known_extra_cost_and_registered_common_allowance_seconds": fees[name],
        "net_complete_stage_upper_bound_before_this_publisher_and_following_fees":
        1 - (stage["candidate"] + fees[name] / 6) / (stage["native"] + fees[name] / 6)}
        for name, stage in stages.items()}
    checks = {"quality_and_full_function_recovery_passed_not_speed_acceptance": all(review["checks"].values()),
        "all_three_net_upper_bounds_below_uniform2percent": all(row["net_complete_stage_upper_bound_before_this_publisher_and_following_fees"] < .02 for row in bounds.values()),
        "known_failed_and_text_execution_costs_not_waived": len(text_rows) == 21 and sparse_text > 5,
        "all_review_and_completion_owned_instances_closed": all(row["ended"] for row in instances),
        "parent33_of50_still_active_no_candidates_added": state["additional_accepted_count"] == 33 and state["status"].startswith("active"),
        "old_results_not_rewritten": read(OUT / "sparse_ply_oracle_001/campaign_002/RESULT.json")["status"] == "complete_export_cost_guard_failed"}
    if not all(checks.values()):
        raise AssertionError("The negative cost publication is inconsistent")
    atomic_json(OWN / "NEGATIVE_COST_CLOSURE_V6_001.json", {
        "status": "all_three_old_candidates_closed_negative_uniform_V6_known_cost_upper_bounds",
        "checks": checks, "candidates": bounds,
        "native_text_read_receipts": [{"tool_chunk_id": chunk, "seconds": seconds, "owner": owner} for chunk, seconds, owner in text_rows],
        "known_text_read_execution_seconds": shared_text + sparse_text,
        "common_p0_actual_three_way_charge_and_registered3tail_seconds_each": common,
        "review_is_independent_agent": False, "owned_postreview_instances": instances,
        "R111_full_public_and_actual_SIGINT_recovery_verified_speed_gate_failed": True,
        "R111_SIGINT_file_already_complete_not_claimed_partial_physical_write": True,
        "remaining_costs_only_lower_speed_upper_bounds": True,
        "unknown_text_research_time_not_zero_no_total_R_and_D_ROI": True,
        "actual_seconds_before_publication": time.perf_counter() - STARTED,
        "publisher_exit_not_yet_proved": True, "new_optimization_count": 0,
        "parent_goal_complete": False})
    print(json.dumps({"checks": checks, "candidates": bounds}), flush=True)


if __name__ == "__main__":
    main()
