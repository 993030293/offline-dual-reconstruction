"""Rank saved unaccepted consumers using preserved measured cost lower bounds."""

import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = PARENT / "v5_cost_rank_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def cost_headroom(native, candidate, known):
    if not all(math.isfinite(v) and v >= 0 for v in (native, candidate, known)):
        raise ValueError("Nonfinite or negative measured time")
    maximum_fee = 6 * ((native - candidate) / .03 - native)
    adjusted = [native + known / 6, candidate + known / 6]
    return {"native_candidate_stage_seconds": [native, candidate],
            "known_fee_lower_bound_seconds": known,
            "adjusted_stage_seconds_lower_bound": adjusted,
            "stage_reduction_upper_bound": 1 - adjusted[1] / adjusted[0],
            "maximum_total_fee_at_fixed3percent_seconds": maximum_fee,
            "remaining_fee_headroom_upper_bound_seconds": maximum_fee - known}


def main():
    started = time.perf_counter()
    output = OWN / "r0"
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("Insufficient free disk")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    paths = [PARENT / name for name in (
        "machine_json_001/campaign_001/RESULT.json", "mj_v4_001/V4_NEGATIVE_CLOSURE_001.json",
        "mj_v4_001/a1/RESULT.json", "ud001/s0/RESULT.json", "ud001/r1/REVIEW.json",
        "ud001/POST_EXIT_CLOSURE_001.json")]
    formal, mj, mj_review, shared, shared_review, shared_closed = [read(path) for path in paths]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    shutil.copyfile(__file__, snapshots / Path(__file__).name)
    phases = {mode: statistics.median(pair["rows"][mode]["phase_seconds"] for pair in formal["pairs"])
              for mode in ("native", "compact")}
    dataset = [row["result"]["operation_seconds"] for row in shared["rows"]]
    rows = mj["all_saved_actual_Windows_instances_closed"] + shared_closed["instances"]
    rows.append({"pid": mj["publisher_pid"], "creation_time": mj["publisher_creation_time"]})
    rows = list({(row["pid"], row["creation_time"]): row for row in rows}.values())
    checks = {
        "current_parent28_uniform3percent_and_no_duplicate_families": state["additional_accepted_count"] == 28
            and state["V5_minimum_complete_stage_net_reduction"] == .03
            and not any(name.startswith(("R110", "R123")) for name in state["accepted_mechanisms"]),
        "R110_three_saved_alternating_complete_pairs_and_exact_saved_medians": len(formal["pairs"]) == 3
            and phases == {mode: formal["medians"][mode]["phase_seconds"] for mode in phases}
            and all(pair["all1016_paths_and_complete_RECEIPT_exact"] for pair in formal["pairs"]),
        "R110_previous_six_source_and_output_checks_passed_not_new_browser_checks": len(mj_review["checks"]) == 6
            and all(mj_review["checks"].values()) and mj["new_optimization_count"] == 0,
        "R123_single_full1016_saved_pair_not_formal_or_full1050": len(shared["rows"]) == 2
            and shared["complete_file_pairs"] == 1025 and all(shared["checks"].values())
            and shared_review["not_three_pair_medians_or_full1050_R132_R140_result"],
        "R123_saved14_and11_root_checks_and_full_cost_closure_kept": len(shared_review["checks"]) == 14
            and all(shared_review["checks"].values()) and len(shared_closed["checks"]) == 11
            and all(shared_closed["checks"].values()),
        "all_saved_old_instances_and_R110_publisher_actually_exited": all(closed(row) for row in rows),
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_RANK.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A preserved rank input or process condition failed")
    candidates = {
        "R110": cost_headroom(phases["native"], phases["compact"],
                               mj["additional_known_fee_lower_bound_before_publisher_exit_seconds"]),
        "R123": cost_headroom(*dataset, shared_closed["known_family_fee_lower_bound_after_both_review_exit_seconds"]),
    }
    atomic_json(output / "RESULT.json", {
        "status": "rank_inputs_checked_actual_lease_exit_cost_addition_pending_not_acceptance",
        "checks": checks, "candidates": candidates, "input_identities": [data_identity(p) for p in paths],
        "source_identity": data_identity(Path(__file__)), "closed_instances": rows,
        "own_process": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "common_rank_fee_allocation": "one_half_each_two_consumers_actual_closed_lease_only",
        "unmetered_historical_text_research_publisher_tail": "unknown_not_zero",
        "new_optimization_count": 0, "parent_count": 28, "goal_remains_active": True,
        "internal_before_result_seconds": time.perf_counter() - started,
    })
    print(json.dumps({"checks_passed": len(checks), "candidates": candidates, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
