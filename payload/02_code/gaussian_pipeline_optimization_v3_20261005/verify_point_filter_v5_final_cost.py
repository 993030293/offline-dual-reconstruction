"""Verify the final counter, native text, evidence links and actual resource drain."""

import json
import msvcrt
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json
from close_point_filter_v5 import BASE, OWN, DOC, read, closed, costs


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A duplicate machine-state key was introduced: " + key)
        result[key] = value
    return result


def main():
    started = time.perf_counter()
    destination = OWN / "FINAL_COST_001.json"
    if destination.exists():
        raise FileExistsError("A final cost receipt cannot overwrite prior evidence")
    state = json.loads((DOC / "CURRENT_STATE.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    candidate = read(OWN / "ACCEPTANCE_V5_001.json")
    post = read(OWN / "x0/RESULT.json")
    original = read(BASE / "point_filter_predicate_001/campaign_001/RESULT.json")
    checks = {"parent28_all_unique_mechanisms_and_receipts": state["additional_accepted_count"] == len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 28,
        "only_one_R113_and_no_R151_accepted": sum(name.startswith("R113") for name in state["accepted_mechanisms"]) == 1 and not any(name.startswith("R151") for name in state["accepted_mechanisms"]),
        "R113_new1_R151_new0_V5_3percent_and_quality_scope_kept": candidate["new_optimization_count"] == 1 and state["R113_V5_new_optimization_count"] == 1 and state["R151_new_optimization_count"] == 0 and state["V5_minimum_complete_stage_net_reduction"] == .03 and not state["production_modified"] and not candidate["production_deployed"],
        "actual_post_exit11_all_passed": len(post["checks"]) == 11 and all(post["checks"].values())}
    recomputed = costs(original, candidate["known_all_metered_fees_before_finalizer_seconds"] + candidate["reserved_remaining_measured_execution_seconds"])
    checks["complete_stage_lower_bound_and_all_whole_costs_recomputed_exact"] = recomputed == candidate["costs"] and recomputed["complete_phase_reduction_lower_bound"] >= .03 and recomputed["complete_whole_median_ratio"] <= 1.01 and max(recomputed["each_whole_ratio"]) <= 1.1
    instances = list(post["all_saved_earlier_instances_exited"])
    closed_envelope_seconds = 0.
    for label in ("F0", "X0"):
        lease, guard = read(OWN / ("LEASE_" + label + ".json")), read(OWN / ("GUARD_" + label + ".json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] or guard["returncode"]:
            raise ValueError("The final positive-branch instance did not actually exit")
        instances.extend(guard["owned_process_identities"])
        instances.extend({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]} for role in ("guard", "child"))
        closed_envelope_seconds += lease["child_elapsed_seconds"]
    unique = {(row["pid"], row["creation_time"]): row for row in instances}
    checks["all_saved_actual_PID_ctime_including_finalizer_and_post_checker_exited"] = all(closed(row) for row in unique.values())
    headers = ("PLAN.md", "OPTIMIZE_CHECKLIST.md", "CANDIDATE_BOARD.md", "GENERALIZATION_PROTOCOL.md")
    texts, linked = {}, set()
    for name in (*headers, "ENGINEERING_ACCEPTANCE_V5.md", "R113_V5_OPTIONAL_FILTER_TASK.md", "R113_V5_OPTIONAL_FILTER_RESULTS.md", "R151_NATIVE_QUALITY_CACHE_RESULTS.md", "NEXT_V5_COST_RANK_BRIEF.md"):
        content = (DOC / name).read_text(encoding="utf-8")
        if not content.strip() or "\ufffd" in content:
            raise ValueError("A current native-text report is empty or has replacement characters")
        texts[name] = {"native_text_read": True, "characters": len(content)}
        if name in headers and "28/50" not in content[:2600]:
            raise ValueError("A current frontier header has not been updated to28")
        if name == "R113_V5_OPTIONAL_FILTER_RESULTS.md" and not all(token in content for token in ("3.110480", "0.677879", "128.7659700", "33项", "未保存耗时仍未计量")):
            raise ValueError("The R113 text contradicts its actual cost or claim boundary")
        for url in re.findall(r"\]\((C:/[^)]+)\)", content[:3000] if name in headers else content):
            if not Path(url).is_file():
                raise FileNotFoundError("A current source/evidence link is missing: " + url)
            linked.add(url)
    checks["all_nine_current_texts_and_modified_headers_native_read_and_consistent"] = len(texts) == 9
    checks["all_current_local_evidence_and_report_links_resolve"] = bool(linked)
    checks["all28_preserved_acceptance_records_parse_as_machine_objects"] = all(isinstance(read(ROOT / relative), dict) for relative in state["acceptance_receipts"])
    owner = read(BASE / "coordination_001/LOCK_OWNER.json")
    if owner["status"] != "released" or not owner.get("lease_released_utc"):
        raise ValueError("The last measured task has not released its resource lease")
    with (BASE / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["OS_exclusive_resource_lock_actually_acquired_and_released"] = True
    checks["closed_tail_envelopes_and_saved_precheck_command_within10second_reserve_before_own_exit"] = closed_envelope_seconds == candidate["actual_finalizer_and_post_exit_closed_envelopes_seconds"] and closed_envelope_seconds + .2288719 + time.perf_counter()-started < 10
    if not all(checks.values()):
        raise ValueError("A final counter, semantic, text or actual resource check failed: " + str([name for name, value in checks.items() if not value]))
    atomic_json(destination, {"status": "final_counter28_native_text_links_cost_lower_bound_and_prior_exit_checks_passed_own_exit_pending_tool_result",
        "checks": checks, "new_optimization_count": 1, "parent_additional_accepted_count": 28, "remaining": 22, "goal_active": True,
        "conservative_costs": recomputed, "native_text_checks": texts, "resolved_local_links": len(linked),
        "all_saved_prior_and_final_instances_exited": list(unique.values()), "actual_closed_finalizer_and_post_checker_seconds": closed_envelope_seconds,
        "saved_precheck_closed_command_seconds": .2288719, "reserved_remaining_execution_seconds": 10.,
        "final_command_actual_exit_pending": True, "final_command_closed_tool_wall_seconds": None,
        "all_known_costs_plus_verified_reserve_not_total_research_wall": True,
        "unmetered_text_research_and_actual_final_write_tail": "unknown_not_assumed_zero_no_total_ROI_claim",
        "root_self_review_not_independent_agent": True, "no_new_training_GT_unseen_full_chain_latest_combo_or_deployment": True,
        "current_node_before_receipt_seconds": time.perf_counter()-started})
    print({"status": "final_cost_checks_passed", "checks": len(checks), "parent": 28, "remaining": 22,
           "stage_reduction_lower_bound": recomputed["complete_phase_reduction_lower_bound"], "closed_prior_instances": len(unique)}, flush=True)


if __name__ == "__main__":
    main()
