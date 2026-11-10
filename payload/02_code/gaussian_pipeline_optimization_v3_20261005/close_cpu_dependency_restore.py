"""Verify actual publication exit and close the scoped V6 ledger decision."""

import time

STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import re
import shutil

import psutil

from review_cpu_dependency_restore import OWN, DOC, ROOT, read, atomic_json, closed, calculate_cost, data_identity


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-seconds", type=float, required=True)
    parser.add_argument("--publisher-seconds", type=float, required=True)
    parser.add_argument("--publisher-chunk", required=True)
    parser.add_argument("--previous-post-command-seconds", type=float, required=True)
    args = parser.parse_args()
    destination = OWN / "FINAL_POST_EXIT_V6_001.json"
    if destination.exists():
        raise FileExistsError("The original closure cannot be overwritten")
    parent = json.loads((DOC / "CURRENT_STATE.json").read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    review, decision, previous_cost = (read(OWN / name) for name in
        ("SELF_REVIEW_V6_001.json", "ACCEPTANCE_V6_001.json", "FINAL_COST_V6_001.json"))
    cost = calculate_cost(args.root_seconds + args.publisher_seconds)
    documents = [DOC / name for name in ("R142_CPU_DEPENDENCY_V6_RESULTS.md", "NEXT_V6_RESIDUAL_COST_RANK_BRIEF.md",
        "V6_NET_SPEED_POLICY.md", "R142_V6_CPU_RECOVERY_COMPLETION_TASK.md")]
    texts = {path.name: path.read_text(encoding="utf-8") for path in documents}
    links = sorted(set(target for body in texts.values() for target in re.findall(r"\]\((C:/[^\)]+)\)", body)))
    ledgers = [DOC / name for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md")]
    prefixes = {path.name: path.read_text(encoding="utf-8").split("\n\n")[1] for path in ledgers}
    checks = {
        "root16_and_actual_reviewer_exit": len(review["checks"]) == 16 and all(review["checks"].values()) and closed(review["reviewer"]),
        "actual_successful_publisher_instance_exited": closed(previous_cost["publisher"]) and closed(decision["publisher"]),
        "executed_review_and_publisher_sources_still_exact": all(data_identity(Path(row["path"])) == row
            for row in (review["actual_executed_review_source"], previous_cost["actual_publisher_source"])),
        "ledger32_unique_mechanisms_and32_unique_existing_receipts": parent["additional_accepted_count"] == 32
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 32
            and len(parent["acceptance_receipts"]) == len(set(parent["acceptance_receipts"])) == 32
            and all((ROOT / name).is_file() for name in parent["acceptance_receipts"]),
        "old_R134_existing_acceptance_not_newly_counted": parent["R134_V4_new_optimization_count"] == 1
            and parent["R134_existing_receipt_list_omission_corrected_new_count0"]
            and read(ROOT / parent["R134_V4_acceptance"])["new_optimization_count"] == 1,
        "one_R142_family_and_old_GPU_failure_still0": sum(name.startswith("R142") for name in parent["accepted_mechanisms"]) == 1
            and parent["R142_new_optimization_count"] == 0 and parent["R142_CPU_V6_new_optimization_count"] == decision["new_optimization_count"] == 1,
        "goal50_active_and_remaining18": parent["additional_target"] == 50 and parent["status"].startswith("active")
            and parent["additional_target"] - parent["additional_accepted_count"] == 18,
        "V6_uniform2_and_old_V5_3_failure_preserved": parent["V6_minimum_complete_stage_net_reduction"] == .02
            and parent["V5_minimum_complete_stage_net_reduction"] == .03
            and read(OWN / "V5_COST_SCREEN_001.json")["new_optimization_count"] == 0,
        "exact_final_known_cost_and_original_whole_protections": cost["net_complete_entry_reduction_under_registered_six_side_model"] >= .02
            and abs(cost["net_complete_entry_reduction_under_registered_six_side_model"] - parent[
                "R142_CPU_V6_complete_entry_net_reduction_under_registered_six_side_model"]) < 1e-14
            and cost["raw_whole_median_ratio"] <= 1.01 and cost["adjusted_whole_median_ratio"] <= 1.01
            and all(value <= 1.1 for value in cost["each_original_whole_ratios"]),
        "all_final_text_local_targets_exist_or_owned_receipt_being_published": all(Path(target).is_file() or Path(target) == destination for target in links),
        "four_current_ledger_prefixes_state32_and_net2_without_new_quality_claim": all("32/50" in body and "2.045080" in body for body in prefixes.values()),
        "current_report_and_scope_direct_text": "剩18" in texts["R142_CPU_DEPENDENCY_V6_RESULTS.md"]
            and "目标50项尚未完成" in texts["R142_CPU_DEPENDENCY_V6_RESULTS.md"]
            and decision["training_steps"] == 0 and not decision["production_deployed"]
            and not decision["unknown_object_GT_cross_hardware_generalization_verified"],
        "all_recorded_actual_Windows_instances_remain_closed": all(closed(row) for row in review["Windows_instances"]),
    }
    if not all(checks.values()):
        atomic_json(OWN / "FAILED_POST_EXIT_V6_001.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError({key: value for key, value in checks.items() if not value})
    elapsed = time.perf_counter() - STARTED
    if args.previous_post_command_seconds + elapsed + 1. > 5.:
        raise AssertionError("The declared final post-exit reserve needs a larger actual cost before publication")
    atomic_json(OWN / "FINAL_COST_V6_001.json", dict(cost, status="accepted_final_actual_publisher_exit_verified",
        actual_root_tool=previous_cost["actual_root_tool"], actual_publisher_tool={"chunk_id": args.publisher_chunk,
            "exit_code": 0, "wall_time_seconds": args.publisher_seconds}, publisher=previous_cost["publisher"],
        actual_publisher_source=previous_cost["actual_publisher_source"], final_tail_actual_so_far_seconds=args.previous_post_command_seconds + elapsed))
    decision.update(status="accepted_optional_V6_CPU_dependency_restore_DAG_actual_publisher_exit_and_counter32_verified",
        actual_counter_publication_pending=False, actual_publisher_exit_verified=True)
    atomic_json(OWN / "ACCEPTANCE_V6_001.json", decision)
    shutil.copyfile(__file__, OWN / "EXECUTED_V6_CLOSURE_SOURCE.py")
    atomic_json(destination, {"status": "actual_publisher_exited_root13_text_and_counter_checks_passed_parent32_active",
        "checks": checks, "local_text_link_targets": len(links), "additional_accepted_count": 32, "remaining": 18,
        "new_optimization_count": 1, "same_family_maximum_one": True, "cost": "FINAL_COST_V6_001.json",
        "root_review_is_independent_agent": False, "goal_status": "active", "finalizer": {
            "pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "actual_finalizer_source": data_identity(Path(__file__).resolve()),
        "actual_post_commands_and_finalizer_so_far_seconds": args.previous_post_command_seconds + elapsed,
        "last_tool_exit_check_remaining_within_reserved5seconds": True})
    print(json.dumps({"status": "parent32_active_publisher_closed", "checks": len(checks), "links": len(links),
        "net_reduction": cost["net_complete_entry_reduction_under_registered_six_side_model"]}), flush=True)


if __name__ == "__main__":
    main()
