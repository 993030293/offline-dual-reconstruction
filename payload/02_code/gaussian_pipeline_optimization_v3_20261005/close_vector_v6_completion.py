"""Close the actual publication after direct text and counter verification."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import re
import shutil

import psutil

from publish_vector_v6_completion import final_cost
from review_vector_v6_completion import OWN, DOC, ROOT, BASE, read, atomic_json, closed, data_identity


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def main():
    destination = OWN / "FINAL_POST_EXIT_V6_001.json"
    if destination.exists():
        raise FileExistsError("An original actual closure cannot be replaced")
    parent = json.loads((DOC / "CURRENT_STATE.json").read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    review, decision, prior_cost = (read(OWN / name) for name in
        ("SELF_REVIEW_V6_001.json", "ACCEPTANCE_V6_001.json", "FINAL_COST_V6_001.json"))
    calculated = final_cost(45.6900952, .4238773)
    documents = {name: (DOC / name).read_text(encoding="utf-8") for name in
        ("R123_V6_COMPLETE_SAMPLING_RESULTS.md", "R123_V6_COMPLETION_TASK.md", "NEXT_V6_RESIDUAL_COST_002_BRIEF.md")}
    links = sorted(set(target for body in documents.values() for target in re.findall(r"\]\((C:/[^\)]+)\)", body)))
    prefixes = [(DOC / name).read_text(encoding="utf-8").split("\n\n")[1] for name in
        ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md")]
    checks = {
        "root15_all_passed_and_actual_review_instance_closed": len(review["checks"]) == 15 and all(review["checks"].values()) and closed(review["reviewer"]),
        "actual_successful_publisher_instance_closed": closed(prior_cost["publisher"]) and closed(decision["publisher"]),
        "review_and_publication_executed_sources_exact": all(data_identity(Path(row["path"])) == row
            for row in (review["actual_executed_review_source"], prior_cost["actual_publisher_source"])),
        "all150_saved_old_and_new_instances_still_closed": len(review["actual_closed_instances"]) == 150
            and all(closed(row) for row in review["actual_closed_instances"]),
        "33_unique_mechanisms_and33_existing_unique_receipts": parent["additional_accepted_count"] == 33
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 33
            and len(parent["acceptance_receipts"]) == len(set(parent["acceptance_receipts"])) == 33
            and all((ROOT / name).is_file() for name in parent["acceptance_receipts"]),
        "single_R123_new1_old_V5_family0_preserved": sum(name.startswith("R123") for name in parent["accepted_mechanisms"]) == 1
            and parent["R123_V6_new_optimization_count"] == decision["new_optimization_count"] == 1
            and parent["R123_vector_new_optimization_count"] == 0 and read(BASE / "FINAL_POST_EXIT_001.json")["new_optimization_count"] == 0,
        "V6_net2_and_original_raw_adjusted_whole_protections": calculated["net_complete_stage_reduction_under_registered_six_side_model"] >= .02
            and calculated["raw_whole_median_ratio"] <= 1.01 and calculated["adjusted_whole_median_ratio"] <= 1.01
            and all(value <= 1.1 for value in calculated["each_original_whole_ratios"]),
        "exact_final_parent_cost_and_metric_fields": abs(calculated["net_complete_stage_reduction_under_registered_six_side_model"]
            - parent["R123_V6_net_complete_stage_reduction_under_registered_six_side_model"]) < 1e-14
            and abs(calculated["known_extra_execution_seconds_including_declared_allowances"]
                - parent["R123_V6_known_extra_cost_plus_allowances_and_conservative_root_ack_envelope_seconds"]) < 1e-10,
        "all_local_text_targets_exist_or_owned_current_receipt": all(Path(target).is_file() or Path(target) == destination for target in links),
        "four_current_ledger_prefixes_match33_and_net2point372480": all("33/50" in text and "2.372480" in text for text in prefixes),
        "direct_text_report_scope_old_evidence_memory_and_unfinished50_kept": "剩17" in documents["R123_V6_COMPLETE_SAMPLING_RESULTS.md"]
            and "50项目标尚未完成" in documents["R123_V6_COMPLETE_SAMPLING_RESULTS.md"]
            and decision["original_memory_increase_preserved_not_a_memory_optimization"]
            and decision["original_observed_formal_and_component_evidence_reused_not_new_fresh_or_blind"],
        "goal_active50_remaining17_no_training_production_or_GT_claim": parent["status"].startswith("active")
            and parent["additional_target"] == 50 and decision["training_steps"] == 0 and not decision["production_deployed"]
            and not decision["unknown_object_GT_cross_hardware_or_training_quality_verified"],
    }
    if not all(checks.values()):
        atomic_json(OWN / "FAILED_FINAL_TEXT_V6_001.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError({key: value for key, value in checks.items() if not value})
    elapsed = time.perf_counter() - STARTED
    if .5645659 + elapsed + 1. > 5.:
        raise AssertionError("The original post-exit reserve needs actual excess charged")
    atomic_json(OWN / "FINAL_COST_V6_001.json", dict(calculated, status="final_actual_publisher_exit_and_counter33_verified",
        root_actual_tools=prior_cost["root_actual_tools"], root_observed_terminal_snapshot_rounded_up_UTC=prior_cost[
            "root_observed_terminal_snapshot_rounded_up_UTC"], root_observed_envelope_upper_seconds=45.6900952,
        actual_publisher_tool={"chunk_id": "d0e9ef", "exit_code": 0, "wall_time_seconds": .4238773},
        publisher=prior_cost["publisher"], actual_publisher_source=prior_cost["actual_publisher_source"],
        final_post_native_seconds_so_far=.5645659 + elapsed))
    decision.update(status="accepted_optional_V6_strict_SIMD_complete_sampling_actual_publisher_exit_and_counter33_verified",
        actual_counter_publication_pending=False, actual_publisher_exit_verified=True)
    atomic_json(OWN / "ACCEPTANCE_V6_001.json", decision)
    shutil.copyfile(__file__, OWN / "EXECUTED_CLOSURE_V6.py")
    atomic_json(destination, {"status": "actual_publisher_exited_root12_text_and_counter_checks_parent33_active",
        "checks": checks, "actual_text_local_link_targets": len(links), "additional_accepted_count": 33, "remaining": 17,
        "new_optimization_count": 1, "goal_status": "active", "root_review_is_independent_agent": False,
        "cost": "FINAL_COST_V6_001.json", "actual_finalizer_source": data_identity(Path(__file__).resolve()),
        "finalizer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "final_post_native_seconds_so_far": .5645659 + elapsed, "last_native_tool_exit_verification_remains_within_reserved5seconds": True})
    print(json.dumps({"status": "parent33_active_publisher_closed", "checks": len(checks), "links": len(links),
        "net_stage_reduction": calculated["net_complete_stage_reduction_under_registered_six_side_model"]}), flush=True)


if __name__ == "__main__":
    main()
