"""Publish complete measured costs and verify native text and actual process exits."""

import argparse
import json
import msvcrt
import re
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
SHARED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gone(row):
    try:
        return abs(psutil.Process(row["pid"]).create_time() - row["creation_time"]) >= .001
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cost", "publication"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = OWN / ("COST_PUBLICATION_001.json" if args.stage == "cost" else "FINAL_PUBLICATION_001.json")
    if output.exists():
        raise FileExistsError("Preserve each final cost and publication execution")
    current = psutil.Process()
    execution_path = OWN / (args.stage.upper() + "_EXECUTION_001.json")
    execution = {"pid": current.pid, "creation_time": current.create_time(),
                 "actual_self_and_parents": [{"pid": item.pid, "creation_time": item.create_time()}
                                            for item in [current] + current.parents()]}
    atomic_json(execution_path, execution)
    (OWN / (args.stage.upper() + "_PUBLISHER_SOURCE_001.py")).write_bytes(Path(__file__).read_bytes())
    review = load(OWN / "r0/RESULT.json")
    check = {"all77_root_self_review_checks_passed": len(review["checks"]) == 77 and all(review["checks"].values())}
    lease = load(OWN / "REVIEW_LEASE_001.json")
    reviewer = load(OWN / "r0/EXECUTION.json")
    instances = [{"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]} for prefix in ("guard", "child")]
    instances.append({"pid": reviewer["pid"], "creation_time": reviewer["creation_time"]})
    check["actual_reviewer_three_Windows_instances_and_review_lease_closed"] = all(gone(row) for row in instances)
    check["actual_review_lease_terminal_success_released"] = lease["status"] == "child_exited" and lease["returncode"] == 0 and bool(lease["lease_released_utc"])
    owner = load(SHARED / "LOCK_OWNER.json")
    check["actual_final_owner_record_released"] = owner["status"] == "released" and owner["owner"] == "root_R141_complete_cost_source_resource_review_001"
    with (SHARED / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            check["actual_shared_OS_lock_available"] = True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    if args.stage == "cost":
        formal = load(OWN / "c1/RESULT.json")
        for name in ("native_quality_workspace.py", "run_raw_quality_workspace.py", "run_quality_workspace_campaign.py",
                     "quality_histogram.py", "run_raw_histogram_quality.py", "run_raw_calibration.py"):
            rows = [row for row in formal["sources"] if Path(row["path"]).name == name]
            check[name + "_current_frozen_identity"] = len(rows) == 1 and rows[0] == data_identity(Path(rows[0]["path"]))
        prior = review["known_shared_fees_before_review_publication_seconds"] + lease["child_elapsed_seconds"]
        elapsed = time.perf_counter() - started
        shared = prior + elapsed
        medians = {name: {mode: seconds + shared / 6 for mode, seconds in values.items()}
                   for name, values in formal["medians"].items()}
        reductions = {name: 1 - values["workspace"] / values["native"] for name, values in medians.items()}
        check["all_known_fees_complete_phase_reduces5percent"] = reductions["phase_seconds"] >= .05
        check["whole_median_nonincrease_and_each_whole_at_most1point1"] = reductions["whole_seconds"] >= 0 and all(
            pair["whole_seconds"]["workspace"] + shared / 6 <= 1.1 * (pair["whole_seconds"]["native"] + shared / 6)
            for pair in formal["pairs"])
        check["parent21_candidate0_unique_receipts_no_production_edit"] = (lambda state: state["additional_accepted_count"] == 21
            and state["R141_new_optimization_count"] == 0 and not state["production_modified"])(load(DOC / "CURRENT_STATE.json"))
        if not all(check.values()):
            raise AssertionError(json.dumps(check))
        payload = {"status": "accepted_scoped_engineering_complete_cold_RAW_three_pairs_all_original_science_errors_recovery_external_quality",
                   "checks": check, "mechanism": "R141_PRIVATE_NATIVE_QUALITY_THREAD_DESTINATION_PLANES", "accepted_count_contribution": 1,
                   "minimum_phase_reduction": .05, "fresh_complete_pairs": 3, "review_kind": "root_self_review_no_subagents",
                   "all_measured_shared_fees_through_cost_publication_seconds": shared,
                   "fees_accounting": "all_five_campaign_envelopes_minus_six_already_charged_leaf_envelopes_minus_actual_once_comparison_plus_actual_review_envelope_plus_current_cost_publication_elapsed_six_side_split",
                   "all_measured_shared_fees_before_publication_seconds": prior,
                   "complete_phase_and_whole_medians_seconds": medians, "complete_reductions": reductions,
                   "raw_complete_pairs_before_shared_fees": formal["medians"],
                   "external_complete_medians_seconds": review["external_complete_medians"],
                   "external_complete_reduction": review["external_complete_reduction"],
                   "cold_is_fresh_score_cache_and_fresh_process_not_OS_disk_flush": True,
                   "formal_full_raw_frames": 7338, "original_eligible": 6797, "selected": 1002, "pairs": 11583,
                   "all_six_side_original_decisions_cache_JPEG_K_rawposes_and_current_reference_exact": True,
                   "source_input_identities": len(review["sources_and_inputs"]), "executed_source_snapshots": len(review["snapshots"]),
                   "root_checks": len(review["checks"]), "registered_owned_process_instances_exited": len(review["process_instances"]) + len(instances),
                   "peak_complete_tree_RSS_medians_bytes": review["peak_complete_tree_RSS_medians"],
                   "additional_measured_memory_cost_is_not_a_memory_optimization": True,
                   "original_source_quality_formula_OpenCV_functions_parameters_workers_queue_contracts_unchanged": True,
                   "not_run": "production_integration_new_training_GT_unseen_object_whole_pipeline_new_preview",
                   "final_native_text_publication_and_its_actual_fee_pending": True,
                   "elapsed_current_cost_publication_before_final_write_seconds": elapsed}
        atomic_json(OWN / "ACCEPTANCE.json", payload)
        atomic_json(output, {**payload, "actual_review_instances_exited": instances, "counted": False})
        print({name: payload[name] for name in ("status", "all_measured_shared_fees_through_cost_publication_seconds", "complete_reductions", "accepted_count_contribution")})
        return
    acceptance = load(OWN / "ACCEPTANCE.json")
    cost_instance = load(OWN / "COST_EXECUTION_001.json")
    check["actual_cost_publisher_process_exited"] = gone(cost_instance)
    text_rows, links = [], []
    for name in ("R141_NATIVE_QUALITY_WORKSPACE_TASK.md", "R141_NATIVE_QUALITY_WORKSPACE_RESULTS.md", "CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md"):
        text = (DOC / name).read_text(encoding="utf-8")
        if not text.strip() or "\ufffd" in text:
            raise AssertionError("A native source publication is empty or corrupt")
        text_rows.append({"path": str(DOC / name), "direct_text_characters": len(text), "direct_text_lines": len(text.splitlines()),
                          "screenshot_OCR_document_hash_used": False})
        if name == "R141_NATIVE_QUALITY_WORKSPACE_RESULTS.md":
            check["native_result_scope_cold_denominators_fees_and_unrun_limits"] = all(word in text for word in (
                "7338", "6797", "1002", "11583", "23", "128", "360", "77", "22/50", "新增1", "5%", "未部署", "未见物体", "OS"))
            for target in re.findall(r"\]\((C:/[^)\n]+)\)", text):
                if not Path(target).is_file():
                    raise AssertionError("A result link has no current evidence source")
                links.append(target)
    parent = load(DOC / "CURRENT_STATE.json")
    check["parent22_unique22_receipts_no_active_candidate_or_production_change"] = parent["additional_accepted_count"] == 22 and len(set(
        parent["accepted_mechanisms"])) == len(set(parent["acceptance_receipts"])) == 22 and parent["active_candidates"] == [] and not parent["production_modified"]
    elapsed = time.perf_counter() - started
    shared = acceptance["all_measured_shared_fees_through_cost_publication_seconds"] + elapsed
    formal = load(OWN / "c1/RESULT.json")
    medians = {name: {mode: seconds + shared / 6 for mode, seconds in values.items()} for name, values in formal["medians"].items()}
    reductions = {name: 1 - values["workspace"] / values["native"] for name, values in medians.items()}
    check["final_publication_fee_included_complete_phase5percent_whole_nonincrease"] = reductions["phase_seconds"] >= .05 and reductions["whole_seconds"] >= 0
    if not all(check.values()):
        raise AssertionError(json.dumps(check))
    acceptance.update(all_measured_shared_fees_through_cost_publication_seconds=shared,
                      complete_phase_and_whole_medians_seconds=medians, complete_reductions=reductions,
                      final_native_text_publication_and_its_actual_fee_pending=False,
                      actual_final_native_text_publication_before_write_seconds=elapsed,
                      final_publication=str(output), final_publication_checks=check)
    atomic_json(OWN / "ACCEPTANCE.json", acceptance)
    atomic_json(OWN / "FINAL_COST_001.json", {"status": "all_measured_generic_pilot_full_source_errors_external_review_publication_fees_included",
        "all_measured_shared_fees_seconds": shared, "complete_medians_seconds": medians, "complete_reductions": reductions,
        "last_receipt_serialization_stdout_process_shutdown_not_separately_timed": True, "counted": False})
    atomic_json(output, {"status": "native_text_result_links_parent22_review_and_cost_exit_actual_OS_lock_and_complete_final_cost_verified",
        "checks": check, "root_review_and_cost_instances_exited": instances + [{"pid": cost_instance["pid"], "creation_time": cost_instance["creation_time"]}],
        "reviewed_native_text": text_rows, "actual_result_links": links, "review_kind": "root_self_review_no_subagents",
        "all_measured_shared_fees_seconds": shared, "complete_reductions": reductions, "parent_goal_active_not_complete": True,
        "current_publication_before_write_seconds": elapsed, "counted": False})
    print({"status": "final_native_text_cost_parent22_and_resource_publication_verified", "checks": len(check), "shared_fees_seconds": shared,
           "complete_reductions": reductions, "accepted_count_contribution": 1})


if __name__ == "__main__":
    main()
