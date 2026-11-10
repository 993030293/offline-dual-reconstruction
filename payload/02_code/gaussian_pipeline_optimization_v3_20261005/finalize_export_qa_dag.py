"""Publish one scoped acceptance only after the real audit process has exited."""

import time
STARTED = time.perf_counter()

import datetime
import json
from pathlib import Path

import psutil

from profile_export_qa_dag import BASE, ROOT
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    if (BASE / "ACCEPTANCE.json").exists():
        raise FileExistsError("A scoped acceptance cannot overwrite a prior decision")
    review = read(BASE / "COST_REVIEW_PENDING_001.json")
    guard = read(BASE / "CLOSE_GUARD_001.json")
    identity = review["closer_identity"]
    try:
        exited = psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        exited = True
    if not exited or guard["status"] != "completed" or guard["returncode"] != 0 or not all(review["checks"].values()):
        raise ValueError("The actual root closure or its protections did not complete")
    # Charge the complete audited process envelope and the measured publisher work.
    fees = review["all_known_measured_fees_before_this_closure_and_publication_seconds"] + guard["elapsed_seconds"]
    publisher_seconds = time.perf_counter() - STARTED
    fees += publisher_seconds
    medians = {kind: {mode: value + fees / 6 for mode, value in rows.items()} for kind, rows in review["raw_medians"].items()}
    reduction = {kind: 1 - rows["dag"] / rows["serial"] for kind, rows in medians.items()}
    checks = {"complete_phase_all_measured_fees_speed_at_least5percent": reduction["phase"] >= .05,
        "complete_whole_all_measured_fees_no_regression": reduction["whole"] >= 0,
        "original_each_pair_whole_limit": review["checks"]["original_whole_each_pair_and_median_guards"],
        "all_scientific_inputs_outputs_and_lifecycle_protections": all(review["checks"].values()),
        "actual_closer_PID_creation_time_exit_and_bound_guard": exited and guard["elapsed_seconds"] <= guard["maximum_seconds"] and guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"],
    }
    cost = {"status": "complete_measured_cost_gates_passed" if all(checks.values()) else "failed_complete_cost_no_acceptance", "checks": checks,
        "raw_medians": review["raw_medians"], "all_known_measured_nonoverlapping_fees_seconds": fees,
        "audit_complete_process_envelope_seconds": guard["elapsed_seconds"], "publisher_measured_before_receipt_seconds": publisher_seconds,
        "common_fee_divisor": 6, "complete_medians_including_fees": medians, "reductions_including_fees": reduction,
        "median_sampled_process_tree_RSS_bytes": review["median_sampled_process_tree_RSS_bytes"],
        "subsequent_receipt_text_publication_and_post_exit_check_outside_measured_fee_boundary": True}
    atomic_json(BASE / "FINAL_COST_001.json", cost)
    if not all(checks.values()):
        raise AssertionError("The original full-stage cost gates failed")
    atomic_json(BASE / "ACCEPTANCE.json", {"status": "accepted_scoped_engineering_mechanism", "id": "R140_IMMUTABLE_MAP_QA_ARTIFACT_DAG",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "new_optimization_count": 1,
        "checks": checks, "raw_medians": review["raw_medians"], "complete_cost": cost,
        "actual_complete_dag_composition_with_accepted_R112_R132_verified": True,
        "full_current1021_1016_75767_959120_and1050_outputs_exact": True,
        "generalization": "nine synthetic non-box geometry-scale-calibration cases and existing Oxford360 native artifact component; not unseen-object whole-pipeline validation",
        "Oxford_current_PINHOLE_training_admission": "not_passed_original_native_zero_distortion_OPENCV_remains",
        "review": "root_single_chat_self_review_no_subagents_or_independent_review",
        "production_deployed": False, "new_training_GT_new_object_whole_pipeline_preview": "not_run",
        "startup_instruction_signal_power_loss_and_arbitrary_ACL_recovery": "not_exhaustively_tested",
        "publisher_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time(), "post_exit_check_required": True}})
    print(json.dumps({"status": "accepted_scoped_engineering_mechanism", "new_optimization_count": 1,
        "reductions_including_fees": reduction, "complete_medians_including_fees": medians, "fee_seconds": fees}), flush=True)


if __name__ == "__main__":
    main()
