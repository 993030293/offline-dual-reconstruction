"""Prepare V5 costs, then qualify acceptance after the publisher actually exits."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = BASE / "pf_v5_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
RESERVED_FINAL_ENVELOPE_SECONDS = 10.
MECHANISM = "R113_V5_EXACT_LIVE_POINT_REJECTION_PREDICATE"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def released(label):
    lease, guard = read(OWN / ("LEASE_" + label + ".json")), read(OWN / ("GUARD_" + label + ".json"))
    if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] or guard["returncode"]:
        raise ValueError("The complete " + label + " envelope has not exited and released")
    identities = guard["owned_process_identities"] + [{"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]} for role in ("guard", "child")]
    if not all(closed(identity) for identity in identities):
        raise ValueError("An actual " + label + " instance is still running")
    return lease, identities


def costs(original, fee):
    rows = {label: {field: [pair["rows"][label][field] + fee/6 for pair in original["pairs"]]
                   for field in ("phase_seconds", "whole_seconds")} for label in ("native", "columns")}
    medians = {label: {field: statistics.median(values) for field, values in fields.items()} for label, fields in rows.items()}
    ratios = [right/left for left, right in zip(rows["native"]["whole_seconds"], rows["columns"]["whole_seconds"])]
    reduction = 1 - medians["columns"]["phase_seconds"]/medians["native"]["phase_seconds"]
    return {"complete_adjusted_medians": medians, "complete_phase_reduction_lower_bound": reduction,
            "complete_whole_median_ratio": medians["columns"]["whole_seconds"]/medians["native"]["whole_seconds"],
            "each_whole_ratio": ratios, "known_fee_plus_reserved_final_envelope_seconds": fee,
            "six_original_complete_sides_same_fee_not_formal_window_double_charged": True}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "finalize"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    (output / "EXECUTED_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    review = read(OWN / "r0/RESULT.json")
    current = read(DOC / "CURRENT_STATE.json")
    if len(review["checks"]) != 33 or not all(review["checks"].values()) or current["additional_accepted_count"] != 27 or len(set(current["accepted_mechanisms"])) != 27 or len(set(current["acceptance_receipts"])) != 27 or any(name.startswith("R113") for name in current["accepted_mechanisms"]):
        raise ValueError("The current semantic review or unique parent27 frontier changed")
    if current["V5_minimum_complete_stage_net_reduction"] != .03 or current["V5_maximum_whole_median_ratio"] != 1.01 or current["V5_maximum_each_whole_ratio"] != 1.1:
        raise ValueError("The preregistered uniform V5 gate changed")
    if data_identity(Path(__file__).with_name("point_filter_predicate.py")) != review["original_candidate_source"] or data_identity(Path(__file__).with_name("review_point_filter_v5.py")) != review["V5_review_source"]:
        raise ValueError("The actually reviewed candidate or checker source changed")
    root_lease, process_rows = released("R0")
    known_before_publisher = review["known_fee_before_this_review"] + root_lease["child_elapsed_seconds"]
    original = read(BASE / "point_filter_predicate_001/campaign_001/RESULT.json")
    if args.mode == "prepare":
        result = {"status": "prepared_V5_qualification_actual_publisher_exit_pending", "known_complete_fees_before_publisher": known_before_publisher,
            "source": data_identity(Path(__file__)), "reserved_final_envelope_seconds": RESERVED_FINAL_ENVELOPE_SECONDS,
            "known_costs": costs(original, known_before_publisher), "original33_semantic_checks_passed": True,
            "new_optimization_count": 0, "parent_count": 27, "all_prior_review_instances_exited": process_rows,
            "node_seconds": time.perf_counter()-started}
        atomic_json(output / "RESULT.json", result)
        print({"status": result["status"], "known_complete_fees_before_publisher": known_before_publisher}, flush=True)
        return
    prepared = read(OWN / "p0/RESULT.json")
    if prepared["status"] != "prepared_V5_qualification_actual_publisher_exit_pending" or prepared["source"] != data_identity(Path(__file__)) or prepared["known_complete_fees_before_publisher"] != known_before_publisher:
        raise ValueError("The prepared publisher source or cost changed")
    publisher_lease, publisher_rows = released("P0")
    process_rows.extend(publisher_rows)
    fee = known_before_publisher + publisher_lease["child_elapsed_seconds"]
    conservative = costs(original, fee + RESERVED_FINAL_ENVELOPE_SECONDS)
    gate = {"complete_stage_lower_bound_at_least3percent": conservative["complete_phase_reduction_lower_bound"] >= .03,
            "whole_median_at_most1point01": conservative["complete_whole_median_ratio"] <= 1.01,
            "each_whole_at_most1point1": all(ratio <= 1.1 for ratio in conservative["each_whole_ratio"]),
            "publisher_actually_exited_and_source_semantics33_unchanged": True}
    if not all(gate.values()):
        atomic_json(output / "FAILURE.json", {"gate": gate, "costs": conservative, "new_optimization_count": 0})
        raise ValueError("The complete conservative V5 cost guard failed")
    target = OWN / "ACCEPTANCE_V5_001.json"
    if target.exists():
        raise FileExistsError("The V5 qualification receipt must not overwrite an existing result")
    result = {"status": "qualified_V5_finalizer_exit_and_remaining_execution_reserve_validation_pending", "candidate": "R113_V5",
        "mechanism": MECHANISM, "new_optimization_count": 0, "parent_before": 27, "parent_after_if_verified": 28,
        "review": str(OWN / "r0/RESULT.json"), "review_kind": "separate_root_self_review_not_independent_agent",
        "review_checks": 33, "formal_three_pairs_reused_not_new": True, "fresh_native_delete_cases": 3,
        "fresh_external_initial_denominator": [360, 360, 133020, 848219], "old_strict_and_V4_failures_preserved": True,
        "original_all_quality_input_frames_native_fields_order_recovery_kept": True, "gate": gate, "costs": conservative,
        "known_all_metered_fees_before_finalizer_seconds": fee, "reserved_remaining_measured_execution_seconds": RESERVED_FINAL_ENVELOPE_SECONDS,
        "reserve_is_conservative_cost_upper_bound_not_measured_elapsed": True, "actual_publisher_exit_lease": str(OWN / "LEASE_P0.json"),
        "publisher_source": data_identity(Path(__file__)), "actual_prior_closed_process_instances": process_rows,
        "production_deployed": False, "latest_R132_R140_combination_new_training_GT_unseen_full_chain_verified": False,
        "unmetered_text_research_historical91check_time": "unknown_not_assumed_zero_not_total_research_wall_or_deployment_ROI",
        "goal_active": True, "node_seconds": time.perf_counter()-started}
    atomic_json(target, result)
    atomic_json(output / "RESULT.json", result)
    print({"status": result["status"], "conservative_stage_reduction": conservative["complete_phase_reduction_lower_bound"],
           "whole_ratio": conservative["complete_whole_median_ratio"], "new_optimization_count": 0}, flush=True)


if __name__ == "__main__":
    main()
