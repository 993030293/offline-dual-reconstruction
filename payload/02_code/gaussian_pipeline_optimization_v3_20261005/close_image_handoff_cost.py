"""Close actual execution fees without recharging the six formal windows."""

import json
from pathlib import Path
import statistics
import sys
import time

import psutil

from run_open_image_handoff_campaign import OUT
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    review = read(OUT / "r1/RESULT.json")
    formal = read(OUT / "c0/RESULT.json")
    if not all(review["checks"].values()) or len(review["checks"]) != 21:
        raise ValueError("The separately executed root review did not pass")
    fees = {"registered_inherited_preparation_inverse_and_main_diagnostic": 80.0677363}
    leases = [OUT.parent / "shf001_preflight_lease.json", *[OUT / (name + "_lease.json") for name in ("g0", "v1", "v2", "v3", "g1", "r0", "r1")]]
    owners = []
    for path in leases:
        row = read(path)
        if not row["lease_released_utc"] or row["returncode"] not in (0, 1):
            raise ValueError("A required validation lease did not release its actual owner")
        fees[path.stem] = row["child_elapsed_seconds"]
        owners.extend({"pid": row[key], "creation_time": row[key.replace("pid", "creation_time")]} for key in ("guard_pid", "child_pid"))
    for label, selected in (("first", formal["pairs"][:1]), ("rest", formal["pairs"][1:])):
        lease = read(OUT / ("c0_" + label + "_lease.json"))
        occupied = sum(sum(row["actual_spawn_to_exit_seconds"] for row in pair["rows"].values()) + pair["actual_compare_seconds_charged_each_side"] for pair in selected)
        extra = lease["child_elapsed_seconds"] - occupied
        if extra < -1e-3:
            raise ValueError("The original formal windows exceeded their actual campaign envelope")
        fees["actual_formal_" + label + "_envelope_overhead_only"] = max(0., extra)
    terminal = {"7cab8e_syntax": .1940775, "0e28b4_native_preflight_text": .2094558,
        "396c27_owned_entry_read": .3226815, "ad11c7_first_actual_output_read": .2124059,
        "cb1062_actual_three_pair_read": .1713005, "660356_failed_validation_sources_copy": .1610641,
        "05bca4_actual_fault_entry_import_regression": .7162434,
        "7b1ba9_prior_family_and_native_failure_read": .3473701, "184ed9_failed_injection_native_log": .2541516,
        "7e046b_actual_filename_and_fault_source_regression": .2873441, "bac608_failed_V2_sources_copy": .1551661,
        "5b6c79_actual_corrected_failure_and_recovery_read": .3018495, "4dbabc_actual_inherited_fees_and_owner_shapes": .1438156,
        "38083e_failed_root_query_log": .1792663, "ecd7e5_failed_root_source_copy": .2759738,
        "a5fa7e_actual_passed_root_checks_read": .2477607}
    fees["actual_terminal_nonformal_quality_regression_and_evidence_commands"] = sum(terminal.values())
    fees["registered_last_execution_reserve"] = 10.
    known = sum(fees.values())
    control, handoff = (formal["raw_whole_medians"][mode] for mode in ("control", "handoff"))
    adjusted = {"control": control + known / 6, "handoff": handoff + known / 6}
    reductions = review["net_complete_source_work_reduction_under_declared_six_side_model"]
    native_results = {mode: [read(OUT / "c0" / ("t" + str(index)) / mode / "RESULT.json") for index in range(3)] for mode in ("control", "handoff")}
    peaks = {mode: statistics.median(row["full_process_tree_peak_RSS_bytes"] for row in native_results[mode]) for mode in native_results}
    checks = {"all21_separate_root_checks_passed": all(review["checks"].values()),
        "all_extra_validation_owners_actual_exited": all(windows_closed(owner) for owner in owners) and windows_closed(review["own_process"]),
        "all_original_speed_and_whole_gates_unchanged": handoff / control <= 1.01 and adjusted["handoff"] / adjusted["control"] <= 1.01 and max(formal["whole_ratios"]) <= 1.1,
        "registered_net_file_work_above10percent": reductions >= .10,
        "formal_source_versions_and_actual_root_review_source_exact": formal["sources"] == [data_identity(row["path"]) for row in formal["sources"]] and review["sources"] == [data_identity(row["path"]) for row in review["sources"]]}
    result = {"status": "scoped_work_checks_passed_actual_closer_exit_and_text_publication_pending0" if all(checks.values()) else "failed_verification",
        "checks": checks, "execution_fees_seconds": fees, "terminal_tool_receipts": terminal,
        "known_fee_plus_registered10_reserve_seconds": known,
        "raw_whole_medians": formal["raw_whole_medians"], "raw_whole_reduction": formal["raw_whole_reduction"],
        "known_fee_adjusted_whole_medians": adjusted, "known_fee_adjusted_whole_reduction": 1 - adjusted["handoff"] / adjusted["control"],
        "V5_speed3percent_gate_passed": 1 - adjusted["handoff"] / adjusted["control"] >= .03,
        "raw_source_native_read_work_control": 4064, "raw_source_native_read_work_handoff": 3048,
        "raw_source_native_read_work_reduction": .25, "measured_extra_source_read_work": review["measured_extra_source_read_work"],
        "conservative_inherited_work_budget_not_direct_historical_counter": review["conservative_inherited_source_work_budget_not_direct_counter_measurement"],
        "net_source_read_work_under_registered_six_side_model": reductions,
        "median_full_process_tree_RSS_bytes": peaks, "RSS_increase_bytes": peaks["handoff"] - peaks["control"],
        "new_optimization_count": 0, "parent_count": 29, "own_process": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "internal_before_result_seconds": time.perf_counter() - started, "source": data_identity(Path(__file__)),
        "unknown_text_research_and_development_time_not_assumed_zero": True,
        "training_GT_FPS_energy_or_unknown_object_pipeline_improvement": "not_claimed", "production_default_deployed": False}
    atomic_json(OUT / "COST_CLOSURE_PENDING_EXIT_001.json", result)
    print({"checks": checks, "known_fee_plus_reserve": known, "net_source_read_work": reductions,
           "raw_whole": formal["raw_whole_medians"], "adjusted_whole": adjusted, "RSS_increase_bytes": result["RSS_increase_bytes"], "new": 0})
    if not all(checks.values()):
        raise AssertionError("A final declared cost or scoped whole gate failed")


if __name__ == "__main__":
    main()
