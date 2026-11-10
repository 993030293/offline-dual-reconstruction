"""Reuse unchanged semantic checks with V5 paths and verify every current cost."""

import time

STARTED = time.perf_counter()

import argparse
import ast
import difflib
import inspect
from pathlib import Path
import statistics
import sys

import psutil

import review_point_filter_v4 as old
from point_filter_oracles import compare_outputs, exact_values
from point_filter_predicate import ROOT, native, ORIGINAL_FILTER
from runtime import atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v5_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OWN)
    if output.exists():
        raise FileExistsError("The V5 separate review needs a fresh directory")
    source = inspect.getsource(old.main)
    changes = [("output.relative_to(OWN)", "output.relative_to(V5_OWN)", 1),
               ("v4_delete_001", "v5_delete_001", source.count("v4_delete_001")),
               ('OWN / "e1/', 'OWN / "v5_e0/', 2),
               ("== 24 and not any", "== 27 and not any", 1),
               ("unique_R113_not_among_existing24_or_old35", "unique_R113_not_among_existing27_or_old35", 1)]
    adapted = source
    for before, after, count in changes:
        if not count or adapted.count(before) != count:
            raise ValueError("The frozen reused semantic review adapter anchor changed: " + before)
        adapted = adapted.replace(before, after)
    namespace = dict(vars(old))
    namespace.update(V5_OWN=OWN, STARTED=STARTED)
    exec(compile(adapted, "<point_filter_V5_owned_paths_and_current_count>", "exec"), namespace)
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--output", str(output)]
        namespace["main"]()
    finally:
        sys.argv = previous_argv
        if output.is_dir():
            (output / "EXECUTED_V5_REVIEW_SOURCE.py").write_bytes(Path(__file__).read_bytes())
            (output / "EXECUTED_ADAPTED_SEMANTIC_MAIN.py").write_text(adapted, encoding="utf-8")
            (output / "SEMANTIC_REVIEW_ADAPTER.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), adapted.splitlines(True),
                fromfile="original_V4_review_main", tofile="V5_owned_outputs_and_count")), encoding="utf-8")
    result = old.read(output / "RESULT.json")
    if len(result["checks"]) != 25 or not all(result["checks"].values()):
        raise ValueError("An unchanged original semantic review check failed")
    checks = dict(result["checks"])
    external_root = old.OWN / "v5_e0"
    external = old.read(external_root / "RESULT.json")
    comparisons = {step: compare_outputs(external_root / "native" / step, external_root / "columns" / step)
                   for step in external["comparisons"]}
    exact_values(comparisons, external["comparisons"])
    exact_values(old.read(external_root / "native/LIVE_STATES.json"), old.read(external_root / "columns/LIVE_STATES.json"))
    checks["fresh_public360_all_six_five_BIN_full_bytes_and_live_fields_recompared"] = len(comparisons) == 6 and all(row["complete_byte_pairs"] == 5 for row in comparisons.values())
    process_rows = list(result["closed_process_records"])
    fees = []
    for parent, labels in ((old.OLD, ("generic_001", "generic_002")), (old.OWN, ("g0", "e0", "e1", "r0"))):
        for label in labels:
            path = parent / (label + ("_LEASE.json" if parent == old.OLD else "_lease.json"))
            lease = old.read(path)
            if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
                raise ValueError("An old complete fee has no released process envelope")
            fees.append({"label": "old_" + label, "receipt": str(path), "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
            process_rows.extend({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]} for role in ("guard", "child"))
    history = old.read(OWN / "h8/RESULT.json")
    fees.extend([{"label": "old_V4_publisher_full_not_internal_plus_full", "seconds": history["actual_publisher_complete_command_seconds"]},
                 {"label": "old_post_exit_all_two_command_walls_sum", "seconds": history["old_post_exit_complete_command_seconds"]}])
    checks["old_actual_publisher_and_post_exit_costs_recovered_no_tail_assumed_zero"] = history["status"] == "original_publisher_actual_closed_command_timing_recovered" and history["result"]["exit_codes"] == [0] and history["old_post_exit_result"]["exit_codes"] == [0, 0] and history["actual_publisher_complete_command_seconds"] >= history["original_publisher_internal_seconds_already_in_V4"]
    process_rows.extend(old.read(old.OWN / "V4_NEGATIVE_CLOSURE_001.json")["all_saved_owned_process_instances_exited"])
    closure = old.read(old.OWN / "V4_NEGATIVE_CLOSURE_001.json")
    process_rows.append({"pid": closure["publisher_pid"], "creation_time": closure["publisher_creation_time"]})
    codes = {"G0": 0, "E0": 0, **{"H" + str(number): (0 if number in (3, 8) else 1) for number in range(9)}}
    for label, code in codes.items():
        path = OWN / ("LEASE_" + label + ".json")
        lease, guard = old.read(path), old.read(OWN / ("GUARD_" + label + ".json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != code or guard["returncode"] != code:
            raise ValueError("A fresh, first or failed V5 task has no actual drain and release: " + label)
        if guard["elapsed_seconds"] > guard["maximum_seconds"] or guard["peak_tree_RSS_bytes"] > guard["maximum_RSS_bytes"]:
            raise ValueError("A registered V5 time or sampled RSS budget was exceeded")
        fees.append({"label": "V5_" + label, "receipt": str(path), "seconds": lease["child_elapsed_seconds"], "returncode": code})
        process_rows.extend(guard["owned_process_identities"])
        process_rows.extend({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]} for role in ("guard", "child"))
    checks["all_eleven_current_actual_semantic_and_historical_parser_tasks_budgets_expected_errors_preserved"] = len(codes) == 11
    checks["original_parser_failures_retained_not_used_as_algorithm_results"] = all((OWN / ("h" + str(number)) / "EXECUTED_SOURCE.py").is_file() for number in range(1, 9)) and (OWN / "h0/FAILED_SOURCE.py").is_file()
    unique = {(row["pid"], row["creation_time"]): row for row in process_rows}
    checks["old_and_fresh_actual_saved_process_instances_exited"] = all(old.closed(row) for row in unique.values())
    current = old.read(DOC / "CURRENT_STATE.json")
    checks["all27_mechanisms_and_receipts_unique_R113_not_recounted"] = current["additional_accepted_count"] == len(set(current["accepted_mechanisms"])) == len(set(current["acceptance_receipts"])) == 27 and not any(name.startswith("R113") for name in current["accepted_mechanisms"])
    checks["fixed_V5_3percent_whole_limits_and_old_V4_negative_unchanged"] = current["V5_minimum_complete_stage_net_reduction"] == .03 and current["V5_maximum_whole_median_ratio"] == 1.01 and current["V5_maximum_each_whole_ratio"] == 1.1 and closure["new_optimization_count"] == 0 and closure["complete_phase_reduction_upper_bound"] < .05
    checks["actual_original_point_filter_binding_restored_after_review"] = native.filter_points is ORIGINAL_FILTER
    if not all(checks.values()):
        atomic_json(output / "V5_FAILURE.json", {"checks": checks, "new_optimization_count": 0})
        raise ValueError("A current V5 semantic, source, cost or resource condition failed")
    original = old.read(old.OLD / "campaign_001/RESULT.json")
    known_before_review = sum(row["seconds"] for row in fees)
    internal = time.perf_counter() - STARTED
    provisional = {label: {field: original["medians"][label][field] + (known_before_review + internal)/6 for field in ("phase_seconds", "whole_seconds")} for label in ("native", "columns")}
    result.update(status="V5_current_semantics_full_outputs_costs_root_self_review_passed_actual_review_exit_and_publication_pending", checks=checks,
        V5_review_kind="separate_root_self_review_not_independent_agent", formal_three_pairs_reused_not_new=True,
        original_source_old_strict_V4_failures_preserved=True, fresh_public360_comparisons=comparisons,
        all_actual_first_failed_generic_external_review_and_publication_fees=fees, known_fee_before_this_review=known_before_review,
        V5_internal_review_seconds=internal, provisional_complete_medians=provisional,
        actual_saved_closed_process_instances=list(unique.values()), V5_review_source=data_identity(Path(__file__)),
        original_candidate_source=data_identity(Path(__file__).with_name("point_filter_predicate.py")),
        V5_source_adapter_scope="output_directory_current27_and_new_original_checker_artifacts_only_all25_protected_semantic_conditions_kept",
        new_optimization_count=0, parent_goal_active=True, node_seconds=time.perf_counter()-STARTED)
    atomic_json(output / "RESULT.json", result)
    print({"status": result["status"], "checks": len(checks), "known_fee_before_current_review": known_before_review,
           "provisional_phase_reduction": 1-provisional["columns"]["phase_seconds"]/provisional["native"]["phase_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
