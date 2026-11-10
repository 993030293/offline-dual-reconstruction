"""Audit reused full filter exports and all measured V4 supplement fees."""

import time

STARTED = time.perf_counter()

import argparse
import ast
import difflib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import psutil
import pycolmap

from point_filter_oracles import compare_outputs, exact_values
from point_filter_predicate import ROOT, V1, MAXIMUM_TEMPORARY_BYTES, ORIGINAL_FILTER, native
from profile_point_filter_predicate import input_paths
from check_point_filter_v4_external import native_count
from runtime import atomic_json, data_identity


CODE = Path(__file__).parent
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/point_filter_predicate_001"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unchanged(contract):
    expected = contract["sources_and_inputs"]
    if [data_identity(Path(row["path"])) for row in expected] != expected:
        raise AssertionError("A frozen complete source or input identity changed")
    return len(expected)


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    checks, process_rows, leases, comparisons = {}, [], [], []
    sources = [Path(__file__), CODE / "check_point_filter_v4_external.py", CODE / "point_filter_predicate.py",
               CODE / "point_filter_oracles.py", CODE / "check_point_filter_predicate.py",
               CODE / "check_point_filter_delete_prefix.py", CODE / "profile_point_filter_predicate.py",
               CODE / "run_with_resource_lease.py", V1 / "run_bounded_task.py", Path(native.__file__)]
    identities = [data_identity(path) for path in sources]
    copies = output / "executed_sources"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (str(index) + "_" + path.name))
        ast.parse(path.read_text(encoding="utf-8-sig"))
    stage = "original_source_and_input"
    try:
        original = read(OLD / "campaign_001/RESULT.json")
        contract = read(OLD / "campaign_001/CONTRACT.json")
        checks["original_contract1041_sources_and1028_full_inputs_unchanged"] = unchanged(contract) == 1041 and len(contract["complete_input_membership"]) == 1028
        checks["current_complete_input_membership_exact"] = [str(path.resolve()) for path in input_paths()] == contract["complete_input_membership"]
        checks["same_local_runtime_and256MiB_admission"] = pycolmap.__version__ == contract["pycolmap"] == "4.1.1" and np.__version__ == contract["numpy"] and MAXIMUM_TEMPORARY_BYTES == 256 * 1024**2
        snapshot_count = 0
        for index, row in enumerate(contract["sources_and_inputs"][:13]):
            path = Path(row["path"])
            if path.suffix in (".py", ".json"):
                if (OLD / "campaign_001/source_snapshot" / f"{index:02d}_{path.name}").read_bytes() != path.read_bytes():
                    raise AssertionError("The original full export executed source snapshot differs")
                snapshot_count += 1
        checks["eleven_complete_original_text_source_copies_byte_exact"] = snapshot_count == 11
        checks["original_strict_cost_failure_and_no_external_claim_preserved"] = original["status"] == "complete_filter_cost_guard_failed" and not original["checks"]["phase_reduction_at_least10pct"] and not original["checks"]["whole_median_nonregressed"] and not original["external_component_run"]
        checks["same_three_original_alternating_pairs_reused"] = len(original["pairs"]) == 3 and [list(pair["rows"]) for pair in original["pairs"]] == [["native", "columns"], ["columns", "native"], ["native", "columns"]]
        stage = "full_original_output_bytes_and_cost_recalculation"
        for pair in original["pairs"]:
            trial = pair["trial"]
            comparison = compare_outputs(Path(pair["rows"]["native"]["folder"]), Path(pair["rows"]["columns"]["folder"]))
            exact_values(comparison, pair["verification"])
            if comparison["complete_byte_pairs"] != 1046 or comparison["complete_JSON_pairs"] != 4 or comparison["complete_compared_bytes"] != 746975609:
                raise AssertionError("An original full1050 output comparison changed scope")
            comparisons.append(comparison)
            for label, row in pair["rows"].items():
                leaf = read(row["source_bound_leaf_result"])
                trace = leaf["trace"]
                if not trace["original_binding_restored"] or len(trace["calls"]) != 2 or not leaf["original_sparse_PLY_writer_and_grid_metric_used"]:
                    raise AssertionError("An original binding, call count or unchanged co-function differs")
                if any(call["status"] != "original_filter_completed" or call["route"] != ("columns" if label == "columns" else "original") for call in trace["calls"]):
                    raise AssertionError("The actual full original leaf changed route or errors")
                phase = leaf["phase_seconds"] + row["startup_wait_receipt_and_child_close_seconds"] + original["first_prepare_seconds"] / 6
                whole = row["actual_child_envelope_seconds"] + original["first_prepare_seconds"] / 6
                phase += pair["actual_comparison_seconds"]
                whole += pair["actual_comparison_seconds"]
                phase += original["complete_unallocated_publication_and_close_seconds"] / 6
                whole += original["complete_unallocated_publication_and_close_seconds"] / 6
                exact_values(phase, row["phase_seconds"])
                exact_values(whole, row["whole_seconds"])
        checks["all_three_full1050_bytes_and_scientific_JSON_recompared_exact"] = len(comparisons) == 3
        checks["six_original_complete_side_costs_recomputed_bit_exact"] = True
        checks["six_real_original_filter_bindings_and_R111_R112_disabled"] = not contract["R111_and_R112_candidate_functions_enabled"]
        for label in ("native", "columns"):
            for field in ("phase_seconds", "whole_seconds"):
                exact_values(statistics.median(pair["rows"][label][field] for pair in original["pairs"]), original["medians"][label][field])
        checks["original_phase_and_whole_medians_recomputed_bit_exact"] = True
        stage = "generic_and_actual_forced_delete"
        generic = read(OLD / "generic_002/RESULT.json")
        checks["source_bound_original57_generic_scope_preserved"] = generic["status"] == "generic_native_point_filter_oracle_passed" and generic["cases"] == len(generic["rows"]) == 57 and unchanged(read(OLD / "generic_002/CONTRACT.json")) > 0
        generic_cases = {row["case"]: row for row in generic["rows"]}
        missing = generic_cases["missing_camera_id"]["outcomes"][0]
        checks["actual_missing_camera_error_and_context_exception_scope_preserved"] = not missing["ok"] and missing["type"] == "ValueError" and missing["message"] == "[image.h:206] 'camera_ptr_' Must be non NULL" and "context_restore_RuntimeError" in generic_cases and "context_restore_KeyboardInterrupt" in generic_cases
        forced = read(OLD / "v4_delete_001/RESULT.json")
        checks["three_actual_forced_delete_return_error_prefix_and_binding_guards"] = forced["status"] == "forced_native_delete_prefix_oracle_passed" and forced["cases"] == 3 and forced["candidate_semantic_differences"] == 0 and unchanged(read(OLD / "v4_delete_001/CONTRACT.json")) > 0
        for row in forced["rows"]:
            records = row["records"]
            exact_values(records[0]["calls"], records[1]["calls"])
            if any(record["point_ids_after"] or not record["trace"]["original_binding_restored"] or record["trace"]["calls"][0]["removed_count"] != 160 for record in records):
                raise AssertionError("A required forced track did not actually enter full sequential deletion")
            name = row["case"]
            exact_values(read(OLD / "v4_delete_001" / name / "native/STATE_BITS.json"), read(OLD / "v4_delete_001" / name / "columns/STATE_BITS.json"))
        checks["forced_all160_points_eachcase_actual_order_and_complete_state_bits_rechecked"] = True
        stage = "full_external_native_filter"
        external = read(OWN / "e1/RESULT.json")
        checks["external_fixed360_133020_848219_complete_six_native_states_exact"] = external["status"] == "complete_external_native_filter_component_exact" and external["full_native_fields_and_associations_byte_exact"] and external["all_live_order_cache_pose_rig_frame_and_queries_exact"] and len(external["comparisons"]) == 6 and unchanged(read(OWN / "e1/CONTRACT.json")) > 0
        paired = external["paired_rows"]
        exact_values(paired[0]["scientific"], paired[1]["scientific"])
        checks["external_two_pass_original_admission_and7_actual_frames_plus1_stress"] = external["two_filter_passes_and_original_camera_admission"] and external["original_admission_frames_deregistered"] == 7 and external["additional_stress_frame_deregistered"] == 1 and all(row["trace"]["original_binding_restored"] and len(row["trace"]["calls"]) == 3 for row in paired)
        checks["public_scale_not_claimed_metric_training_or_blind_GT"] = external["existing_Oxford_development_not_metric_or_training_admission_or_blind_GT"] and paired[0]["scientific"]["camera_radius_in_unverified_model_units"] > 20
        checks["actual_new_validator_property_method_regression_guard"] = native_count(3) == native_count(lambda: 3) == 3 and read(OWN / "e0/FAILURE.json")["error_type"] == "TypeError" and read(OWN / "e0/FAILURE.json")["stage"] == "before"
        old_checker = OWN / "e0/executed_sources/0_check_point_filter_v4_external.py"
        diff = "".join(difflib.unified_diff(old_checker.read_text().splitlines(True), (CODE / "check_point_filter_v4_external.py").read_text().splitlines(True), fromfile="e0_executed_validator", tofile="e1_executed_validator"))
        (output / "VALIDATOR_PROPERTY_FIX.diff").write_text(diff, encoding="utf-8")
        stage = "actual_closed_resources_and_cost"
        for folder, names in ((OLD, ("generic_001_LEASE.json", "generic_002_LEASE.json", "campaign_001_LEASE.json")),
                              (OWN, ("g0_lease.json", "e0_lease.json", "e1_lease.json"))):
            for name in names:
                lease = read(folder / name)
                if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
                    raise AssertionError("A required actual lease has not closed")
                for role in ("guard", "child"):
                    process_rows.append({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]})
                leases.append({"path": str(folder / name), "returncode": lease["returncode"], "seconds": lease["child_elapsed_seconds"]})
        for name in ("g0", "e0", "e1"):
            guard = read(OWN / (name + "_guard.json"))
            process_rows.extend(guard.get("owned_process_identities", []))
            if guard["peak_tree_RSS_bytes"] > guard["maximum_RSS_bytes"] or guard["maximum_RSS_bytes"] != 8 * 1024**3 or guard["elapsed_seconds"] > guard["maximum_seconds"]:
                raise AssertionError("A new component's actual resource guard exceeded its bound")
        unique = {(row["pid"], row["creation_time"]): row for row in process_rows}
        checks["all_old_and_new_saved_actual_process_instances_closed"] = all(closed(row) for row in unique.values())
        checks["six_original_and_supplement_leases_released_expected_exitcodes"] = [row["returncode"] for row in leases] == [1, 0, 0, 0, 1, 0]
        old_lease = read(OLD / "campaign_001_LEASE.json")
        exact_values(old_lease["child_elapsed_seconds"], original["actual_campaign_seconds_from_released_lease"])
        components = original["first_prepare_seconds"] + sum(pair["actual_comparison_seconds"] + sum(row["actual_child_envelope_seconds"] for row in pair["rows"].values()) for pair in original["pairs"])
        exact_values(old_lease["child_elapsed_seconds"] - components, original["complete_unallocated_publication_and_close_seconds"])
        checks["original404second_formal_window_not_double_charged_and_close_already_present"] = True
        checks["candidate_original_binding_now_restored"] = native.filter_points is ORIGINAL_FILTER
        state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
        checks["unique_R113_not_among_existing24_or_old35"] = state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(state["acceptance_receipts"]) == 24 and not any(name.startswith("R113") for name in state["accepted_mechanisms"])
        checks["current_executed_review_sources_unchanged"] = [data_identity(path) for path in sources] == identities
        output_bytes = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) + sum(path.stat().st_size for path in (OLD / "v4_delete_001").rglob("*") if path.is_file())
        checks["new_output_and_disk_budgets"] = output_bytes <= 30 * 1024**3 and shutil.disk_usage(output).free >= 20 * 1024**3
        fee_before_review = sum(row["seconds"] for row in leases if Path(row["path"]).name != "campaign_001_LEASE.json")
        elapsed = time.perf_counter() - STARTED
        fee = fee_before_review + elapsed
        phase = {label: original["medians"][label]["phase_seconds"] + fee / 6 for label in ("native", "columns")}
        whole = {label: original["medians"][label]["whole_seconds"] + fee / 6 for label in ("native", "columns")}
        reduction = 1 - phase["columns"] / phase["native"]
        costs = {"preliminary_complete_phase_medians_seconds": phase, "preliminary_complete_whole_medians_seconds": whole,
                 "preliminary_phase_reduction": reduction, "preliminary_whole_ratio": whole["columns"] / whole["native"],
                 "fee_before_review_seconds": fee_before_review, "review_internal_seconds": elapsed,
                 "preliminary_total_additional_known_fee_seconds": fee,
                 "metered_execution_only_not_total_research_wall": True,
                 "old_text_source_read_review_and_other_unmetered_overhead": "unquantified_not_assumed_zero",
                 "old_formal_first_prepare_full_compare_and_close_already_charged_not_added_twice": True}
        if not all(checks.values()):
            raise AssertionError("A mandatory source, content, semantic or resource check failed")
        result = {"status": "root_self_review_passed_final_cost_and_exit_pending", "checks": checks,
                  "root_review_kind": "separate_self_review_not_independent_subagent", "source_identities": identities,
                  "old_contract_identity_count": 1041, "original_source_snapshots": snapshot_count,
                  "original_full_output_comparisons": comparisons, "closed_process_records": list(unique.values()),
                  "closed_leases": leases, "costs": costs, "count_contribution": 0,
                  "scope": "optional_original_full_export_live_point_rejection_predicate_without_R111_R112",
                  "formal_three_pairs_reused_not_new": True, "old_strict_failure_preserved": True,
                  "original_formal_full_tree_peak_RSS_history_not_available": True,
                  "new_components_tree_peak_measured": True,
                  "new_training_latest_combination_GT_unseen_object_or_production": "not_verified"}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": len(checks), "phase_reduction": reduction,
                          "whole_ratio": costs["preliminary_whole_ratio"], "known_fee": fee}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"stage": stage, "error_type": type(error).__name__, "error": str(error),
                                               "checks": checks, "review_before_failure_seconds": time.perf_counter() - STARTED,
                                               "count_contribution": 0})
        raise


if __name__ == "__main__":
    main()
