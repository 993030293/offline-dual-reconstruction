"""Audit the measured full-consumer cost bound before funding unrun recovery."""

import json
import msvcrt
from pathlib import Path
import statistics
import time

import psutil

from profile_vector_full_export import ROOT, BASE, LIBRARY, read
from runtime import atomic_json, data_identity

PARENT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    output = BASE / "r0"
    output.mkdir(parents=True, exist_ok=False)
    formal = read(BASE / "c0/RESULT.json")
    generic = read(BASE / "g1_adapter/RESULT.json")
    component = read(BASE / "s0/RESULT.json")
    metadata = read(BASE / "m0/RESULT.json")
    combo = read(BASE / "d0/RESULT.json")
    old = read(PARENT / "v5_cost_rank_001/CLOSURE_001.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    guard_records, identities, fees = [], [], {}
    for label, expected in (("b0", 0), ("g0", 1), ("g1", 0), ("s0", 0), ("m0", 0), ("d0", 0),
                            ("c0_first", 0), ("c0_remaining", 0)):
        lease, guard = read(BASE / (label + "_lease.json")), read(BASE / (label + "_guard.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != expected:
            raise AssertionError("An actual vector sampling lease did not close as expected")
        if guard["status"] != ("failed_child" if expected else "completed") or guard["returncode"] != expected:
            raise AssertionError("An actual vector sampling guard disagrees with its lease")
        guard_records.append(guard)
        fees[label] = lease["child_elapsed_seconds"]
        identities.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}, *guard["owned_process_identities"]])
    telemetry = read(BASE / "b0/BUILD_TELEMETRY_CLOSE.json")
    for row in telemetry["owned_instances"]:
        identities.extend([row, {"pid": row["compiler_pid"], "creation_time": row["compiler_creation_time"]}])
    for pair in formal["pairs"]:
        for mode in ("native", "vector"):
            identities.append(pair["rows"][mode]["owner"])
            identities.extend(read(BASE / "c0" / ("t" + str(pair["pair"])) / (mode + "_GUARD.json"))["owned_process_identities"])
    for mode, row in combo["rows"].items():
        identities.extend([row["owner"], row["result"]["trace"]["worker"]["identity"]])
    unique = {(row["pid"], row["creation_time"]): row for row in identities}
    medians = {mode: {key: statistics.median(p["rows"][mode][key] for p in formal["pairs"])
                     for key in ("phase_seconds", "whole_seconds")} for mode in ("native", "vector")}
    covered_formal = sum(sum(p["rows"][mode]["whole_seconds"] for mode in ("native", "vector"))
                         - p["actual_compare_seconds_charged_each_side"] for p in formal["pairs"])
    formal_envelopes = fees["c0_first"] + fees["c0_remaining"]
    unallocated = formal_envelopes - covered_formal
    if unallocated < 0: raise ValueError("The actual formal envelopes do not cover their recorded work")
    extra = sum(v for k, v in fees.items() if not k.startswith("c0_"))
    known = old["candidates"]["R123"]["known_fee_lower_bound_seconds"] + extra + unallocated
    phases = {mode: medians[mode]["phase_seconds"] + known / 6 for mode in medians}
    wholes = {mode: medians[mode]["whole_seconds"] + known / 6 for mode in medians}
    reduction = 1 - phases["vector"] / phases["native"]
    current_identity = True
    for mode in ("native", "vector"):
        saved = read(BASE / "c0/t0" / mode / "CONTRACT.json")
        current_identity = current_identity and saved["sources"] == [data_identity(row["path"]) for row in saved["sources"]]
    source_snapshots = (BASE / "b0/source_snapshot/native_vector_camera_sampling.cpp").read_bytes() == Path(__file__).with_name("native_vector_camera_sampling.cpp").read_bytes()
    checks = {
        "parent28_and_R123_family_not_already_accepted": state["additional_accepted_count"] == 28
            and not any(name.startswith("R123") for name in state["accepted_mechanisms"]),
        "registered_three_alternating_pairs_all_actual_scientific_comparisons": len(formal["pairs"]) == 3
            and all(p["comparison"]["complete_byte_pairs"] == 1046 and p["comparison"]["complete_JSON_pairs"] == 4 for p in formal["pairs"]),
        "raw_binary64_medians_and_saved_ratios_recompute_exact": medians == formal["medians"]
            and 1 - medians["vector"]["phase_seconds"] / medians["native"]["phase_seconds"] == formal["raw_stage_reduction"],
        "old_family_fees_492_and_all_new_build_failure_generic_component_metadata_combo_kept": known > 640 and len(fees) == 8,
        "actual_formal_windows_only_unallocated_tail_added_not_double_counted": unallocated >= 0,
        "known_fee_lower_bound_complete_stage_upper_below_registered3percent": reduction < .03,
        "generic43_and37_alias_boundary_prefix_admission_guards_passed": all(generic["checks"].values()) and len(generic["boundary_cases"]) == 37,
        "original43_fixture_results_and_legacy_H_scope_kept": all(read(BASE / "g1/RESULT.json")["checks"].values())
            and generic["legacy_H3_semantic_fixtures_already_previously_used_not_blind_GT"],
        "actual_g0_numpy_camera_reader_failure_saved_and_charged": "truth value of an array" in (BASE / "g0.log").read_text(encoding="utf-8")
            and (BASE / "g0_adapter/source_snapshot/check_vector_camera_sampling.py").is_file(),
        "fixed32_component_all_original_JPEG_metadata_K_and_source_guards": all(component["checks"].values()) and component["continue_to_full_consumer"],
        "real_EXIF_DPI_ICC_before_write_native_fallback_all12_files_each": all(metadata["checks"].values())
            and all(row["all_complete_native_files_exact"] == 12 for row in metadata["rows"]),
        "actual_R132_R140_DAG_combo_full1050_semantics_and_one_pair_only": all(combo["checks"].values())
            and combo["not_three_pair_medians_or_independent_optimization"],
        "all_current_first_formal_sources_and_compiled_source_snapshots_exact": current_identity and source_snapshots,
        "actual_SDK_owned_telemetry_instances_closed_with_identity": telemetry["strict_parent_birth_window_exe_and_PID_only"]
            and all(row["exact_instance_exited"] and closed(row) for row in telemetry["owned_instances"]),
        "all_saved_actual_unique_instances_closed": all(closed(row) for row in unique.values()),
        "all_eight_actual_time_and_RSS_resource_guards_kept": all(g["elapsed_seconds"] <= g["maximum_seconds"]
            and g["peak_tree_RSS_bytes"] <= g["maximum_RSS_bytes"] for g in guard_records),
        "real_partial_and_full_recovery_not_run_after_cost_bound_failure": not (BASE / "v0").exists(),
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A vector sampling negative-cost review failed")
    with (PARENT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    (snapshots / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    atomic_json(output / "RESULT.json", {"status": "closed_negative_known_fee_lower_bound_stage_upper_below3percent_R123_vector0",
        "checks": checks, "root_separate_self_review_not_independent_agent": True, "actual_fees": fees,
        "old_family_cost_rank_known_lower_bound_seconds": old["candidates"]["R123"]["known_fee_lower_bound_seconds"],
        "actual_formal_envelopes_seconds": formal_envelopes, "actual_formal_covered_once_seconds": covered_formal,
        "actual_formal_unallocated_tail_seconds": unallocated, "known_fee_lower_bound_before_this_review_exit_seconds": known,
        "medians_before_fees": medians, "complete_phase_medians_lower_bound_seconds": phases,
        "complete_whole_medians_lower_bound_seconds": wholes, "complete_stage_reduction_upper_bound": reduction,
        "raw_stage_reduction": formal["raw_stage_reduction"], "raw_whole_reduction": formal["raw_whole_reduction"],
        "combo_single_pair_raw_whole_reduction": combo["raw_whole_reduction"],
        "closed_instances": list(unique.values()), "OS_lock_actually_acquired_released": True,
        "recovery_normal_CPP_filesystem_error_order_external_full_GT_training_production_not_proven": True,
        "prepared_recovery_script_not_executed": True, "historical_unknown_text_research_or_publisher_fees_not_zero": True,
        "source": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started,
        "new_optimization_count": 0, "parent_count": 28, "goal_remains_active": True})
    print(json.dumps({"checks_passed": len(checks), "known_fee_lower_bound": known,
        "complete_stage_reduction_upper_bound": reduction, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
