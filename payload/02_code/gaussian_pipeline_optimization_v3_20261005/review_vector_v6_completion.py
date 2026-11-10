"""Separately audit complete export recovery, actual closure, and V6 costs."""

import time

STARTED = time.perf_counter()

import json
import msvcrt
from multiprocessing import shared_memory
from pathlib import Path
import shutil
import statistics

import psutil

from profile_vector_full_export import BASE, ROOT, read
from profile_sparse_ply_oracle import compare_outputs
from runtime import atomic_json, data_identity

OWN = BASE / "v6_completion_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def cost(actual_review_and_publication_seconds=0.):
    formal = read(BASE / "c0/RESULT.json")
    medians = {mode: {key: statistics.median(pair["rows"][mode][key] for pair in formal["pairs"])
        for key in ("phase_seconds", "whole_seconds")} for mode in ("native", "vector")}
    if medians != formal["medians"]:
        raise AssertionError("An original formal median differs")
    components = {
        "original_known_old_family_and_V5_execution_lower_bound_seconds": read(BASE / "FINAL_POST_EXIT_001.json")["known_fee_lower_bound_seconds"],
        "old_last_postcheck_unallocated_execution_tail_allowance_not_measured_seconds": 3.,
        "actual_current_complete_evidence_audit_lease_seconds": read(OWN / "p0_lease.json")["child_elapsed_seconds"],
        "actual_partial_interruption_and_full1050_recovery_lease_seconds": read(OWN / "recovery_lease.json")["child_elapsed_seconds"],
        "actual_native_JPEG_write_fault_and_cleanup_lease_seconds": read(OWN / "filesystem_error_lease.json")["child_elapsed_seconds"],
        "current_native_text_and_metadata_execution_allowance_not_measured_seconds": 15.,
        "actual_root_review_and_publication_so_far_seconds": actual_review_and_publication_seconds,
        "final_post_exit_text_ledger_and_exit_reserved_seconds": 5.,
    }
    extra = sum(components.values())
    adjusted = {mode: {key: value + extra / 6 for key, value in row.items()} for mode, row in medians.items()}
    return {"components": components, "known_extra_execution_seconds_including_declared_allowances": extra,
        "six_side_symmetric_seconds_surcharge": extra / 6, "original_formal_raw_medians": medians,
        "adjusted_medians_under_registered_six_side_model": adjusted,
        "raw_stage_reduction": formal["raw_stage_reduction"], "raw_whole_reduction": formal["raw_whole_reduction"],
        "net_complete_stage_reduction_under_registered_six_side_model": 1 - adjusted["vector"]["phase_seconds"] / adjusted["native"]["phase_seconds"],
        "raw_whole_median_ratio": medians["vector"]["whole_seconds"] / medians["native"]["whole_seconds"],
        "adjusted_whole_median_ratio": adjusted["vector"]["whole_seconds"] / adjusted["native"]["whole_seconds"],
        "each_original_whole_ratios": [pair["whole_ratio"] for pair in formal["pairs"]],
        "all_formal_work_already_charged_once_no_six_side_recharge": True,
        "unmetered_text_research_time_not_zero_and_no_total_research_ROI": True}


def main():
    destination = OWN / "SELF_REVIEW_V6_001.json"
    if destination.exists():
        raise FileExistsError("A unique actual self-review receipt is required")
    current = read(OWN / "p0/CURRENT_IDENTITIES.json")
    if any(data_identity(row["path"]) != row for row in current):
        raise AssertionError("A current complete original source or input changed")
    old_review, old_post = read(BASE / "r0/RESULT.json"), read(BASE / "FINAL_POST_EXIT_001.json")
    old_negative = read(BASE / "FINAL_NEGATIVE_CLOSURE_001.json")
    recovered = read(OWN / "recovery/RESULT.json")
    fault = read(OWN / "filesystem_error/FAULT_RESULT.json")
    formal = read(BASE / "c0/RESULT.json")
    comparison = compare_outputs(BASE / "c0/t0/native/export", OWN / "recovery/recovered/export")
    source_copies = 0
    new_sources = {}
    for folder in (OWN / "recovery/recovered", OWN / "filesystem_error"):
        contract = read(folder / "CONTRACT.json")
        for row in contract["sources"] + contract["inputs"]:
            if data_identity(row["path"]) != row:
                raise AssertionError("An actual completion source or original input changed")
        for index, row in enumerate(contract["sources"]):
            if Path(row["path"]).suffix in (".py", ".cpp", ".json"):
                copy = folder / "source_snapshot" / (f"{index:03d}_" + Path(row["path"]).name)
                if data_identity(copy)["sha256"] != row["sha256"]:
                    raise AssertionError("An actual executed completion source copy differs")
                source_copies += 1
        new_sources.update({row["path"]: row for row in contract["sources"]})
    for name in ("RECOVERY_BRIDGE_CONTRACT.json", "FILESYSTEM_FAULT_CONTRACT.json"):
        for row in read(OWN / name)["sources"]:
            if data_identity(row["path"]) != row:
                raise AssertionError("The executed completion bridge changed")
            new_sources[row["path"]] = row
    interrupted = recovered["interruptions"]
    headers = interrupted[1]["headers"] + read(OWN / "filesystem_error/sampling_proof/SHARED_SEGMENTS_CREATED.json")["headers"]
    for header in headers:
        try:
            segment = shared_memory.SharedMemory(name=header["name"], create=False)
        except FileNotFoundError:
            continue
        segment.close()
        raise AssertionError("An actual owned shared segment remains available")
    identities = {(row["pid"], row["creation_time"]): row for row in old_review["closed_instances"]}
    for path in OWN.rglob("*_GUARD.json"):
        for row in read(path)["owned_process_identities"]:
            identities[(row["pid"], row["creation_time"])] = row
    leases = [read(OWN / (name + "_lease.json")) for name in ("p0", "recovery", "filesystem_error")]
    for lease in leases:
        for prefix in ("guard", "child"):
            row = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
            identities[(row["pid"], row["creation_time"])] = row
    for row in [*recovered["interruptions"][0]["instances"], *recovered["interruptions"][1]["instances"], recovered["recovered_process"]]:
        identities[(row["pid"], row["creation_time"])] = row
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    parent = read(DOC / "CURRENT_STATE.json")
    calculated = cost(time.perf_counter() - STARTED)
    guards = [read(path) for path in OWN.rglob("*_GUARD.json")]
    checks = {
        "current32_unique_and_R123_family_unaccepted": parent["additional_accepted_count"] == 32
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 32
            and not any(name.startswith("R123") for name in parent["accepted_mechanisms"]),
        "V6_uniform2_and_original_whole_protections_unchanged": parent["V6_minimum_complete_stage_net_reduction"] == .02
            and calculated["raw_whole_median_ratio"] <= 1.01 and calculated["adjusted_whole_median_ratio"] <= 1.01
            and all(value <= 1.1 for value in calculated["each_original_whole_ratios"]),
        "three_original_full1050_pairs_revalidated_and_no_reselected_window": len(formal["pairs"]) == 3
            and all(read(OWN / "p0/RESULT.json")["checks"].values()),
        "both_actual_partial_exports_and_each_actual_instance_closed": len(interrupted) == 2
            and all(row["actual_nonzero_returncode"] != 0 and row["actual_partial_JPEG_count_at_trigger"] >= 17
                and row["all_actual_instances_closed"] for row in interrupted),
        "original_full_recovery_guards_and_full1050_comparison_exact": all(recovered["checks"].values())
            and comparison["complete_byte_pairs"] == 1046 and comparison["complete_JSON_pairs"] == 4
            and comparison["complete_compared_bytes"] == 746975609,
        "actual_native_JPEG_error_and_hook_bindings_RNG_environment_preserved": len(fault["checks"]) == 6 and all(fault["checks"].values()),
        "all_ten_actual_shared_names_unavailable_now": len(headers) == 10,
        "all_original_current_sources_inputs_and_executed_new_copies_exact": len(current) == 1053 and source_copies >= 48,
        "all_old_generic_alias_boundaries_metadata_and_related_DAG_not_recounted": all(old_review["checks"].values()),
        "old_V5_cost_negative_and_old_g0_failure_preserved": old_post["new_optimization_count"] == old_negative["new_optimization_count"] == 0
            and old_post["complete_stage_reduction_upper_bound"] < .03
            and "truth value of an array" in (BASE / "g0.log").read_text(encoding="utf-8"),
        "all_actual_old_and_new_owned_instances_closed": all(closed(row) for row in identities.values()),
        "all_three_new_leases_released_and_OS_lock_reacquired": all(lease["returncode"] == 0 and lease.get("lease_released_utc") for lease in leases),
        "all_current_time_RSS_and_full_namespace_disk_budgets": all(row["elapsed_seconds"] <= row["maximum_seconds"]
            and row["peak_tree_RSS_bytes"] <= row["maximum_RSS_bytes"] for row in guards)
            and sum(lease["child_elapsed_seconds"] for lease in leases) <= 600
            and sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) <= 30 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        "all_known_old_new_costs_allowances_with_net2percent": calculated["net_complete_stage_reduction_under_registered_six_side_model"] >= .02,
        "single_family_scope_no_new_training_mapper_LIO_GT_or_production_claim": parent["additional_target"] == 50
            and parent["status"].startswith("active") and recovered["new_optimization_count"] == fault["new_optimization_count"] == 0,
    }
    shutil.copyfile(__file__, OWN / "EXECUTED_ROOT_REVIEW_V6.py")
    receipt = {"status": "root_V6_completion_review_passed" if all(checks.values()) else "failed_root_V6_completion_review",
        "checks": checks, "cost": calculated, "actual_closed_instances": list(identities.values()),
        "actual_shared_names_checked_unavailable": headers, "actual_new_executed_source_copies": source_copies,
        "current_original_source_input_identities": len(current), "current_new_sources": list(new_sources.values()),
        "reviewer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "actual_executed_review_source": data_identity(Path(__file__).resolve()),
        "root_review_is_independent_agent": False, "new_optimization_count": 0,
        "review_seconds_before_publication": time.perf_counter() - STARTED}
    atomic_json(destination, receipt)
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "net_stage_reduction": calculated[
        "net_complete_stage_reduction_under_registered_six_side_model"]}), flush=True)


if __name__ == "__main__":
    main()
