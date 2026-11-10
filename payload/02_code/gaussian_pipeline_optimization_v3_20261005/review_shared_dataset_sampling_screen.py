"""Review exact shared-map output, actual lifecycle and the single-pair cost bound."""

import json
from multiprocessing import shared_memory
from pathlib import Path
import sys
import time

import psutil

from dataset_sampling_carrier import ROOT, OWN, closed
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    output = OWN / "r1"
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    pair = read(OWN / "s0/RESULT.json")
    workers = pair["rows"][1]["result"]["trace"]["workers"]
    trace = pair["rows"][1]["result"]["trace"]
    pairs = [(read(OWN / (name + "_lease.json")), read(OWN / (name + "_guard.json"))) for name in ("p0", "p1", "c0", "r0", "s0")]
    identities = []
    for lease, guard in pairs:
        identities.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}, *guard["owned_process_identities"]])
    identities.extend(row["owner"] for row in pair["rows"])
    identities.extend(row["owner"] for row in workers)
    identities.append(read(OWN / "r0/PROCESS_IDENTITY.json"))
    identities = list({(row["pid"], row["creation_time"]): row for row in identities}.values())
    contracts = [read(OWN / "s0" / mode / "CONTRACT.json") for mode in ("native", "cached")]
    source_exact = True
    for mode, contract in zip(("native", "cached"), contracts):
        source_exact = source_exact and contract["sources"] == [data_identity(row["path"]) for row in contract["sources"]]
        for index, row in enumerate(contract["sources"]):
            source = Path(row["path"])
            if source.suffix in (".py", ".cpp"):
                source_exact = source_exact and source.read_bytes() == (OWN / "s0" / mode / "source_snapshot" / (f"{index:02d}_" + source.name)).read_bytes()
    segments_absent = True
    for header in trace["shared_headers"]:
        try:
            segment = shared_memory.SharedMemory(name=header["name"], create=False)
        except FileNotFoundError:
            continue
        segment.close()
        segments_absent = False
    old = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001/FINAL_COST_NEGATIVE_001.json")
    current_cost = sum(lease["child_elapsed_seconds"] for lease, _ in pairs)
    lower = old["total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds"] + current_cost
    stage = [row["result"]["operation_seconds"] for row in pair["rows"]]
    whole = [row["actual_process_whole_seconds"] for row in pair["rows"]]
    adjusted_stage = [value + lower / 6 for value in stage]
    adjusted_whole = [value + lower / 6 for value in whole]
    diagnostic_stage_reduction_upper = 1 - adjusted_stage[1] / adjusted_stage[0]
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    closed_record = read(OWN / "s0/cached/sampling_proof/SHARED_SEGMENTS_CLOSED.json")
    driver_a = (OWN / "s0/native/source_snapshot/COMPLETE_EXECUTED_DRIVER.py").read_text(encoding="utf-8")
    driver_b = (OWN / "s0/cached/source_snapshot/COMPLETE_EXECUTED_DRIVER.py").read_text(encoding="utf-8")
    checks = {
        "complete_same1016_75767_959120_both_original_side_guards": all(all(row["result"]["checks"].values()) for row in pair["rows"]),
        "all1025_original_files_byte_exact": pair["complete_file_pairs"] == 1025 and all(pair["checks"].values()),
        "actual_shared_eight_process_route_and_full_work": trace["route"] == "native_full_model_carrier_shared_eight_process_RGB_sampling"
            and len(workers) == 8 and sum(row["count"] for row in workers) == 1015,
        "actual_shared_source_guards_and_saved_driver_exact": source_exact and driver_a == driver_b
            and '"native_full_model_carrier_shared_eight_process_RGB_sampling"' in driver_a,
        "one_parent_build_zero_worker_rebuilds_original_DLL_closed": trace["parent_sampler_trace"]["builds"] == 1
            and trace["parent_sampler_trace"]["closed"] and trace["parent_sampler_trace"]["native_DLL_closed"]
            and all(row["sampler_trace"]["builds"] == 0 and row["sampler_trace"]["closed"] and row["sampler_trace"]["native_DLL_closed"] for row in workers),
        "all_actual_shared_handles_closed_and_names_cannot_reopen": segments_absent and closed_record["all_handles_closed"]
            and closed_record["all_names_unavailable_after_shutdown"] and all(row["sampler_trace"]["shared_handles_closed"] for row in workers),
        "geometry_map_copy_peak_within_original256MiB": trace["map_copy_peak_bytes"] <= 256 * 1024**2 and len(trace["shared_headers"]) == 5,
        "every_saved_actual_process_instance_exited": all(closed(row) for row in identities),
        "all_five_previous_guards_and_leases_completed_and_released": all(lease["status"] == "child_exited" and lease["returncode"] == 0
            and lease["lease_released_utc"] and guard["status"] == "completed" and guard["returncode"] == 0 for lease, guard in pairs),
        "parent_environment_RNG_full_work_and_source_checks_kept": all(trace["checks"].values()),
        "registered_raw_first_pair_stage5_and_whole1point1_passed": pair["stage_ratio_before_family_fees"] <= .95
            and pair["whole_ratio_before_family_fees"] <= 1.1 and pair["continue_to_full1050_only_if_stage_at_least5percent_and_whole_at_most1point1"],
        "this_single_pair_known_cost_bound_below5_not_formal_acceptance": 0 < diagnostic_stage_reduction_upper < .05,
        "old_four_process_failure_and_full_family_cost_not_removed": read(OWN / "c0/RESULT.json")["stage_ratio_before_family_fees"] > 1
            and lower > 490 and old["new_optimization_count"] == 0,
        "parent_count24_and_zero_new_acceptance_kept": parent["additional_accepted_count"] == 24 and pair["new_optimization_count"] == 0,
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A shared complete dataset screen review failed")
    component_sums = {key: sum(row["component_task_seconds"][key] for row in workers)
                      for key in ("read_seconds", "sample_seconds", "write_seconds")}
    result = {"status": "deferred_single_complete1016_pair_raw_faster_but_known_cost_bound_stage_below5_no_formal_acceptance",
        "checks": checks, "root_separate_self_review_not_independent_agent": True, "instances": identities,
        "native_shared_complete_stage_seconds": stage, "native_shared_actual_process_whole_seconds": whole,
        "raw_stage_reduction": 1 - pair["stage_ratio_before_family_fees"], "raw_whole_reduction": 1 - pair["whole_ratio_before_family_fees"],
        "complete_file_pairs": pair["complete_file_pairs"], "complete_exact_bytes_per_side": pair["complete_byte_pairs_bytes"],
        "complete_current_five_lease_fees_seconds": current_cost,
        "old_family_measured_envelopes_including_other_consumer_formal_lower_bound_seconds": old["total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds"],
        "all_known_family_fee_lower_bound_before_this_review_exit_seconds": lower,
        "single_pair_stage_with_fee_lower_bound": adjusted_stage,
        "single_pair_whole_with_fee_lower_bound": adjusted_whole,
        "single_pair_stage_reduction_upper_bound": diagnostic_stage_reduction_upper,
        "not_three_pair_medians_or_full1050_R132_R140_result": True,
        "full1050_formal_generic_actual_fault_recovery_new_training_GT_unknown_objects_production_not_run": True,
        "stripe_component_task_sums_NOT_whole_wall": component_sums,
        "actual_worker_thread_peaks": [row["actual_process_thread_peak"] for row in workers],
        "map_copy_peak_bytes": trace["map_copy_peak_bytes"],
        "unmetered_text_research_time_not_zero": True, "new_optimization_count": 0,
        "parent_goal_complete": False, "review_before_result_seconds": time.perf_counter() - started}
    atomic_json(output / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "native_shared_complete_stage_seconds", "raw_stage_reduction", "raw_whole_reduction", "all_known_family_fee_lower_bound_before_this_review_exit_seconds", "single_pair_stage_reduction_upper_bound", "map_copy_peak_bytes", "new_optimization_count")}))


if __name__ == "__main__":
    main()
