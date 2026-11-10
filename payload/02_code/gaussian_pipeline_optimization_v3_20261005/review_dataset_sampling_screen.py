"""Close the complete dataset negative screen and inspect actual stripe costs."""

import datetime
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "ud001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    output = OWN / "r0"
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    campaign = read(OWN / "c0/RESULT.json")
    p0, p1 = read(OWN / "p0/RESULT.json"), read(OWN / "p1/RESULT.json")
    pairs = [(read(OWN / (name + "_lease.json")), read(OWN / (name + "_guard.json"))) for name in ("p0", "p1", "c0")]
    identities = []
    for lease, guard in pairs:
        identities.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                           {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
        identities.extend(guard["owned_process_identities"])
    identities.extend(row["owner"] for row in campaign["rows"])
    workers = campaign["rows"][1]["result"]["trace"]["workers"]
    identities.extend(row["owner"] for row in workers)
    identities = list({(row["pid"], row["creation_time"]): row for row in identities}.values())
    contracts = [read(OWN / "c0" / mode / "CONTRACT.json") for mode in ("native", "cached")]
    sources_exact = all(contract["sources"] == [data_identity(row["path"]) for row in contract["sources"]] for contract in contracts)
    snapshots_exact = True
    for mode, contract in zip(("native", "cached"), contracts):
        for index, row in enumerate(contract["sources"]):
            source = Path(row["path"])
            if source.suffix in (".py", ".cpp"):
                snapshot = OWN / "c0" / mode / "source_snapshot" / (f"{index:02d}_" + source.name)
                snapshots_exact = snapshots_exact and snapshot.read_bytes() == source.read_bytes()
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    original_sources = (CODE / "screen_complete_dataset_sampling.py").read_text(encoding="utf-8")
    adapter_source = (CODE / "dataset_sampling_carrier.py").read_text(encoding="utf-8")
    checks = {
        "p0_failed_configuration_evidence_preserved": p0["status"] == "screen_rejected_pixel_JPEG_metadata_or_native_carrier_complete_contract",
        "p1_original_pixel_JPEG_metadata_model_configuration_checks_passed": p1["status"] == "candidate_interface_screen_passed_full1016_cost_quality_recovery_unverified"
            and all(p1["candidate_checks"].values()) and all(p1["protection_checks"].values())
            and all(all(row["checks"].values()) for row in p1["rows"]),
        "both_full_original1016_75767_959120_side_checks_passed": all(all(row["result"]["checks"].values()) for row in campaign["rows"]),
        "all1025_complete_native_file_pairs_exact": all(campaign["checks"].values()) and campaign["complete_file_pairs"] == 1025,
        "raw_cost_rejects_before_any_full1050_or_formal": campaign["stage_ratio_before_family_fees"] > 1
            and campaign["whole_ratio_before_family_fees"] > 1.1
            and not campaign["continue_to_full1050_only_if_stage_at_least5percent_and_whole_at_most1point1"],
        "all_current_and_executed_sources_exact": sources_exact and snapshots_exact,
        "all_source_and_input_pre_post_guards_from_actual_side_execution_kept": all(row["result"]["checks"]["all_input_and_source_identities_unchanged"] for row in campaign["rows"]),
        "every_saved_actual_process_instance_exited": all(closed(row) for row in identities),
        "three_actual_guards_and_leases_completed_and_released": all(lease["status"] == "child_exited" and lease["returncode"] == 0 and lease["lease_released_utc"]
            and guard["status"] == "completed" and guard["returncode"] == 0 for lease, guard in pairs),
        "four_actual_workers_same_complete_work_and_closed": len(workers) == 4 and sum(row["count"] for row in workers) == 1015
            and all(row["sampler_trace"]["builds"] == 1 and row["sampler_trace"]["closed"] and row["sampler_trace"]["native_DLL_closed"] for row in workers),
        "actual_runtime_thread_counts_not_inferred_from_environment": all(type(row["actual_process_thread_peak"]) is int and row["actual_process_thread_peak"] >= 1 for row in workers),
        "full_file_comparison_and_join_metering_remain_in_executed_source": "members[0][name].read_bytes() != members[1][name].read_bytes()" in original_sources
            and "code = child.wait()" in original_sources and "with ProcessPoolExecutor" in adapter_source,
        "parent_count24_target50_and_no_new_acceptance_kept": parent["additional_accepted_count"] == 24 and parent["additional_target"] == 50
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == len(parent["acceptance_receipts"]) == 24
            and campaign["new_optimization_count"] == p0["new_optimization_count"] == p1["new_optimization_count"] == 0,
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A complete dataset negative screen review failed")
    component_sums = {key: sum(row["component_task_seconds"][key] for row in workers)
                      for key in ("read_seconds", "sample_seconds", "write_seconds")}
    build_seconds = sum(row["sampler_trace"]["build_seconds"] for row in workers)
    old = read(OUT / "rm_native001/FINAL_COST_NEGATIVE_001.json")
    current_cost = sum(lease["child_elapsed_seconds"] for lease, _ in pairs)
    result = {"status": "closed_negative_complete1016_cold_dataset_cost_screen", "checks": checks,
        "root_separate_self_review_not_independent_agent": True, "instances": identities,
        "raw_native_and_cached_stage_seconds": [row["result"]["operation_seconds"] for row in campaign["rows"]],
        "raw_native_and_cached_actual_process_whole_seconds": [row["actual_process_whole_seconds"] for row in campaign["rows"]],
        "raw_stage_ratio": campaign["stage_ratio_before_family_fees"], "raw_whole_ratio": campaign["whole_ratio_before_family_fees"],
        "complete_file_pairs": campaign["complete_file_pairs"], "complete_exact_bytes_per_side": campaign["complete_byte_pairs_bytes"],
        "stripe_component_task_sums_NOT_whole_wall": component_sums, "all_four_cold_map_build_task_sum_seconds": build_seconds,
        "stripe_whole_seconds": [row["whole_stripe_seconds"] for row in workers],
        "actual_worker_thread_peaks": [row["actual_process_thread_peak"] for row in workers],
        "sample_task_fraction_of_all_stripe_components": component_sums["sample_seconds"] / sum(component_sums.values()),
        "current_complete_probe_and_pair_lease_cost_seconds": current_cost,
        "old_family_complete_measured_envelopes_including_old24_formal_lower_bound_seconds": old["total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds"],
        "old_different24_consumer_not_reused_as_this1016_formal_or_excluded_from_fees": True,
        "all_measured_family_cost_lower_bound_before_this_review_exit_seconds": current_cost + old["total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds"],
        "full1050_R132_R140_formal_generic_recovery_not_run_after_raw_cost_failure": True,
        "unknown_object_GT_new_training_full_mapping_LIO_production_preview_not_run": True,
        "new_optimization_count": 0, "parent_goal_complete": False,
        "review_before_result_seconds": time.perf_counter() - started,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "raw_native_and_cached_stage_seconds", "raw_stage_ratio", "raw_whole_ratio", "stripe_component_task_sums_NOT_whole_wall", "all_four_cold_map_build_task_sum_seconds", "actual_worker_thread_peaks", "sample_task_fraction_of_all_stripe_components", "new_optimization_count")}))


if __name__ == "__main__":
    main()
