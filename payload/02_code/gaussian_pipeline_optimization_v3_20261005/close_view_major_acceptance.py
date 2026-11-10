"""Close the exited review and decide the unchanged memory policy from actual fees."""

import time

STARTED = time.perf_counter()

import ast
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    if (OUT / "ACCEPTANCE.json").exists():
        raise FileExistsError("Never overwrite a mechanism decision")
    review = read(OUT / "a2/REVIEW.json")
    linux = read(OUT / "a2/LINUX_REVIEW.json")
    lease = read(OUT / "REVIEW_LEASE_003.json")
    guard = read(OUT / "REVIEW_GUARD_003.json")
    rows = review["Windows"] + guard["owned_process_identities"]
    rows += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
             {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}, read(OUT / "a2/PROCESS_IDENTITY.json")]
    unique = {(row["pid"], row["creation_time"]): {"pid": row["pid"], "creation_time": row["creation_time"]} for row in rows}
    windows = [{**row, "gone": gone(row)} for row in unique.values()]
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    checks = {"all_twelve_original_review_gates_passed": all(review["checks"].values()),
              "actual_successful_review_guard_and_lease_closed": guard["status"] == "completed" and lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
              "all_actual_prior_and_review_Windows_instances_gone": all(row["gone"] for row in windows),
              "actual_original_Linux_closure_preserved": all(linux["checks"].values()),
              "all_prior13_group_leases_released": len(list(OUT.glob("*_LEASE_*.json"))) == 13
                  and all(bool(read(path).get("lease_released_utc")) for path in OUT.glob("*_LEASE_*.json")),
              "native_parent50_goal_active_same_thread": read(OUT / "NATIVE_GOAL_SOURCE_GET.json")["goal"]["status"] == "active"
                  and read(OUT / "NATIVE_GOAL_SOURCE_GET.json")["goal"]["threadId"] == "01a079ba-7973-7780-bf60-a034f8b50aa3"}
    fees = review["all_prior_shared_fees_before_this_exited_review_seconds"] + lease["child_elapsed_seconds"]
    close_seconds = time.perf_counter() - STARTED
    fees += close_seconds
    raw = review["raw_medians"]
    medians = {mode: {key: raw[mode][key] + fees/6 for key in ("complete_phase_seconds", "complete_whole_seconds")}
               for mode in ("native", "view")}
    campaign = read(OUT / "c2/RESULT.json")
    ratios = [(next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "view")["complete_whole_seconds"] + fees/6) /
              (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "native")["complete_whole_seconds"] + fees/6) for index in range(3)]
    checks["final_memory20percent_and_whole_original_gates_after_all_measured_fees"] = review["actual_CUDA_reserved_peak_reduction"] >= .2
    checks["final_memory20percent_and_whole_original_gates_after_all_measured_fees"] &= medians["view"]["complete_whole_seconds"] <= medians["native"]["complete_whole_seconds"] and all(ratio <= 1.1 for ratio in ratios)
    result = {"status": "accepted_scoped_engineering" if all(checks.values()) else "failed_close", "checks": checks,
              "new_optimization_count": 1 if all(checks.values()) else 0,
              "mechanism": "R135_VIEW_MAJOR_LAST_USE_REFERENCE_STATISTICS_EVALUATION",
              "primary_benefit": "actual_evaluation_process_Torch_CUDA_reserved_peak",
              "actual_CUDA_reserved_peak_bytes": {mode: raw[mode]["peak_CUDA_reserved_bytes"] for mode in ("native", "view")},
              "actual_CUDA_reserved_peak_reduction": review["actual_CUDA_reserved_peak_reduction"],
              "complete_medians_with_all_measured_fees": medians,
              "complete_whole_reduction": 1-medians["view"]["complete_whole_seconds"]/medians["native"]["complete_whole_seconds"],
              "profile_phase_reduction_not_speed_acceptance": 1-medians["view"]["complete_phase_seconds"]/medians["native"]["complete_phase_seconds"],
              "whole_pair_ratios_with_fees": ratios, "all_measured_shared_fees_seconds": fees,
              "per_side_surcharge_seconds": fees/6, "prior_R134_all_measured_fees_seconds": review["prior_R134_all_measured_first_and_failed_groups_seconds"],
              "exited_successful_review_seconds": lease["child_elapsed_seconds"], "closure_before_publication_seconds": close_seconds,
              "all_current_source_and_input_identities": len(review["current_original_sources_and_inputs"]),
              "saved_recorded_revision_source_copies": len(review["recorded_revision_source_copies"]),
              "actual_prior_and_review_Windows_instances_closed": len(windows), "actual_prior_Linux_instances_closed": len(linux["instances"]),
              "maximum_absolute_SSIM_error": review["maximum_absolute_SSIM_error"],
              "root_single_chat_self_review": True, "no_subagent": True,
              "scope": "current_existing_two12000_models_newmap1016_complete24_evaluation_Q30_Q61_R100",
              "external_scope": "Oxford360_nonblind_reference_statistic_component_two_synthetic_tensor_variants_not_trained_external_models",
              "production_new_training_new_preview_independent_GT": "not_run",
              "small_final_metadata_report_publication_after_exited_review_and_close_timer_not_claimed_included": True,
              "own_closing_script_exit_requires_outer_caller_check": True}
    atomic_json(OUT / "FINAL_COST_001.json", result)
    atomic_json(OUT / "FINAL_CLOSURE_001.json", {"status": "closed_scoped_acceptance_ready" if all(checks.values()) else "failed_close", "checks": checks, "Windows": windows,
                "new_optimization_count": result["new_optimization_count"], "closing_script_own_pid": psutil.Process().pid,
                "closing_script_own_creation_time": psutil.Process().create_time()})
    if not all(checks.values()):
        raise AssertionError("A final actual owner, fee, native goal or registered memory gate failed")
    atomic_json(OUT / "ACCEPTANCE.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "actual_CUDA_reserved_peak_reduction", "all_measured_shared_fees_seconds", "complete_medians_with_all_measured_fees", "complete_whole_reduction", "actual_prior_and_review_Windows_instances_closed")}), flush=True)


if __name__ == "__main__":
    main()
