"""Publish scoped file-backend acceptance only after reconciling complete costs."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "hc001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    target = OUT / "ACCEPTANCE.json"
    if target.exists():
        raise FileExistsError("Never replace a published mechanism decision")
    review = read(OUT / "a2/REVIEW.json")
    campaign = read(OUT / "c0/RESULT.json")
    generic = read(OUT / "g0/HOST_GENERIC_RESULT.json")
    recovery = read(OUT / "r0/RESULT.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    prior = [read(ROOT / path) for path in state["acceptance_receipts"]]
    contributions = [row.get("additional_accepted_count", row.get("accepted_count_contribution", row.get("new_optimization_count"))) for row in prior]
    checks = {
        "required_current_source_cost_and_resource_self_review": all(review["checks"].values()),
        "original_generic_and_real_fallback_host_recovery": all(generic["checks"].values()) and all(recovery["checks"].values()),
        "all18_prior_unique_actual_acceptance_contributions_retained": len(prior) == 18 and contributions == [1] * 18
            and len(set(state["acceptance_receipts"])) == 18,
        "three_fresh_complete_pairs_retained": len(campaign["pairs"]) == 3,
        "R133_not_already_counted_and_R126_zero": not any(name.startswith("R133") for name in state["accepted_mechanisms"])
            and state["R126_new_optimization_count"] == 0}
    leases = [BASE / "dp001/PROFILE_LEASE_001.json", OUT / "GENERIC_LEASE_001.json",
        OUT / "CAMPAIGN_LEASE_001.json", OUT / "RECOVERY_LEASE_001.json",
        OUT / "REVIEW_LEASE_001.json", OUT / "REVIEW_LEASE_002.json", OUT / "REVIEW_LEASE_003.json"]
    guard_paths = [OUT / f"REVIEW_GUARD_{index:03d}.json" for index in (1, 2, 3)]
    windows = []
    for path in guard_paths:
        windows.extend(read(path)["owned_process_identities"])
    for path in leases:
        row = read(path)
        windows.extend(({"pid": row["guard_pid"], "creation_time": row["guard_creation_time"]},
            {"pid": row["child_pid"], "creation_time": row["child_creation_time"]}))
    checks["all_prior_experiment_and_review_leases_released"] = all(read(path).get("lease_released_utc") for path in leases)
    checks["all_actual_review_and_experiment_Windows_instances_gone"] = all(gone(row) for row in windows)
    ledger = [{"lease": str(path), "seconds": read(path)["child_elapsed_seconds"], "returncode": read(path)["returncode"]} for path in leases]
    first = sum(row["seconds"] for row in ledger if "CAMPAIGN_LEASE" not in row["lease"])
    whole_sides = sum(row["complete_whole_seconds"] for row in campaign["rows"])
    campaign_extra = read(OUT / "CAMPAIGN_LEASE_001.json")["child_elapsed_seconds"] - whole_sides
    checks["campaign_extra_preserves_all_parent_comparison_and_setup"] = campaign_extra > 0
    measured_finalizer = time.perf_counter() - STARTED
    common = first + campaign_extra + measured_finalizer
    surcharge = common / 6
    medians = {mode: {key: statistics.median(row[key] for row in campaign["rows"] if row["mode"] == mode) + surcharge
        for key in ("complete_phase_seconds", "complete_whole_seconds")} for mode in ("cpu", "host")}
    phase_reduction = 1 - medians["host"]["complete_phase_seconds"] / medians["cpu"]["complete_phase_seconds"]
    whole_reduction = 1 - medians["host"]["complete_whole_seconds"] / medians["cpu"]["complete_whole_seconds"]
    ratios = [{"pair": index, "complete_whole_ratio_with_common_fees":
        (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "host")["complete_whole_seconds"] + surcharge) /
        (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "cpu")["complete_whole_seconds"] + surcharge)} for index in range(3)]
    checks.update(complete_phase_speed_at_least_registered5percent=phase_reduction >= .05,
        complete_whole_separate_side_median_no_regression=whole_reduction >= 0,
        each_complete_whole_with_fees_below_original1point1=all(row["complete_whole_ratio_with_common_fees"] <= 1.1 for row in ratios))
    result = {"status": "accepted_scoped_engineering" if all(checks.values()) else "failed_final_verification",
        "mechanism_id": "R133", "mechanism": "owned_NTFS_copy_and_two_independent_complete_file_reads_on_existing_native_Windows_host",
        "new_optimization_count": 1 if all(checks.values()) else 0, "checks": checks,
        "review_kind": "root_single_chat_self_review_no_subagent", "baseline_common": ["R127_002", "P25", "Q61"],
        "raw_medians": campaign["raw_medians_not_acceptance"], "medians_with_all_measured_fees": medians,
        "complete_phase_reduction": phase_reduction, "complete_whole_reduction": whole_reduction,
        "all_original_first_failure_recovery_and_successful_review_costs": ledger,
        "first_and_required_review_seconds": first, "campaign_envelope_extra_seconds": campaign_extra,
        "finalizer_before_publication_seconds": measured_finalizer, "common_fee_seconds": common,
        "per_side_surcharge_seconds": surcharge, "pair_ratios_with_fees": ratios,
        "mathematical_model_and_training_output_changed": False,
        "scope": "existing_Windows_WSL_C_drive_and_owned_admitted_full_Gaussian_outputs_up_to256MiB",
        "complete_expected_bytes_and_independent_physical_copy_retained": True,
        "original_fallback_permission_cancel_prefixes_and_actual_host_cancel_drained": True,
        "R126_new_count": 0, "no_repeated_training_for_exact_model_bytes": True,
        "unverified": review["unverified"], "source": data_identity(Path(__file__)),
        "small_final_metadata_publication_and_outer_process_close_accounting": "charged_in_post_exit_closure_before_parent_count_update"}
    atomic_json(OUT / "FINAL_COST_INITIAL_001.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "phase_reduction": phase_reduction,
        "whole_reduction": whole_reduction, "common_fee_seconds": common}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A required final complete host-file cost gate failed")


if __name__ == "__main__":
    main()
