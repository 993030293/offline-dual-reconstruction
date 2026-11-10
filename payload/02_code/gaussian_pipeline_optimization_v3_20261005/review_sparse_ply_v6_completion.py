"""Separately audit sparse completion evidence without changing old results."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OLD = OUT / "sparse_ply_oracle_001"
OWN = OUT / "v6_residual_cost_003"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ended(pid, born):
    try:
        return psutil.Process(pid).create_time() != born
    except psutil.NoSuchProcess:
        return True


def main():
    formal = read(OLD / "campaign_002/RESULT.json")
    contract = read(OLD / "campaign_002/CONTRACT.json")
    current = read(OWN / "sparse2/RESULT.json")
    current_contract = read(OWN / "sparse2/CONTRACT.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks = {"parent33_active_R111_not_accepted_or_duplicate": state["additional_accepted_count"] == 33
        and state["additional_target"] == 50 and state["status"].startswith("active")
        and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 33
        and len(state["acceptance_receipts"]) == 33
        and not any(name.startswith("R111") for name in state["accepted_mechanisms"])}
    medians = {mode: {key: statistics.median(pair["rows"][mode][key] for pair in formal["pairs"])
               for key in ("phase_seconds", "whole_seconds")} for mode in ("native", "columns")}
    checks["three_old_fresh_complete_exports_all_full_contents_kept_not_new_fresh"] = (
        len(formal["pairs"]) == 3 and medians == formal["medians"] and all(
            pair["verification"]["complete_byte_pairs"] == 1046
            and pair["verification"]["complete_JSON_pairs"] == 4
            and pair["verification"]["complete_compared_bytes"] == 746975609
            and all(row["child_returncode"] == 0 and row["trace"]["original_binding_restored"]
                and len(row["trace"]["calls"]) == 1 and row["trace"]["calls"][0]["points"] == 75767
                and row["trace"]["calls"][0]["status"] == "passed_complete_official_roundtrip"
                for row in pair["rows"].values()) for pair in formal["pairs"]))
    checks["old_failed10percent_result_preserved"] = formal["status"] == "complete_export_cost_guard_failed" and not formal["checks"]["phase_reduction_at_least10pct"]
    checks["full1037_current_old_source_and_original_input_identities"] = (
        len(contract["sources_and_inputs"]) == 1037
        and [data_identity(row["path"]) for row in contract["sources_and_inputs"]] == contract["sources_and_inputs"])
    copies = sorted((OLD / "campaign_002/source_snapshot").iterdir())
    checks["all_nine_old_actual_executed_source_copies_exact"] = len(copies) == 9 and all(
        data_identity(path)["sha256"] == row["sha256"]
        for path, row in zip(copies, contract["sources_and_inputs"][:9], strict=True))
    checks["old_full_export_file_membership_currently_present"] = all(
        len([path for path in Path(row["folder"]).rglob("*") if path.is_file()]) == 1050
        for pair in formal["pairs"] for row in pair["rows"].values())
    old_guards = [read(OLD / name / "RESULT.json") for name in ("generic_002", "rejection_001", "compare_guard_001")]
    checks["saved_generic27_badPLY3_comparator13_and_their_current_sources_kept"] = all(
        all(row["checks"].values()) and all(data_identity(identity["path"]) == identity for identity in row["source"])
        for row in old_guards)
    checks["current_all_public_quality_protections_passed"] = all(current["checks"].values())
    checks["all_current_public_and_source_identities_and_actual_source_copies_kept"] = (
        [data_identity(row["path"]) for row in current_contract["sources_and_inputs"]] == current_contract["sources_and_inputs"]
        and all(data_identity(OWN / "sparse2" / ("source_" + Path(row["path"]).name))["sha256"] == row["sha256"]
            for row in current_contract["sources_and_inputs"][:5]))
    checks["current_versions_and_original_environment_path_recorded_without_old_binary_hash_claim"] = (
        current_contract["current_versions"]["numpy"] == "2.3.5"
        and current_contract["current_versions"]["pycolmap"] == "4.1.1"
        and Path(current_contract["same_existing_environment_path"]) == Path(sys.executable)
        and current_contract["old_dependency_binary_hashes_not_saved_not_invented"])
    checks["all_three_complete_public_PLY_pairs_and_both_complete_recovery_files_present"] = all(
        Path(row["file"]).stat().st_size == pair["complete_compared_bytes"] == 3591774
        and row["trace"]["original_binding_restored"] and row["trace"]["calls"][0]["points"] == 133020
        and row["trace"]["calls"][0]["status"] == "passed_complete_official_roundtrip"
        for pair in current["public_pairs"] for row in pair["rows"].values()) and all(
            row["complete_compared_bytes"] == 3591774 and row["trace"]["original_binding_restored"]
            and row["new_directory_after_real_cancellation"] for row in current["recoveries"].values())
    checks["both_SIGINT_while_complete_function_active_not_claimed_partial_file_write"] = all(
        row["sent"] and row["signal_during_active_writer"] and row["observer_joined"]
        and row["captured"] == "KeyboardInterrupt" and row["trace"]["calls"][0]["status"] == "failed"
        and row["trace"]["original_binding_restored"] for row in current["cancellations"].values())
    checks["first_auxiliary_denominator_error_and_source_preserved_charged_not_candidate_failure"] = (
        read(OWN / "sparse_lease.json")["returncode"] == 1
        and (OWN / "sparse/source_check_sparse_ply_v6_completion.py").exists()
        and "Public complete model denominator differs" in (OWN / "sparse.log").read_text(encoding="utf-8"))
    instances = {}
    old_fees = {}
    prefixes = ("sparse_ply_oracle_001_generic", "sparse_ply_oracle_002_generic", "sparse_ply_rejection_001",
                "sparse_ply_profile_001", "sparse_export_compare_001", "sparse_ply_profile_002")
    nodes = [(OUT / (name + "_lease.json"), OUT / (name + "_process.json")) for name in prefixes]
    nodes += [(OWN / (name + "_lease.json"), OWN / (name + "_GUARD.json")) for name in ("p0", "sparse", "sparse2")]
    for lease_path, guard_path in nodes:
        lease, guard = read(lease_path), read(guard_path)
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("An actual old or new owned lease did not close")
        instances[lease["guard_pid"], lease["guard_creation_time"]] = True
        instances[lease["child_pid"], lease["child_creation_time"]] = True
        for row in guard.get("owned_process_identities", []):
            instances[row["pid"], row["creation_time"]] = True
        if lease_path.parent == OUT:
            old_fees[lease_path.name] = lease["child_elapsed_seconds"]
    instances = [{"pid": pid, "creation_time": born, "ended": ended(pid, born)} for pid, born in instances]
    checks["all_owned_prior_nine_nodes_and_recorded_process_instances_closed"] = all(row["ended"] for row in instances)
    covered = sum(row["whole_seconds"] for pair in formal["pairs"] for row in pair["rows"].values())
    covered -= sum(pair["full_comparison_seconds_charged_each_side"] for pair in formal["pairs"])
    residual = old_fees["sparse_ply_profile_002_lease.json"] - covered
    extra_old = sum(value for key, value in old_fees.items() if key != "sparse_ply_profile_002_lease.json") + residual
    checks["all_old_failed_and_formal_unallocated_costs_kept_not_double_charged"] = residual >= 0 and len(old_fees) == 6
    checks["raw_whole_median_and_each_original_guard_kept"] = medians["columns"]["whole_seconds"] / medians["native"]["whole_seconds"] <= 1.01 and all(
        pair["rows"]["columns"]["whole_seconds"] / pair["rows"]["native"]["whole_seconds"] <= 1.1 for pair in formal["pairs"])
    checks["owned_output_budget_and_disk_headroom_kept"] = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 1024**3 and shutil.disk_usage(OWN).free >= 20 * 1024**3
    atomic_json(OWN / "SELF_REVIEW_V6_001.json", {"status": "separate_root_sparse_completion_evidence_passed" if all(checks.values()) else "failed",
        "checks": checks, "old_extra_complete_known_seconds": extra_old, "old_formal_unallocated_seconds": residual,
        "old_raw_medians": medians, "owned_prior_instances": instances,
        "review_is_independent_agent": False, "current_review_process_exit_not_yet_proved": True,
        "old_formal_full_output_comparisons_reused_not_newly_rerun": True,
        "actual_seconds_before_publication": time.perf_counter() - STARTED, "new_optimization_count": 0})
    print(json.dumps({"status": "root_passed" if all(checks.values()) else "failed", "checks": checks,
                      "seconds": time.perf_counter() - STARTED}), flush=True)
    if not all(checks.values()):
        raise AssertionError("Separate sparse completion self-review found a required failure")


if __name__ == "__main__":
    main()
