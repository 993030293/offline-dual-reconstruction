"""Compute complete recorded old costs before funding missing candidate checks."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "v6_residual_cost_003"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bounds(native, candidate, fees):
    maximum = ((native - candidate) / .02 - native) * 6
    return {"raw_stage_seconds": {"native": native, "candidate": candidate}, "known_extra_seconds": fees,
        "maximum_total_extra_at_net2_seconds": maximum, "remaining_extra_budget_upper_seconds": maximum - fees,
        "complete_stage_reduction_upper_bound": 1 - (candidate + fees / 6) / (native + fees / 6)}


def main():
    output = OWN / "p0"
    output.mkdir(parents=True, exist_ok=False)
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if parent["additional_accepted_count"] != 33:
        raise AssertionError("The parent changed before source and cost audit")
    first = read(OUT / "v5_cost_rank_001/CLOSURE_001.json")["candidates"]["R110"]
    json_pair = first["native_candidate_stage_seconds"]
    sparse = read(OUT / "sparse_ply_oracle_001/campaign_002/RESULT.json")
    fees = {}
    prefixes = ("sparse_ply_oracle_001_generic", "sparse_ply_oracle_002_generic", "sparse_ply_rejection_001",
        "sparse_ply_profile_001", "sparse_export_compare_001", "sparse_ply_profile_002")
    for prefix in prefixes:
        row = read(OUT / (prefix + "_lease.json"))
        if row["status"] != "child_exited" or not row.get("lease_released_utc"):
            raise AssertionError("An old required sparse node did not close")
        fees[prefix] = row["child_elapsed_seconds"]
    covered = sum(row["whole_seconds"] for pair in sparse["pairs"] for row in pair["rows"].values())
    covered -= sum(pair["full_comparison_seconds_charged_each_side"] for pair in sparse["pairs"])
    residue = fees["sparse_ply_profile_002"] - covered
    if residue < 0:
        raise AssertionError("Formal components cannot exceed their actual envelope")
    sparse_extra = sum(value for key, value in fees.items() if key != "sparse_ply_profile_002") + residue
    contract = read(OUT / "sparse_ply_oracle_001/campaign_002/CONTRACT.json")
    if any(data_identity(row["path"]) != row for row in contract["sources_and_inputs"]):
        raise AssertionError("A full sparse source or input changed")
    snapshot = OUT / "sparse_ply_oracle_001/campaign_002/source_snapshot"
    copies = sorted(snapshot.iterdir())
    for saved, row in zip(copies, contract["sources_and_inputs"][:len(copies)], strict=True):
        if data_identity(saved)["sha256"] != row["sha256"]:
            raise AssertionError("An actual sparse source copy differs")
    gpu = read(OUT / "gpu_origin_dependency_001/root_profile_001/PROFILE_RESULT.json")
    old_gpu = read(OUT / "gpu_origin_dependency_001/FINAL_SCREEN_PUBLICATION_001.json")
    old_R115 = []
    for path in sorted((OUT / "origin_dependency_001").glob("LEASE_*.json")):
        row = read(path)
        if row["status"] != "child_exited" or not row.get("lease_released_utc"):
            raise AssertionError("An old same-family GPU node remains unclosed")
        old_R115.append({"path": str(path), "seconds": row["child_elapsed_seconds"], "returncode": row["returncode"]})
    if len(old_R115) != 7:
        raise AssertionError("The full old same-family fee denominator changed")
    gpu_extra = old_gpu["all_R118_actual_lease_child_fees_including_this_screen"] + old_gpu["all_three_checker_child_fees"]
    gpu_extra += sum(row["seconds"] for row in old_R115)
    common = (time.perf_counter() - STARTED + 3.) / 3
    candidates = {
        "R110": bounds(*json_pair, first["known_fee_lower_bound_seconds"] + common),
        "R111": bounds(sparse["medians"]["native"]["phase_seconds"], sparse["medians"]["columns"]["phase_seconds"], sparse_extra + common),
        "R118": bounds(gpu["rawCompleteConsumerSeconds"]["native"], gpu["rawCompleteConsumerSeconds"]["candidate"], gpu_extra + common),
    }
    checks = {"current_parent33_no_candidates_counted": parent["additional_target"] == 50 and parent["status"].startswith("active"),
        "full_sparse_current_source_input_and_snapshot_identity_kept": True,
        "sparse_formal_work_charged_once_and_all_six_old_nodes_included": residue >= 0 and len(fees) == 6,
        "old_same_family_R115_seven_nodes_no_waived_failure_cost": len(old_R115) == 7,
        "old_formal_and_failed_statuses_preserved": sparse["status"] == "complete_export_cost_guard_failed",
        "no_saved_or_current_quality_relabelled_as_acceptance": True}
    shutil.copyfile(__file__, output / Path(__file__).name)
    atomic_json(output / "SPARSE_CURRENT_IDENTITIES.json", contract["sources_and_inputs"])
    atomic_json(output / "RESULT.json", {"status": "old_complete_cost_screen_current_source_verified",
        "checks": checks, "candidates": candidates, "sparse_all_old_fees": fees,
        "sparse_actual_formal_covered_once_seconds": covered, "sparse_actual_formal_unallocated_seconds": residue,
        "R118_same_family_old_R115_actual_fee_rows": old_R115,
        "common_current_audit_so_far_and_registered3tail_seconds_each": common,
        "all_metrics_are_upper_bounds_not_acceptance": True, "new_optimization_count": 0,
        "actual_seconds_before_publication": time.perf_counter() - STARTED})
    print(json.dumps({key: {"upper": row["complete_stage_reduction_upper_bound"], "remaining": row["remaining_extra_budget_upper_seconds"]}
        for key, row in candidates.items()}), flush=True)


if __name__ == "__main__":
    main()
