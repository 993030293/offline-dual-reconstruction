"""Publish scoped eligibility only after the separate root reviewer actually exits."""

import json
from pathlib import Path
import shutil
import time

import native_sfm_match_shards as native
from check_native_sfm_match_shards import closed
from review_native_sfm_match_shards import costs, history_rows, scores


def main():
    started = time.perf_counter()
    review = native.read(native.OUT / "root_review0/REVIEW.json")
    guard, births = history_rows("root_review0")
    if not all(review["checks"].values()) or not all(closed(row) for row in review["closed_previous_births"]):
        raise AssertionError("The prior actual separate root review is not complete")
    for row in review["sources"] + review["frozen_identities"]:
        if native.data_identity(row["path"]) != row:
            raise AssertionError("A reviewed source or scientific input changed")
    fees = costs(guard["elapsed_seconds"] + time.perf_counter() - started)
    measured = scores(fees["total_known_execution_plus_allowance_seconds"])
    checks = {
        "all18_separate_root_checks_and_actual_reviewer_exit0": len(review["checks"]) == 18 and all(review["checks"].values()) and guard["returncode"] == 0,
        "every_review_and_prior_exact_process_birth_closed": all(closed(row) for row in births + review["closed_previous_births"]),
        "all_frozen_current_sources_and_input_values_identical_to_reviewed_version": all(native.data_identity(row["path"]) == row for row in review["sources"] + review["frozen_identities"]),
        "full_matching_net_speed_V6_at_least2percent_with_all_known_fees": measured["net_complete_stage_reduction"] >= .02,
        "raw_and_fee_whole_median1point01_and_every_pair1point1_guards": measured["raw_whole_median_ratio"] <= 1.01 and measured["fee_adjusted_whole_median_ratio"] <= 1.01 and all(row["raw_whole_ratio"] <= 1.1 and row["fee_adjusted_whole_ratio"] <= 1.1 for row in measured["pairs"]),
        "registered_total_candidate8GiB_and_disk20GiB_reserve": sum(path.stat().st_size for path in native.OUT.rglob("*") if path.is_file()) <= 8 * 1024 ** 3 and shutil.disk_usage(str(native.ROOT)).free >= 20 * 1024 ** 3,
        "known_fees_and_registered15_seconds_preserved_not_total_research_ROI": fees["preregistered_metadata_allowance_seconds"] == 10 and fees["preregistered_post_allowance_seconds"] == 5 and fees["unmeasured_research_work_time_not_zero_or_total_ROI"],
        "parent35_single_distinct_candidate_no_duplicate_acceptance": native.read(Path(__file__).resolve().parents[2] / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")["additional_accepted_count"] == 35 and not (native.OUT / "ACCEPTANCE_V6_001.json").exists(),
    }
    result = {"status": "scoped_matching_speed_eligible_pending_external_actual_publisher_exit", "checks": checks,
        "mechanism": "R165_COMPLETE_NATIVE_SFM_MATCHING_ISOLATED_SHARDS_FULL_COLUMN_MERGE",
        "scores": measured, "known_costs": fees, "source": native.data_identity(Path(__file__)),
        "root_review": str(native.OUT / "root_review0/REVIEW.json"), "review_guard": str(native.OUT / "root_review0_GUARD.json"),
        "closed_review_births": births, "closed_prior_births": review["closed_previous_births"],
        "original_input_quality_errors_cancel_public_and_full_recovery_preserved": True,
        "optional_independent_entry_not_production_default": True,
        "whole_claim_only_complete_matching_not_mapping_training_or_GT": True,
        "eligible_single_new_optimization": 1, "parent_count_before_actual_publisher_exit": 35,
        "parent_count_if_actual_publisher_exit0_and_exact_births_closed": 36,
        "actual_publisher_exit_must_be_verified_externally_before_parent_count": True}
    native.save(native.OUT / "ACCEPTANCE_V6_001.json", result)
    print(json.dumps({"checks": checks, "net_matching_reduction": measured["net_complete_stage_reduction"],
        "eligible_one_pending_actual_publisher_exit": 1}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The scoped native matching eligibility failed")


if __name__ == "__main__":
    main()
