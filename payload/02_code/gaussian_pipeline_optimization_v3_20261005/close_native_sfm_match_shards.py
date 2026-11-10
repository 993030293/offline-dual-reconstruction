"""Close actual publisher history and final costs before an external count update."""

import json
from pathlib import Path
import shutil
import time

import native_sfm_match_shards as native
from check_native_sfm_match_shards import closed
from review_native_sfm_match_shards import costs, history_rows, scores


def main():
    started = time.perf_counter()
    acceptance = native.read(native.OUT / "ACCEPTANCE_V6_001.json")
    guard, births = history_rows("publisher0")
    root_guard, root_births = history_rows("root_review0")
    if not all(acceptance["checks"].values()):
        raise AssertionError("The closed scoped eligibility checks failed")
    if native.data_identity(acceptance["source"]["path"]) != acceptance["source"]:
        raise AssertionError("The actual publisher source changed")
    fees = costs(root_guard["elapsed_seconds"] + guard["elapsed_seconds"] + .2087339 + time.perf_counter() - started)
    measured = scores(fees["total_known_execution_plus_allowance_seconds"])
    text = (native.ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R165_NATIVE_MATCH_SHARDS_RESULTS.md").read_text(encoding="utf-8-sig")
    rows = births + root_births + acceptance["closed_prior_births"]
    checks = {
        "actual_publisher_and_separate_root_guard_exit0_and_leases_released": guard["returncode"] == root_guard["returncode"] == 0,
        "every_previous_exact_birth_identity_ended": all(closed(row) for row in rows),
        "all_original_acceptance_quality_recovery_and_whole_checks_preserved": all(acceptance["checks"].values()),
        "final_known_guard_reader_review_publish_post_and_reserved_fees_retained": fees["additional_actual_review_and_post_seconds"] >= root_guard["elapsed_seconds"] + guard["elapsed_seconds"],
        "final_matching_net_at_least_uniform2percent": measured["net_complete_stage_reduction"] >= .02,
        "raw_and_adjusted_whole_medians_and_every_pair_guards": measured["raw_whole_median_ratio"] <= 1.01 and measured["fee_adjusted_whole_median_ratio"] <= 1.01 and all(row["raw_whole_ratio"] <= 1.1 and row["fee_adjusted_whole_ratio"] <= 1.1 for row in measured["pairs"]),
        "current_candidate_disk_and_GPU_reserve": sum(path.stat().st_size for path in native.OUT.rglob("*") if path.is_file()) <= 8 * 1024 ** 3 and shutil.disk_usage(str(native.ROOT)).free >= 20 * 1024 ** 3 and bool(native.gpu()),
        "direct_report_text_scopes_preserved_no_GT_training_or_production_claim": all(value in text for value in ("未部署生产流程", "没有运行mapper", "未重新运行公开native matcher", "不报告整套SfM")),
    }
    value = {"status": "all_previous_actual_exits_and_scoped_acceptance_passed_external_post_exit_check_required",
        "checks": checks, "known_costs": fees, "scores": measured, "closed_previous_births": rows,
        "actual_previous_tool_receipts": {"root_review": "69c347_exit0", "publisher": "e3bf07_exit0", "external_publisher_birth_check": "0403c3_exit0"},
        "source": native.data_identity(Path(__file__)), "candidate_output_bytes": sum(path.stat().st_size for path in native.OUT.rglob("*") if path.is_file()),
        "eligible_new_mechanisms_after_current_actual_post_exit0": 1, "parent_if_current_post_exit0": 36,
        "goal_still_active_target50_remaining14_after_actual_exit": True,
        "post_fee_includes_actual_work_until_write_plus_registered5_seconds_for_remaining_metadata_and_external_closure": True}
    native.save(native.OUT / "FINAL_POST_EXIT_V6_001.json", value)
    print(json.dumps({"checks": checks, "final_net": measured["net_complete_stage_reduction"], "parent_after_actual_exit": 36}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The actual native matching post closure failed")


if __name__ == "__main__":
    main()
