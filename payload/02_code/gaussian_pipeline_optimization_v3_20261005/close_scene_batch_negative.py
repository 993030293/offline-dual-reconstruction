"""Publish the preserved negative Scene result with text and cost consistency."""

import re
from pathlib import Path
import shutil
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import CODE, ROOT, OUT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    final = OUT / "FINAL_NEGATIVE_CLOSURE_001.json"
    if final.exists():
        raise FileExistsError(final)
    doc = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    state = read(doc / "CURRENT_STATE.json")
    formal, review, regression = (read(OUT / p) for p in ("c1/RESULT.json", "r0/RESULT.json", "rc0/RESULT.json"))
    native, batch = (formal["raw_complete_stage_medians"][mode] for mode in ("native", "batch"))
    measured_additions = {"actual_root_review_spawn_to_exit_seconds": .7906426,
                          "actual_future_cost_writer_regression_spawn_to_exit_seconds": .2572615,
                          "actual_failed_publisher_link_count_seconds": .2945255,
                          "actual_failed_publisher_source_archive_seconds": .1550052,
                          "actual_native_text_link_listing_seconds": .1285396}
    known = review["known_fee_lower_bound_before_review_exit_seconds"] + sum(measured_additions.values())
    bound = equal_fee_cost_bound(native, batch, known)
    sources_changed = [row["path"] for row in formal["sources"] if data_identity(row["path"]) != row]
    archived_exact = all(data_identity(OUT / "c1/source_snapshot" / (f"{i:03d}_" + Path(row["path"]).name))["sha256"] == row["sha256"]
                         for i, row in enumerate(formal["sources"]))
    report = (doc / "R152_SCENE_BATCH_RESULTS.md").read_text(encoding="utf-8")
    brief = (doc / "NEXT_SCENE_FINGERPRINT_COST_BRIEF.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    pending = {str(final).replace("\\", "/"), str(OUT / "FINAL_POST_EXIT_001.json").replace("\\", "/")}
    lease_checks = {}
    for label, expected in (("p0", 0), ("g0", 0), ("g1", 1), ("g2", 0), ("c0_first", 1), ("c1_first", 0)):
        lease = read(OUT / (label + "_lease.json"))
        lease_checks[label] = lease["status"] == "child_exited" and lease["returncode"] == expected and bool(lease["lease_released_utc"])
    head_checks = {name: "R152" in "\n".join((doc / name).read_text(encoding="utf-8").splitlines()[:4])
                         and "28/50" in "\n".join((doc / name).read_text(encoding="utf-8").splitlines()[:4])
                   for name in ("PLAN.md", "OPTIMIZE_CHECKLIST.md", "CANDIDATE_BOARD.md", "GENERALIZATION_PROTOCOL.md")}
    checks = {
        "parent28_unique28_receipts_no_R152": state["additional_accepted_count"] == 28
            and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 28
            and all((ROOT / p).is_file() for p in state["acceptance_receipts"])
            and not any(name.startswith("R152") for name in state["accepted_mechanisms"]),
        "uniform3_whole_gates_and_active_scope_kept": state["V5_minimum_complete_stage_net_reduction"] == .03
            and state["V5_maximum_whole_median_ratio"] == 1.01 and state["V5_maximum_each_whole_ratio"] == 1.1
            and state["status"].startswith("active"),
        "separate_root16_and_future_writer9_checks_passed": len(review["checks"]) == 16 and all(review["checks"].values())
            and len(regression["checks"]) == 9 and all(regression["checks"].values()),
        "original_c1_receipt_still_exact_before_writer_fix": data_identity(OUT / "c1/RESULT.json") == regression["original_C1_identity"],
        "all_original_c1_executed_source_archives_exact": archived_exact,
        "only_intentional_future_writer_changed_current_source": sources_changed == [str(CODE / "run_batched_camera_scene_campaign.py")],
        "raw_complete_numbers_and_negative_state_agree": native == state["R152_complete_stage_native_seconds"]
            and batch == state["R152_complete_stage_batch_seconds"] and 1 - batch / native == state["R152_complete_stage_raw_reduction"],
        "negative_gain_correct_sample_upper_zero_and_not_acceptance": bound["conservative_sample_reduction_upper_bound"]
            == state["R152_corrected_sample_net_reduction_upper_bound"] == 0. < .03,
        "all_six_expected_leases_exited_and_released": all(lease_checks.values()),
        "actual_saved52_Windows_and_new_regression_instance_closed": len(review["Windows_instances"]) == 52
            and all(windows_closed(row) for row in review["Windows_instances"]) and windows_closed(regression["owner"]),
        "prior_actual_Linux8_checks_and_no_new_WSL_job_since_review": len(review["Linux_instances"]) == 8
            and all(not row["actual_current_alive"] for row in review["Linux_instances"]),
        "no_remaining_formal_or_recovery_output_after_cost_failure": len(formal["pairs"]) == 1
            and not (OUT / "c1/t1").exists() and not (OUT / "c1/t2").exists() and not (OUT / "v0").exists(),
        "native_report_link_targets_exist_except_current_pending_publishers": len(links) == 11
            and all(Path(link).is_file() or link in pending for link in links),
        "native_text_reports_match_scope_and_metric_limits": "33.234719→35.570597" in report
            and "4.910796926997136" in brief and "未分析出唯一性能根因" in report
            and "盲测" in report and "尚未产生新运行目录" in brief,
        "four_frontier_heads_reflect_terminal_R152_and_28": all(head_checks.values()),
        "failed_link_count_publisher_source_and_actual_fee_preserved": read(OUT / "FAILED_PUBLICATION_001.json")["actual_original_text_link_count"] == 11
            and "len(links) == 13" in (OUT / "FAILED_CLOSURE_SOURCE_001.py").read_text(encoding="utf-8"),
    }
    if not all(checks.values()):
        raise AssertionError("Final Scene negative text or cost checks failed: " + repr(checks))
    snapshot = OUT / "final_source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for source in (Path(__file__), CODE / "equal_fee_cost_bound.py", CODE / "run_batched_camera_scene_campaign.py"):
        shutil.copyfile(source, snapshot / source.name)
    atomic_json(final, {"status": "closed_negative_one_complete1016_pair_slower7point028427percent_parent28_active_new0",
        "checks": checks, "complete_stage_seconds": formal["raw_complete_stage_medians"],
        "raw_stage_reduction": formal["raw_stage_reduction"], "corrected_sample_cost_bound": bound,
        "known_fee_lower_bound_before_publisher_exit_seconds": known, "actual_measured_post_review_additions": measured_additions,
        "old_failures_and_original_C1_kept": True, "executed_C1_and_future_writer_source_versions_distinguished": True,
        "text_review": {"method": "original_UTF8_text_and_changed_lines_no_images_or_doc_hash_acceptance",
            "reports": [str(doc / "R152_SCENE_BATCH_RESULTS.md"), str(doc / "NEXT_SCENE_FINGERPRINT_COST_BRIEF.md")],
            "actual_report_links_checked": len(links), "pending_publishers": sorted(pending), "frontier_head_checks": head_checks},
        "actual_original_windows_instances_closed": 52, "prior_actual_linux_instances_closed": 8,
        "new_regression_instance_actually_closed": regression["owner"], "all_six_original_leases_released": lease_checks,
        "own_process": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "source_identity": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started,
        "post_exit_check_pending": True, "unmetered_text_research_and_publish_fees_not_assumed_zero": True,
        "formal3_errors_recovery_GT_new_objects_training_production": "not_run",
        "next_action": "explore_distinct_by_field_pixel_state_cost_attribution_not_implemented_algorithm",
        "new_optimization_count": 0, "parent_count": 28, "goal_remains_active": True})
    print("Final negative Scene closure passed: 16 consistency checks, new 0, parent 28/50 active")


if __name__ == "__main__":
    main()
