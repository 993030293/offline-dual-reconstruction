"""Regress the future cost writer against the preserved slower full Scene pair."""

import math
from pathlib import Path
import shutil
import time

import psutil

from run_batched_camera_scene_campaign import complete_cost_screen
from review_scene_batch_negative import read
from scene_transform_cost_node import CODE, OUT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = OUT / "rc0"
    output.mkdir(exist_ok=False)
    original = OUT / "c1/RESULT.json"
    original_identity = data_identity(original)
    actual = read(original)
    medians = actual["raw_complete_stage_medians"]
    fee = actual["initial_known_family_fee_lower_bound_seconds"]
    corrected = complete_cost_screen(medians, fee)
    larger = complete_cost_screen(medians, fee + 1e8)
    positive = complete_cost_screen({"native": 40., "batch": 36.}, 0.)
    positive_fee = complete_cost_screen({"native": 40., "batch": 36.}, 120.)
    rejected = 0
    for invalid in (-1., math.inf, math.nan):
        try:
            complete_cost_screen(medians, invalid)
        except ValueError:
            rejected += 1
    changed = [row["path"] for row in actual["sources"] if data_identity(row["path"]) != row]
    saved = OUT / "r0/source_snapshot/run_batched_camera_scene_campaign.py"
    old_writer = next(row for row in actual["sources"] if Path(row["path"]).name == saved.name)
    checks = {
        "real_c1_cost_denominators_and_raw_values_kept": corrected["complete_stage_medians_lower_bound_seconds"]
            == actual["complete_stage_medians_lower_bound_seconds"],
        "old_signed_value_preserved_as_signed_not_upper": corrected["complete_stage_signed_reduction_at_known_lower_fee"]
            == actual["complete_stage_reduction_upper_bound"] < 0.,
        "negative_gain_true_upper_zero_still_below_uniform3": corrected["complete_stage_reduction_upper_bound"] == 0. < .03,
        "additional_equal_fee_negative_gain_moves_toward_zero": larger["complete_stage_reduction_upper_bound"] == 0.
            and larger["complete_stage_signed_reduction_at_known_lower_fee"] > corrected["complete_stage_signed_reduction_at_known_lower_fee"],
        "positive_and_equal_cost_cases_keep_conservative_bound": positive_fee["complete_stage_reduction_upper_bound"]
            < positive["complete_stage_reduction_upper_bound"] and complete_cost_screen({"native": 1., "batch": 1.}, 100.)["complete_stage_reduction_upper_bound"] == 0.,
        "negative_nonfinite_fees_are_explicit_errors": rejected == 3,
        "only_future_campaign_writer_changed_after_review": changed == [str(CODE / "run_batched_camera_scene_campaign.py")],
        "executed_original_writer_archive_matches_C1_source": data_identity(saved)["sha256"] == old_writer["sha256"],
        "original_C1_receipt_not_rewritten": data_identity(original) == original_identity,
    }
    if not all(checks.values()):
        raise AssertionError("The cost report correction failed: " + repr(checks))
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), CODE / "run_batched_camera_scene_campaign.py", CODE / "equal_fee_cost_bound.py"):
        shutil.copyfile(source, snapshot / source.name)
    atomic_json(output / "RESULT.json", {"status": "future_writer_label_regression_passed_original_full_run_preserved_new0",
        "checks": checks, "corrected_real_C1_screen": corrected, "original_C1_identity": original_identity,
        "changed_after_executed_C1_source": changed, "source_identity": data_identity(Path(__file__)),
        "owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "before_result_seconds": time.perf_counter() - started, "full_Scene_rerun": False,
        "new_optimization_count": 0})
    print("Cost writer regression passed: 9 checks, original full receipt preserved, new 0")


if __name__ == "__main__":
    main()
