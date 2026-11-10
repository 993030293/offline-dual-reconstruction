"""Prepare a scoped receipt without counting before the actual publisher exit."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-command-seconds", type=float, required=True)
    parser.add_argument("--review-command-chunk", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = BASE / "PUBLICATION_001.json"
    if output.exists() or (BASE / "ACCEPTANCE_V5_001.json").exists():
        raise FileExistsError("The scoped publisher needs a fresh receipt path")
    parent = read(DOC / "CURRENT_STATE.json")
    review = read(BASE / "r0/RESULT.json")
    formal = read(BASE / "c0/RESULT.json")
    if args.review_command_seconds < review["internal_before_result_seconds"]:
        raise ValueError("Actual review exit cannot precede its recorded internal window")
    if len(review["checks"]) != 17 or not all(review["checks"].values()):
        raise ValueError("The separately executed root review did not pass")
    if (parent["additional_accepted_count"] != 28 or
            len(parent["accepted_mechanisms"]) != 28 or
            len(set(parent["accepted_mechanisms"])) != 28 or
            len(parent["acceptance_receipts"]) != 28 or parent["R153_algorithm_accepted"] or
            any(name.startswith("R153") for name in parent["accepted_mechanisms"])):
        raise ValueError("The protected parent must still be unpromoted at 28")
    if (parent["V5_minimum_complete_stage_net_reduction"] != .03 or
            parent["V5_maximum_whole_median_ratio"] != 1.01 or
            parent["V5_maximum_each_whole_ratio"] != 1.1):
        raise ValueError("The uniform current protocol changed")
    if formal["sources"] != [data_identity(row["path"]) for row in formal["sources"]]:
        raise ValueError("An executed complete algorithm source changed")
    if review["source"] != data_identity(review["source"]["path"]):
        raise ValueError("The executed root review source changed")
    reports = [DOC / name for name in (
        "R153_FUSED_FINGERPRINT_RESULTS.md", "NEXT_Q68_FUSED_FINGERPRINT_COMBINATION_BRIEF.md")]
    for report in reports:
        if not report.read_text(encoding="utf-8").strip():
            raise ValueError("A report is empty; text review remains a separate root action")
    native, fused = (formal["raw_complete_stage_medians"][key] for key in ("original", "fused"))
    fee = review["known_fee_plus_registered_reserve_before_review_exit_seconds"] + args.review_command_seconds
    bound = equal_fee_cost_bound(native, fused, fee)
    if (bound["signed_reduction_at_known_lower_fee"] < .03 or fused / native > 1.01 or
            max(formal["whole_ratios"]) > 1.1):
        raise ValueError("Final review cost or an unchanged whole gate failed")
    archive = BASE / "FINALIZER_SOURCES"
    archive.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), archive / Path(__file__).name)
    process = psutil.Process()
    atomic_json(output, {
        "status": "all_scoped_checks_passed_actual_publisher_exit_and_post_cost_pending0",
        "new_optimization_count": 0, "parent_count": 28,
        "own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "actual_root_review_command": {"returncode": 0, "seconds": args.review_command_seconds,
            "tool_chunk": args.review_command_chunk},
        "known_fee_plus_registered10_reserve_before_publisher_exit_seconds": fee,
        "cost_bound_under_declared_execution_fee_model": bound,
        "raw_complete_stage_medians": formal["raw_complete_stage_medians"],
        "raw_stage_reduction": formal["raw_stage_reduction"],
        "review": str(BASE / "r0/RESULT.json"), "report_text_paths": [str(p) for p in reports],
        "source": data_identity(Path(__file__)),
        "internal_before_result_seconds": time.perf_counter() - started,
        "unknown_text_research_not_assumed_zero_or_total_ROI": True,
        "production_default_changed": False, "goal_remains_active": True,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    print({"prepared_not_counted": True, "parent": 28,
        "net_at_recorded_fee_plus_reserve": bound["signed_reduction_at_known_lower_fee"]})


if __name__ == "__main__":
    main()
