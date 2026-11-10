"""Publish one scoped work mechanism after the actual review writer exits."""

import json
from pathlib import Path
import re
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "oup019/R171"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ended(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    review = read(OUT / "review1/RESULT.json")
    lease = read(OUT.parent / "REVIEW_V2_LEASE.json")
    state_path = DOC / "CURRENT_STATE.json"
    state = read(state_path)
    report = (DOC / "R171_STRICT_SIFT_ENCODING_RESULTS.md").read_text(encoding="utf-8")
    task = (DOC / "R171_STRICT_SIFT_ENCODING_TASK.md").read_text(encoding="utf-8")
    time_fee = review["known_related_all_measured_fee_plus_registered120_seconds"] + lease["child_elapsed_seconds"]
    charged = {k: v + time_fee/6 for k, v in review["raw_whole_medians_seconds"].items()}
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    mechanism = "R171_STRICT_SIFT_CPU_SINGLE_SCAN_ARRAY_OUTPUT_WORK"
    acceptance = OUT / "ACCEPTANCE_WORK_V7_001.json"
    checks = {
        "all17_separate_root_checks_passed": len(review["checks"]) == 17 and all(review["checks"].values()),
        "actual_review_writer_and_both_lease_processes_ended": ended(review["writer"]) and all(
            ended({"pid": lease[p+"_pid"], "creation_time": lease[p+"_creation_time"]}) for p in ("guard", "child")),
        "actual_review_v2_exit0_lease_released": lease["returncode"] == 0 and "lease_released_utc" in lease,
        "charged_whole_within_original_V7": charged["native"] <= 1.03*charged["raw"],
        "net_work_original_registered_threshold": review["net_array_output_reduction"] >= .1,
        "original38_target40_unique_source_and_receipt_pairs": state["additional_target"] == 40
            and state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 38
            and len(state["acceptance_receipts"]) == len(set(state["acceptance_receipts"])) == 38
            and mechanism not in state["accepted_mechanisms"] and not acceptance.exists(),
        "native_report_original_numbers_failed_branch_scope_and_links": "23.485606562%" in report
            and "review0" in report and "非独立代理" in report and "没有执行新训练" in report
            and "六个明确" in task and bool(links) and all(Path(p).exists() for p in links),
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    result = {
        "status": "accepted_scoped_work_V7_not_speed_or_production", "new_optimization_count": 1,
        "mechanism": mechanism, "checks": checks, "root_review": str(OUT / "review1/RESULT.json"),
        "report": str(DOC / "R171_STRICT_SIFT_ENCODING_RESULTS.md"),
        "complete_extra_array_fee_bytes": review["complete_extra_array_fee_bytes"],
        "net_registered_array_output_reduction": review["net_array_output_reduction"],
        "raw_whole_medians_seconds": review["raw_whole_medians_seconds"],
        "known_actual_fee_seconds_plus_registered120_seconds": time_fee,
        "charged_whole_medians_seconds": charged,
        "registered120_seconds_includes_remaining_text_publication_and_post_exit_execution": True,
        "writer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "publication_seconds_before_write": time.perf_counter()-started,
        "independent_agent_review": False, "single_chat_no_subagents": True,
        "quality_GT_training_unknown_object_full_pipeline_improvement_not_claimed": True,
    }
    atomic_json(acceptance, result)
    state["accepted_mechanisms"].append(mechanism)
    state["acceptance_receipts"].append(acceptance.relative_to(ROOT).as_posix())
    state.update(additional_accepted_count=39,
        R171_status=result["status"], R171_new_optimization_count=1,
        R171_acceptance=acceptance.relative_to(ROOT).as_posix(),
        R171_report=(DOC / "R171_STRICT_SIFT_ENCODING_RESULTS.md").relative_to(ROOT).as_posix(),
        R169_R170_final_text_closure="04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/FINAL_TEXT_STATE_CLOSURE_002.json",
        R163_current_status="paused_candidate0_unknown_GPU_tail_cost_no_forced_acceptance",
        current_next_action="target40_effective39_remaining1_R171_work_accepted_all17_root_and_actual_exit_closed_R172_SSIM_elementwise_fusion_evaluation_only_next_single_chat",
        current_next_brief=(DOC / "NEXT_SSIM_ELEMENTWISE_FUSION_020_BRIEF.md").relative_to(ROOT).as_posix())
    atomic_json(state_path, state)
    print(json.dumps({"status": result["status"], "count":39, "target":40, "checks":checks,
        "net_work":result["net_registered_array_output_reduction"], "charged_seconds":charged}), flush=True)


if __name__ == "__main__":
    main()
