"""Close the preserved negative branch after native text and state review."""

import time

STARTED = time.perf_counter()

from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import re

import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001"
DESTINATION = OWN / "FINAL_CLOSURE_001.json"


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    if DESTINATION.exists():
        raise FileExistsError("Never overwrite the final negative closure")
    parent = read(DOC / "CURRENT_STATE.json")
    review = read(OWN / "NEGATIVE_REVIEW_001.json")
    screen = read(OWN / "s1/RESULT.json")
    failed = read(OWN / "FAILED_REVIEW_TOOL_001.json")
    texts = {}
    names = ("Q65_EVALUATION_GRAPH_TASK.md", "Q65_EVALUATION_GRAPH_RESULTS.md",
             "NEXT_NATIVE_CHECKPOINT_RESTORE_COST_BRIEF.md", "PLAN.md",
             "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md")
    for name in names:
        texts[name] = (DOC / name).read_text(encoding="utf-8")
    report = texts["Q65_EVALUATION_GRAPH_RESULTS.md"]
    local_links = [Path(value) for value in re.findall(r"\]\((C:/[^)]+)\)", report)]
    missing = [str(path) for path in local_links if path != DESTINATION and not path.exists()]
    leases = [read(OWN / name) for name in ("g0_lease.json", "s0_lease.json", "s1_lease.json")]
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with open(lock, "r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    checks = {
        "all_native_JSON_keys_unique": True,
        "authoritative_exact31_unique_target50_active": parent["additional_accepted_count"] == 31
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 31
            and parent["additional_target"] == 50 and parent["status"].startswith("active"),
        "Q65_negative0_not_in_accepted_mechanisms": parent["Q65_evaluation_graph_new_optimization_count"] == 0
            and screen["new_optimization_count"] == 0
            and not any(value.startswith("Q65") for value in parent["accepted_mechanisms"]),
        "R134_remains_exactly_one_previous_family": parent["R134_V4_new_optimization_count"] == 1
            and sum(value.startswith("R134") for value in parent["accepted_mechanisms"]) == 1,
        "root17_separate_self_review_not_independent_agent": len(review["checks"]) == 17
            and all(review["checks"].values()) and not review["root_review_is_independent_agent"],
        "actual_reviewer_instance_closed": closed(review["reviewer_instance"]),
        "all26_saved_Windows_instances_still_closed": len(review["Windows_instances"]) == 26
            and all(closed(value) for value in review["Windows_instances"]),
        "three_actual_Linux_instances_recorded_closed": len(review["Linux_instance_closure"]) == 3
            and all(value["instance_closed"] for value in review["Linux_instance_closure"]),
        "three_leases_released_and_native_OS_lock_reacquired": all("lease_released_utc" in value for value in leases),
        "raw_registered_negative_not_relabelled": screen["raw_stage_reduction"] < 0
            and not screen["checks"]["raw_stage_reduction_at_least3percent"]
            and not screen["formal_remaining_pairs_and_full_recovery_executed"],
        "original_numerical_and_JPEG_protection_passed": all(screen["comparison"][key]
            for key in ("all48_PSNR_MSE_and_metadata_exact", "all96_SSIM_original_tolerance",
                        "original_full_camera_and_photo_contract_exact", "complete_JPEG_bytes_exact"))
            and screen["comparison"]["maximum_absolute_SSIM_error"] == 0,
        "actual_checker_failure_and_original_source_preserved": failed["actual_tool_exit_code"] == 1
            and (OWN / "FAILED_REVIEW_SOURCE_001.py").exists()
            and (OWN / "s0/FAILED_DRIVER_SOURCE.py").exists(),
        "native_full_new_documents_read_and_local_links_exist": not missing
            and all(texts[name].strip() for name in names),
        "all_four_current_ledger_prefixes_describe_Q65_negative31": all("Q65" in texts[name].splitlines()[2]
            and "31/50" in texts[name].splitlines()[2]
            for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md")),
        "report_values_and_claim_limits_match_raw_receipts": all(value in report
            for value in ("0.514569751", "0.518063564", "256MiB", "31/50", "1.642643%", "不是", "没有运行另外两对"))
            and report.count("截图") == 1,
        "next_source_brief_records_existing_direct_restore_without_fake_cost": "没有CPU map_location" in
            texts["NEXT_NATIVE_CHECKPOINT_RESTORE_COST_BRIEF.md"]
            and "未实际restore/传输/峰值诊断" in texts["NEXT_NATIVE_CHECKPOINT_RESTORE_COST_BRIEF.md"],
    }
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    known = sum(value["child_elapsed_seconds"] for value in leases)
    receipt = {
        "status": "negative_branch_native_text_ledger_and_resource_closure_passed",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "new_optimization_count": 0,
        "parent_accepted": 31,
        "parent_target": 50,
        "parent_remaining": 19,
        "parent_goal_status": "active",
        "root_review_is_independent_agent": False,
        "native_text_documents": [{"path": str(DOC / name), "lines": len(value.splitlines()),
            "characters": len(value)} for name, value in texts.items()],
        "native_local_report_links": [str(path) for path in local_links],
        "reviewer_actual_tool": {"chunk_id": "398a67", "exit_code": 0, "wall_seconds": 0.9915319,
            "instance": review["reviewer_instance"], "instance_now_closed": True},
        "failed_reviewer_actual_tool": failed,
        "known_three_lease_execution_seconds": known,
        "known_new_lease_and_two_root_review_tool_seconds_before_final_publication_lower_bound":
            known + failed["actual_tool_wall_seconds"] + 0.9915319,
        "unmeasured_costs": "old_Q65_full_envelope_text_research_and_non_native_tool_time_not_zero_not_used_to_accept",
        "raw_stage_seconds": review["recalculated_stage_seconds"],
        "raw_stage_reduction": screen["raw_stage_reduction"],
        "not_executed": ["remaining_two_formal_pairs", "graph_full_entry_error_cancel_recovery",
            "public360_graph_component", "training", "mapper", "full_LIO", "GT", "production"],
        "finalizer_instance": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "actual_finalizer_seconds_before_publication": time.perf_counter() - STARTED,
        "finalizer_post_tool_exit_receipt": str(OWN / "FINAL_TOOL_EXIT_001.json"),
        "document_review_method": "native_text_and_strict_JSON_no_document_hash_or_screenshot",
    }
    with open(DESTINATION, "x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if not all(path.exists() for path in local_links):
        raise AssertionError("A final local report link is missing")
    print(json.dumps({"status": receipt["status"], "checks": checks,
        "parent_accepted": 31, "new_optimization_count": 0,
        "finalizer_instance": receipt["finalizer_instance"]}), flush=True)


if __name__ == "__main__":
    main()
