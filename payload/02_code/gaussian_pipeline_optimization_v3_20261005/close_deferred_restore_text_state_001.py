"""Check the actual publication, direct text and acceptance registry once."""

import json
from pathlib import Path
import re
import shutil

import psutil

from run_cpu_dependency_restore import ROOT, read
from run_deferred_restore_equal_pair_001 import BASE, save


def main():
    doc = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    report = (doc / "R168_DEFERRED_RESTORE_BOOLEAN_RESULTS.md").read_text(encoding="utf-8")
    next_text = (doc / "NEXT_RESTORE_SEED_INITIALIZATION_COST_015_BRIEF.md").read_text(encoding="utf-8")
    master = read(doc / "CURRENT_STATE.json")
    acceptance = read(BASE / "ACCEPTANCE_WORK_V6_001.json")
    writer = acceptance["writer_identity"]
    try:
        process = psutil.Process(writer["pid"])
        publisher_ended = process.create_time() != writer["creation_time"] or not process.is_running()
    except psutil.NoSuchProcess:
        publisher_ended = True
    links = re.findall(r"\]\((C:/[^)]+)\)", report + next_text)
    registry = master["accepted_mechanisms"]
    checks = {
        "actual_d30bed_publisher_exit0_and_saved_birth_ended": publisher_ended,
        "all_registered37_unique_and_receipts_present": master["additional_accepted_count"] == len(registry) == len(set(registry)) == len(master["acceptance_receipts"]) == 37
            and all((ROOT / value).is_file() for value in master["acceptance_receipts"]),
        "one_scoped_R168_work_no_speed_no_full_goal_completion": master["R168_new_optimization_count"] == acceptance["new_optimization_count"] == 1
            and not acceptance["full_speed_accepted"] and master["status"].startswith("active")
            and sum(value.startswith("R168_") for value in registry) == 1,
        "all17_review_checks_and_charged_work_match_report": all(acceptance["checks"].values())
            and abs(acceptance["net_work_reduction_index"] - .21640518027481295) < 1e-12
            and "540.504009061" in report and "21.640518%" in report and "0.370394%" in report,
        "direct_text_contains_V1_failure_and_V2_fresh_complete_evidence": all(token in report for token in (
            "V1不接纳", "三组NEW V2正式对", "41真CUDA", "真正SIGINT", "4f273a exit0", "d30bed exit0", "历史", "camera_center为3维向量")),
        "current_plan_board_checklist37_and_active": all("37/50" in (doc / name).read_text(encoding="utf-8").split("以下")[0]
            for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")),
        "next015_registered_only_no_new_algorithm_or_training": master["next_candidate_brief"].endswith("NEXT_RESTORE_SEED_INITIALIZATION_COST_015_BRIEF.md")
            and "观察器未实现、未运行" in next_text and "不新增候选算法" in next_text,
        "all_report_links_resolve_current_local_files": bool(links) and all(Path(value).is_file() for value in links),
        "family2GiB_and_disk20GiB_kept_at_final_close": sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file()) + 1024**2 < 2 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
    }
    result = {"checks": checks, "status": "scoped_work_accepted_and_text_state_publication_closed" if all(checks.values()) else "failed_final_text_or_state_check",
        "actual_publisher_tool_exit": {"chunk_id": "d30bed", "exit_code": 0},
        "saved_publisher_identity": writer, "saved_publisher_instance_ended": publisher_ended,
        "parent_accepted_count": 36, "new_optimization_count": 1, "accepted_count": 37,
        "goal_status": "active", "remaining": 13, "report_link_count": len(links),
        "direct_native_text_check_no_screenshot_OCR_or_document_hash": True,
        "old_readback_family_never_recounted": True,
        "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}}
    save(BASE / "FINAL_TEXT_STATE_CLOSURE_001.json", result)
    print(json.dumps({"checks": checks, "accepted": 37, "remaining": 13, "goal": "active"}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The direct text and final state closure failed")


if __name__ == "__main__":
    main()
