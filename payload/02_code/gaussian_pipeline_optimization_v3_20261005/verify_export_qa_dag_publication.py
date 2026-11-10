"""Verify native text, current counters, local links and post-audit process exit."""

import datetime
import json
from pathlib import Path
import re
import shutil

import psutil

from profile_export_qa_dag import ROOT, BASE
from runtime import atomic_json, data_identity

CODE = Path(__file__).resolve().parent
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    destination = BASE / "FINAL_PUBLICATION_001.json"
    if destination.exists():
        raise FileExistsError("A publication check cannot overwrite prior evidence")
    names = ("R140_EXPORT_QA_UNDISTORT_DAG_TASK.md", "R140_EXPORT_QA_UNDISTORT_DAG_RESULTS.md")
    texts = {name: (DOC / name).read_text(encoding="utf-8-sig") for name in names}
    changed_texts = {name: (DOC / name).read_text(encoding="utf-8-sig") for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")}
    state = read(DOC / "CURRENT_STATE.json")
    acceptance, cost, review = (read(BASE / name) for name in ("ACCEPTANCE.json", "FINAL_COST_001.json", "COST_REVIEW_PENDING_001.json"))
    links = [Path(target) for text in texts.values() for target in re.findall(r"\]\((C:/[^)]+)\)", text)]
    identities = [review["closer_identity"], acceptance["publisher_identity"]]
    identities += read(BASE / "CLOSE_GUARD_001.json")["owned_process_identities"]
    unique = {(row["pid"], row["creation_time"]): row for row in identities}
    report = texts[names[1]]
    phases = cost["reductions_including_fees"]
    receipt_statuses = [read(ROOT / name)["status"] for name in state["acceptance_receipts"]]
    checks = {
        "native_complete_task_and_report_text_read_no_document_hash": all(text.strip() and len(text.splitlines()) >= 40 for text in texts.values()),
        "report_numbers_bound_to_actual_final_cost": all(f"{phases[kind] * 100:.7f}%" in report for kind in ("phase", "whole"))
            and "223.5084057" in report and state["R140_all_measured_shared_fees_seconds"] == cost["all_known_measured_nonoverlapping_fees_seconds"],
        "current21_unique_mechanisms_and_actual_accepted_receipts": state["additional_accepted_count"] == 21
            and len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 21 and all(status.startswith("accepted") for status in receipt_statuses),
        "actual_acceptance_one_with_original5percent_protections": acceptance["new_optimization_count"] == state["R140_new_optimization_count"] == 1
            and all(acceptance["checks"].values()) and state["minimum_complete_stage_speed_reduction_current"] == .05,
        "plan_board_checklist_current_R140_text_and_count_consistent": "21/50" in changed_texts["PLAN.md"] and "21/50" in changed_texts["CANDIDATE_BOARD.md"]
            and "父21/50 active" in changed_texts["OPTIMIZE_CHECKLIST.md"] and "R140" in "".join(changed_texts.values()),
        "all_local_report_links_resolve": len(links) >= 6 and all(path.is_file() for path in links),
        "actual_root_closer_and_publisher_PID_creation_time_exited": all(gone(row) for row in unique.values()),
        "scoped_public_OPENCV_failure_and_unrun_training_GT_bounds_preserved": "训练准入" in report and "原失败" in report
            and acceptance["Oxford_current_PINHOLE_training_admission"] == "not_passed_original_native_zero_distortion_OPENCV_remains"
            and acceptance["new_training_GT_new_object_whole_pipeline_preview"] == "not_run",
        "no_live_candidate_or_production_edit_no_independent_review_claim": state["active_candidates"] == [] and not state["production_modified"]
            and not acceptance["production_deployed"] and acceptance["review"] == "root_single_chat_self_review_no_subagents_or_independent_review",
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    saved = BASE / "PUBLISHER_SOURCE_001.py"
    if saved.exists():
        raise FileExistsError("The publisher source receipt must be fresh")
    shutil.copyfile(CODE / "finalize_export_qa_dag.py", saved)
    atomic_json(destination, {"status": "accepted_R140_native_text_and_actual_post_exit_publication_verified", "checks": checks,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "documents": [{"path": str(DOC / name), "line_count": len(text.splitlines()),
            "reading_mode": "complete_native_UTF8_text", "screenshot_OCR_or_document_hash": False} for name, text in texts.items()],
        "changed_current_document_sections": list(changed_texts), "local_links": [str(path) for path in links],
        "post_exit_instances": [{**row, "gone": gone(row)} for row in unique.values()], "publisher_source": data_identity(saved),
        "accepted_additional_count": 21, "parent50_goal_not_complete": True, "no_subagents_or_independent_review": True,
        "this_text_and_receipt_publication_after_measured_cost_boundary": True})
    print(json.dumps({"status": "accepted_R140_native_text_and_actual_post_exit_publication_verified", "checks": checks, "accepted_count": 21}), flush=True)


if __name__ == "__main__":
    main()
