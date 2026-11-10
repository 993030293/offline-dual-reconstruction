"""Publish measured negative costs and verify actually exited review owners."""

import json
from pathlib import Path
import re

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "sclc001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    source = data_identity(Path(__file__))
    if (OUT / "FINAL_NEGATIVE_COST_001.json").exists():
        raise FileExistsError("A negative closure must preserve its original publication")
    reviewed = read(OUT / "r2/RESULT.json")
    if len(reviewed["checks"]) != 19 or not all(reviewed["checks"].values()):
        raise AssertionError("The complete separate self-review did not pass")
    owners = [reviewed["owner"]]
    guards = {}
    for label, expected in (("r0", 1), ("r1", 1), ("r2", 0)):
        guard = read(OUT / (label + "_guard.json"))
        if guard["returncode"] != expected or guard["status"] not in ("completed", "failed_child"):
            raise AssertionError("A saved review tool status changed")
        guards[label] = guard
        owners += guard["owned_process_identities"]
    if not all(windows_closed(row) for row in owners):
        raise AssertionError("An actual saved root-review process still runs")
    fee_record = read(OUT / "ROOT_REVIEW_EXECUTION_FEES_001.json")
    review_fee = sum(fee_record["known_execution_fees_seconds"].values())
    rows = {}
    for name, candidate in (("sip001", "isolated"), ("spr004", "operand")):
        formal = read(BASE / name / "c0/RESULT.json")
        if len(formal["pairs"]) != 1 or formal["new_optimization_count"] != 0:
            raise AssertionError("A negative first screen cannot be promoted to three complete pairs")
        total_fee = formal["initial_known_fee_seconds"] + reviewed["formal_tail_seconds"][name] + review_fee + 10.
        native = formal["raw_complete_stage_medians"]["control"]
        proposed = formal["raw_complete_stage_medians"][candidate]
        bound = equal_fee_cost_bound(native, proposed, total_fee)
        if bound["conservative_sample_reduction_upper_bound"] >= .03:
            raise AssertionError("The known-cost negative screen no longer justifies closing")
        rows[name] = {"status": "negative_first_screen_not_accepted0", "actual_pairs": 1,
            "native_complete_seconds": native, "candidate_complete_seconds": proposed,
            "raw_reduction": 1 - proposed / native, "initial_known_fee_seconds": formal["initial_known_fee_seconds"],
            "formal_envelope_tail_seconds": reviewed["formal_tail_seconds"][name],
            "known_review_and_reader_fees_seconds": review_fee, "original_closing_execution_reserve_seconds": 10.,
            "known_fee_plus_original10_reserve_seconds": total_fee, "sampled_six_side_known_fee_upper_bound": bound,
            "remaining_two_formal_pairs_interfaces_real_errors_recovery_not_run": True,
            "new_optimization_count": 0}
    state = read(DOC / "CURRENT_STATE.json")
    if state["additional_accepted_count"] != 29 or state["active_candidates"]:
        raise AssertionError("The negative results changed the accepted frontier")
    payload = {"status": "both_distinct_complete_first_screens_closed_negative0", "candidates": rows,
        "review_fee_receipt": str(OUT / "ROOT_REVIEW_EXECUTION_FEES_001.json"), "actual_review_guards": guards,
        "all_saved_actual_root_review_owners_now_closed": True, "review_is_independent": False,
        "accepted_count": 29, "remaining": 21, "new_optimization_count": 0, "goal_active_not_complete": True,
        "unmeasured_text_research_and_total_RD_ROI_not_assumed_zero": True,
        "current_writer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time(), "source": source},
        "current_writer_exit_verification_pending_tool_return": True}
    atomic_json(OUT / "FINAL_NEGATIVE_COST_001.json", payload)
    files = [DOC / name for name in ("COMPOSED_SCENE_COST_AND_PHASE_RESULTS.md", "NEXT_IMAGE_HANDOFF_COST_BRIEF.md",
        "NEXT_COMPOSED_SCENE_COST_BRIEF.md", "R154_FRESH_PHASE_ISOLATION_TASK.md", "R155_PATH_OPERAND_TASK.md", "COMPOSED_SCENE_MAIN_CALL_COST_TASK.md")]
    texts = {str(path): path.read_text(encoding="utf-8") for path in files}
    links = [target for text in texts.values() for target in re.findall(r"\]\((C:/[^)]+)\)", text)]
    if not all(Path(target).exists() for target in links):
        raise AssertionError("A current result document refers to a missing local artifact")
    checks = {"all6_current_result_task_brief_texts_directly_read": len(texts) == 6,
        "all_current_local_result_links_resolve": bool(links),
        "report_exact_first_pair_and_failed_scope_present": all(value in texts[str(files[0])] for value in ("一对首筛", "54.929574", "1.521874", "停止后未执行", "自审不包装成独立审查")),
        "next_brief_no_new_handoff_algorithm_or_gain_claim": "尚未实现或执行新算法" in texts[str(files[1])],
        "writer_actual_source_unchanged": source == data_identity(Path(__file__)),
        "root_self_review_actual_tools_exited_with_original_return_codes": all(windows_closed(row) for row in owners)}
    if not all(checks.values()):
        raise AssertionError("The negative closure text or actual exit check failed")
    atomic_json(OUT / "FINAL_TEXT_AND_REVIEW_EXIT_001.json", {"checks": checks, "texts_read": list(texts),
        "local_reference_targets": links, "actual_closed_review_identities": owners, "writer_source": source,
        "document_content_checked_as_native_text_not_hash_or_screenshot": True, "new_optimization_count": 0})
    print(json.dumps({"checks": checks, "candidate_upper_bounds": {k: r["sampled_six_side_known_fee_upper_bound"]["conservative_sample_reduction_upper_bound"] for k, r in rows.items()}, "accepted": 29, "remaining": 21, "new": 0}))


if __name__ == "__main__":
    main()
