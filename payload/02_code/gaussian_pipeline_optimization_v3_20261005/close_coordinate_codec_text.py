"""Verify native report text, current target and the completed audit writer."""

import json
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def main():
    started = time.perf_counter()
    text_paths = [DOC / name for name in ("R169_COORDINATE_OPERAND_REUSE_RESULTS.md",
        "R170_LOSSLESS_MATCH_REQUEST_RESULTS.md", "TARGET_OVERRIDE_40.md",
        "CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md")]
    texts = [path.read_text(encoding="utf-8") for path in text_paths]
    targets = []
    for text in texts[:3]:
        for target in re.findall(r"\]\((C:/[^)]+)\)", text):
            targets.append({"path": target, "exists": Path(target).exists()})
    state = json.loads((DOC / "CURRENT_STATE.json").read_text(encoding="utf-8"))
    review = json.loads((BASE / "oup018/negative_review1/RESULT.json").read_text(encoding="utf-8"))
    writer = review["writer"]
    try:
        ended = psutil.Process(writer["pid"]).create_time() != writer["creation_time"]
    except psutil.NoSuchProcess:
        ended = True
    leases = [json.loads((BASE / "oup018" / name).read_text(encoding="utf-8")) for name in
              ("NEGATIVE_REVIEW_LEASE.json", "NEGATIVE_REVIEW_V2_LEASE.json")]
    checks = {
        "user_target40_current38_unique_no_negative_count": state["additional_target"] == 40
            and state["additional_accepted_count"] == len(set(state["accepted_mechanisms"])) == 38
            and state["R169_new_optimization_count"] == state["R170_new_optimization_count"] == 0,
        "all_six_native_texts_read_not_rendered_or_hashed": all(text.strip() for text in texts),
        "negative_original_numbers_and_unrun_branches_stated": "3.498205%" in texts[0]
            and "未执行" in texts[0] and "10.489105%" in texts[1] and "8.5886853185e-13" in texts[1]
            and "未运行" in texts[1] and "贡献0" in texts[1],
        "current_target_at_top_of_three_state_documents": all("38/40" in text[:600] for text in texts[3:]),
        "all_direct_current_report_links_exist": bool(targets) and all(row["exists"] for row in targets),
        "actual_separate_root14_passed_not_independent_claim": len(review["checks"]) == 14
            and all(review["checks"].values()) and "自审，不是独立代理" in texts[1],
        "actual_root_writer_birth_ended": ended,
        "both_failed_and_successful_review_leases_released": all("lease_released_utc" in row for row in leases)
            and leases[0]["returncode"] == 1 and leases[1]["returncode"] == 0,
    }
    result = {"status": "closed_negative_text_state_and_actual_writer_exit" if all(checks.values()) else "failed_verification",
              "checks": checks, "report_text_paths": [str(path) for path in text_paths], "links": targets,
              "actual_previous_writer": writer, "single_chat_no_subagents": True,
              "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0,
              "goal_status_active_target40_effective38_remaining2": True}
    path = BASE / "oup018/FINAL_TEXT_STATE_CLOSURE_001.json"
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "complete_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A current native text, negative status or writer-exit check failed")


if __name__ == "__main__":
    main()
