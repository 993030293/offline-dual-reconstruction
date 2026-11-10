"""Verify native text, count consistency and post-review process exits."""

import json
import msvcrt
from pathlib import Path
import re
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/v6_residual_cost_003"


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_fields)


def ended(pid, born):
    try:
        return psutil.Process(pid).create_time() != born
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    state = read(DOC / "CURRENT_STATE.json")
    negative = read(OWN / "NEGATIVE_COST_CLOSURE_V6_001.json")
    checks = {"parent33_unique_mechanisms_and_receipts_no_R111_added":
        state["additional_accepted_count"] == len(set(state["accepted_mechanisms"])) == len(state["accepted_mechanisms"]) == len(state["acceptance_receipts"]) == 33
        and state["additional_target"] == 50 and state["status"].startswith("active")
        and state["V6_residual_003_new_optimization_count"] == state["R111_V6_new_optimization_count"] == 0,
        "all_33_existing_acceptance_targets_present": all((ROOT / name).is_file() for name in state["acceptance_receipts"]),
        "all_three_negative_upper_bounds_match_state": all(
            negative["candidates"][key]["net_complete_stage_upper_bound_before_this_publisher_and_following_fees"] < .02
            for key in ("R110", "R111", "R118")),
        "negative_publisher_actual_external_terminal_recorded": state["V6_residual_003_actual_negative_publisher_tool_chunk"] == "380431"
            and state["V6_residual_003_actual_negative_publisher_exit_code"] == 0}
    files = [DOC / name for name in ("V6_RESIDUAL_003_RESULTS.md", "R111_V6_COMPLETION_TASK.md", "NEXT_COMPLETE_COST_AUDIT_004_BRIEF.md")]
    texts = [path.read_text(encoding="utf-8") for path in files]
    targets = []
    for text in texts:
        for target in re.findall(r"\]\((C:/[^)]+)\)", text):
            path = Path(target)
            if path.name != "FINAL_POST_EXIT_V6_001.json" and not path.is_file():
                raise FileNotFoundError("A native-text reference is missing: " + target)
            targets.append(target)
    expected = {
        str(DOC / "R111_V6_COMPLETION_TASK.md"), str(DOC / "NEXT_COMPLETE_COST_AUDIT_004_BRIEF.md"),
        str(OWN / "sparse2/RESULT.json"), str(OWN / "SELF_REVIEW_V6_001.json"),
        str(OWN / "NEGATIVE_COST_CLOSURE_V6_001.json"), str(OWN / "FINAL_POST_EXIT_V6_001.json")}
    checks["native_report_task_and_next_brief_read_complete_and_local_targets_present"] = (
        {str(Path(target)) for target in targets} == expected
        and all("33/50" in text or "33/50" in text.replace(" ", "") for text in texts))
    checks["report_keeps_old_evidence_environment_and_cancellation_limits"] = all(
        token in texts[0] for token in ("1.9918%", "没有重跑这六次", "历史回执未保存包二进制哈希", "不证明文件物理写入中途", "不是独立代理审查", "33/50"))
    checks["three_current_document_prefixes_match_negative33"] = all(
        "33/50" in (DOC / name).read_text(encoding="utf-8").splitlines()[2]
        and ("关闭0" in (DOC / name).read_text(encoding="utf-8").splitlines()[2]
             or "各新增0" in (DOC / name).read_text(encoding="utf-8").splitlines()[2])
        for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md"))
    lease, guard = read(OWN / "review_lease.json"), read(OWN / "review_GUARD.json")
    instances = read(OWN / "SELF_REVIEW_V6_001.json")["owned_prior_instances"]
    instances += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                  {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    instances += guard["owned_process_identities"]
    unique = {(row["pid"], row["creation_time"]) for row in instances}
    rows = [{"pid": pid, "creation_time": born, "ended": ended(pid, born)} for pid, born in sorted(unique)]
    checks["all63_old_new_and_actual_review_instances_ended"] = len(rows) == 63 and all(row["ended"] for row in rows)
    coordination = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    checks["last_actual_shared_owner_released"] = read(coordination / "LOCK_OWNER.json")["status"] == "released" and lease["status"] == "child_exited" and guard["returncode"] == 0
    with (coordination / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_resource_lock_acquired_and_released"] = True
    if not all(checks.values()):
        raise AssertionError("Post-exit text or state verification failed: " + repr(checks))
    payload = {"status": "negative_three_candidate_text_count_and_actual_postreview_exit_verified",
        "checks": checks, "owned_instances": rows, "native_local_link_count": len(targets),
        "publisher_actual_external_tool_receipt": {"chunk_id": "380431", "exit_code": 0, "wall_time_seconds": .304015},
        "review_is_independent_agent": False, "parent_accepted_count": 33, "parent_remaining": 17,
        "new_optimization_count": 0, "parent_goal_complete": False,
        "current_postcheck_pid": psutil.Process().pid, "current_postcheck_creation_time": psutil.Process().create_time(),
        "current_postcheck_self_exit_not_proved_by_own_file": True,
        "actual_seconds_before_publication": time.perf_counter() - started,
        "original_postcheck_link_count_error_and_measured_failed_tool_kept": True,
        "original_postcheck_failed_tool_seconds": .2454826,
        "negative_speed_upper_bounds_remain_upper_bounds_remaining_costs_not_zero": True}
    with (OWN / "FINAL_POST_EXIT_V6_001.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checks": checks, "instances": len(rows), "new_optimization_count": 0}), flush=True)


if __name__ == "__main__":
    main()
