"""Verify review exits, actual lock availability, and native text publication."""

import json
import msvcrt
import re
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OWN = OUT / "gpu_origin_dependency_001"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def closed(pid, creation):
    try:
        return abs(psutil.Process(pid).create_time() - creation) >= 0.001
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    destination = OWN / "FINAL_SCREEN_PUBLICATION_001.json"
    if destination.exists():
        raise FileExistsError("Preserve each actual publication receipt")
    closure = load(OWN / "COMPLETE_SCREEN_CLOSURE_001.json")
    checks = {"all_root_review_checks_passed": all(closure["checks"].values()),
              "actual_review_process_exited": closed(closure["review_pid"], closure["review_creation_time"])}
    leases, instances = [], []
    for number in range(1, 4):
        folder = OUT / ("gpu_origin_dependency_review_00" + str(number))
        lease = load(folder / "LEASE.json")
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("An actual checker lease did not release")
        leases.append(lease)
        for prefix in ("guard", "child"):
            row = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
            row["same_owned_instance_closed"] = closed(row["pid"], row["creation_time"])
            instances.append(row)
        if (folder / "EXECUTION.json").exists():
            execution = load(folder / "EXECUTION.json")
            row = {"pid": execution["pid"], "creation_time": execution["creation_time"]}
            row["same_owned_instance_closed"] = closed(row["pid"], row["creation_time"])
            instances.append(row)
    checks["all_eight_recorded_checker_instances_exited"] = len(instances) == 8 and all(
        row["same_owned_instance_closed"] for row in instances)
    owner = load(OUT / "coordination_001/LOCK_OWNER.json")
    checks["actual_final_review_owner_released"] = owner["status"] == "released" and owner["owner"] == "root_R118_complete_screen_review_003"
    with (OUT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_shared_OS_lock_available"] = True
    state = load(DOC / "CURRENT_STATE.json")
    checks["parent21_and_no_active_candidate_or_production_preserved"] = state["additional_accepted_count"] == 21 and len(set(
        state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 21 and state["active_candidates"] == [] and not state["production_modified"]
    text_rows, links = [], []
    for name in ("R118_COMPLETE_CONSUMER_SCREEN_TASK.md", "R118_COMPLETE_CONSUMER_SCREEN_RESULTS.md",
                 "CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md"):
        text = (DOC / name).read_text(encoding="utf-8")
        if not text.strip() or "\ufffd" in text:
            raise AssertionError("A native text publication is empty or corrupt")
        text_rows.append({"path": str(DOC / name), "direct_native_text_characters": len(text), "lines": len(text.splitlines()),
                          "render_screenshot_OCR_or_document_hash_used": False})
        if name == "R118_COMPLETE_CONSUMER_SCREEN_RESULTS.md":
            checks["result_text_scope_and_not_run_limits_present"] = all(term in text for term in (
                "3.749173%", "598", "1432", "1368", "21/50", "新增0", "不是正式三对", "not_run_after_negative_complete_screen", "两个失败原件"))
            for target in re.findall(r"\]\((C:/[^)\n]+)\)", text):
                if not Path(target).is_file():
                    raise AssertionError("A current result source link is missing")
                links.append(target)
    checks["three_checker_results_and_failures_retained"] = [row["returncode"] for row in leases] == [1, 1, 0] and all(
        (OWN / ("COMPLETE_SCREEN_REVIEWER_FAILURE_00" + str(i) + ".json")).is_file() for i in (1, 2))
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    payload = {"status": "negative_screen_native_text_links_actual_review_exit_and_OS_lock_publication_verified",
               "checks": checks, "review_kind": "root_self_review_no_subagents", "accepted_count_contribution": 0,
               "parent_accepted_count": 21, "parent_goal_not_completed": True,
               "reviewed_native_text": text_rows, "actual_result_links_verified": links,
               "actual_eight_review_process_identities_closed": instances,
               "all_R118_actual_lease_child_fees_including_this_screen": closure["known_R118_actual_child_fees_including_screen_seconds"],
               "all_three_checker_child_fees": sum(row["child_elapsed_seconds"] for row in leases),
               "old_R115_and_other_research_preparation_costs_not_waived_or_claimed_fully_measured": True,
               "publication_seconds_before_write": time.perf_counter() - started, "counted": False}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "recorded_checker_instances_closed": len(instances),
                      "current_result_links": len(links), "accepted_count_contribution": 0, "parent_accepted_count": 21}))


if __name__ == "__main__":
    main()
