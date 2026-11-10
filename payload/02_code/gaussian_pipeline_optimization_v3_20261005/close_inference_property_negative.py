"""Close the negative branch after real reviewer exits and direct text checks."""

import datetime
import json
import msvcrt
from pathlib import Path
import re
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/aw001"
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A state or closure JSON key was repeated")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def main():
    target = OWN / "CLOSURE_001.json"
    if target.exists():
        raise FileExistsError("A complete failure closure must be new")
    review = read(OWN / "r0/RESULT.json")
    records = {(row["pid"], row["creation_time"]): row for row in review["actual_Windows_exited"]}
    records[(review["review_pid"], review["review_creation_time"])] = {
        "pid": review["review_pid"], "creation_time": review["review_creation_time"]}
    lease = read(OWN / "LEASE_R0.json")
    if lease["status"] != "child_exited" or lease["returncode"] != 0 or not lease.get("lease_released_utc"):
        raise AssertionError("The actual separate reviewer lease remains open")
    for role in ("guard", "child"):
        row = {"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]}
        records[(row["pid"], row["creation_time"])] = row
    for row in records.values():
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                raise AssertionError("A saved diagnostic or reviewer instance is still live")
        except psutil.NoSuchProcess:
            pass
    shared = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    if read(shared / "LOCK_OWNER.json")["status"] != "released":
        raise AssertionError("The actual OS benchmark owner did not release")
    with (shared / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    for row in review["source_input_records"]:
        if data_identity(row["path"]) != row:
            raise AssertionError("A reviewed current source or original input changed")
    names = ("R139_FIXED_MODEL_WORK_TASK.md", "R139_FIXED_MODEL_WORK_RESULTS.md", "NEXT_RAW_GRAY_INPUT_BRIEF.md",
             "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")
    texts = {name: (DOC / name).read_text(encoding="utf-8") for name in names}
    links = []
    for name, body in texts.items():
        if not body.strip() or "\ufffd" in body:
            raise AssertionError("A direct source text is empty or has a replacement character")
        for link in re.findall(r"\]\((C:/[^)]+)\)", body):
            path = link.split("#", 1)[0]
            if not Path(path).exists():
                raise AssertionError("An actual local evidence link does not exist: " + link)
            links.append({"document": name, "path": path})
    state = read(DOC / "CURRENT_STATE.json")
    report = texts["R139_FIXED_MODEL_WORK_RESULTS.md"]
    checks = {"all17_separate_root_review_checks_passed": len(review["checks"]) == 17 and all(review["checks"].values()),
        "all_recorded_diagnostic_and_reviewer_Windows_instances_actually_exited": bool(records),
        "all16_recorded_actual_Linux_instances_exited_at_review": len(review["actual_Linux_exited"]) == 16
            and not any(row["alive"] for row in review["actual_Linux_exited"]),
        "all12_actual_measurement_review_leases_returned_and_original_OS_lock_free": len(review["leases"]) == 11,
        "all46_reviewed_current_source_input_identities_intact": len(review["source_input_records"]) == 46,
        "six_direct_original_text_documents_and_actual_local_links_checked": len(texts) == 6 and bool(links),
        "negative_cost_and_procedural_error_not_hidden_or_promoted": "14.660652%" in report and "误启动" in report
            and "240" in report and "不是 Oxford" in report and "未接纳" in report,
        "current_parent24_unique_acceptances_and_no_active_candidate_or_R139_promotion": state["additional_accepted_count"] == 24
            and len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 24
            and state["active_candidates"] == [] and state["R139_fixed_work_new_optimization_count"] == 0}
    total_fee = review["known_fee_lower_bound_before_review_exit"] + lease["child_elapsed_seconds"] + time.perf_counter() - STARTED
    result = {"status": "closed_negative_all_current_original_protections_and_real_review_exit_verified" if all(checks.values()) else "failed_closure",
        "checks": checks, "Windows_instances_exited": list(records.values()), "actual_Linux_exited": review["actual_Linux_exited"],
        "direct_text_read_files": [str(DOC / name) for name in names], "actual_links": links,
        "known_fee_lower_bound_before_this_publisher_exit_seconds": total_fee,
        "fee_is_not_unmetered_text_research_or_total_project_time": True,
        "publisher_pid": psutil.Process().pid, "publisher_creation_time": psutil.Process().create_time(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "new_optimization_count": 0,
        "source": data_identity(Path(__file__)), "review_kind": "root_self_review_not_independent_agent"}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "checks": checks, "known_fee": total_fee}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An actual source, failure, text or exit closure check failed")


if __name__ == "__main__":
    main()
