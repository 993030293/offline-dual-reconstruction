"""Check publication metadata after the diagnostic publisher actually exits."""

import json
import msvcrt
from pathlib import Path
import re
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A JSON object contains a duplicate key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def exited(pid, creation_time):
    try:
        return psutil.Process(pid).create_time() != creation_time
    except psutil.NoSuchProcess:
        return True


def main():
    target = OUT / "olp001/FINAL_POST_EXIT_001.json"
    if target.exists():
        raise FileExistsError("The final diagnostic publication already exists")
    publications = []
    for path in (OUT / "olp001/DIAGNOSTIC_CLOSURE_001.json",
                 OUT / "mj_v4_001/V4_NEGATIVE_CLOSURE_001.json",
                 OUT / "pf_v4_001/V4_NEGATIVE_CLOSURE_001.json"):
        row = read(path)
        actual = exited(row["publisher_pid"], row["publisher_creation_time"])
        publications.append({"path": str(path), "publisher_pid": row["publisher_pid"],
                             "publisher_creation_time": row["publisher_creation_time"],
                             "actual_saved_instance_exited": actual})
    state = read(DOC / "CURRENT_STATE.json")
    names, receipts = state["accepted_mechanisms"], state["acceptance_receipts"]
    checks = {"all_three_actual_publishers_exited": all(row["actual_saved_instance_exited"] for row in publications),
              "counter_and_unique_mechanisms_and_receipts24": state["additional_accepted_count"] == len(names) == len(set(names)) == len(receipts) == len(set(receipts)) == 24,
              "all24_acceptance_receipts_exist_and_have_no_duplicate_keys": True}
    for receipt in receipts:
        read(ROOT / receipt)
    links = []
    for name in ("LOCALIZATION_HANDOFF_PROFILE_RESULTS.md", "R110_V4_COST_BOUND_RESULTS.md",
                 "R113_V4_OPTIONAL_FILTER_RESULTS.md"):
        text = (DOC / name).read_text(encoding="utf-8")
        for value in re.findall(r"\]\((C:/[^)]+)\)", text):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            links.append({"document": name, "target": value})
    checks["all_actual_local_report_links_exist"] = bool(links)
    with (OUT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_exclusive_OS_lock_available_and_released"] = True
    if not all(checks.values()):
        raise AssertionError("A final diagnostic publication check failed")
    result = {"status": "post_exit_metadata_checked_not_new_optimization_or_performance_pair",
              "checks": checks, "publishers": publications, "actual_report_links": links,
              "parent_accepted_count": 24, "goal_still_active": True,
              "new_optimization_count": 0, "text_reports_read_directly": True,
              "not_an_independent_agent_review": True}
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "publishers": len(publications),
                      "local_links": len(links), "parent_accepted_count": 24, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
