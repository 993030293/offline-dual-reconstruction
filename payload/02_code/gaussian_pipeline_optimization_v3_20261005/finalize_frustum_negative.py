"""Publish the negative candidate after checking real exits and released locks."""

import ast
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import re
import time

import psutil

from review_frustum_negative import ROOT, OWN, JOBS, load, save, exited


def main():
    started = time.perf_counter()
    process = psutil.Process()
    publisher = {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()}
    review = load(OWN / "r0/RESULT.json")
    checks = {"root17_negative_review_passed": review["status"] == "negative_candidate_self_review_passed" and len(review["checks"]) == 17 and all(review["checks"].values())}
    leases, identities = [], []
    for name in JOBS + ["r0"]:
        lease = load(OWN / ("LEASE_" + name.upper() + ".json"))
        guard = load(OWN / (name.upper() + "_PROCESS.json"))
        leases.append(lease)
        expected = 1 if name in ["g0", "g1", "p1"] else 0
        checks[name + "_actual_guard_terminal_and_lease_released"] = guard["returncode"] == lease["returncode"] == expected and lease.get("lease_released_utc") is not None and lease["status"] == "child_exited"
        for entry in guard["owned_process_identities"] + [{"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ["guard", "child"]]:
            identities.append({"node": name, **entry, "actual_owned_instance_exited": exited(entry["pid"], entry["creation_time"])})
    checks["all_11_jobs_and_review_actual_owned_instances_exited"] = all(row["actual_owned_instance_exited"] for row in identities)
    coordination = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    owner = load(coordination / "LOCK_OWNER.json")
    checks["last_benchmark_owner_is_actual_released_review"] = owner["status"] == "released" and owner["owner"] == "root_R145_negative_review_001" and exited(owner["guard_pid"], owner["guard_creation_time"])
    with (coordination / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            checks["OS_exclusive_lock_actual_available_after_all_11_leases"] = True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    docroot = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    state = load(docroot / "CURRENT_STATE.json")
    checks["parent22_active_single_chat_candidate0_state_consistent"] = state["additional_accepted_count"] == 22 and state["additional_target"] == 50 and state["R145_new_optimization_count"] == 0 and state["active_candidates"] == [] and state["R145_status"].startswith("closed_negative_")
    report = (docroot / "R145_GPU_FRUSTUM_SORT_RESULTS.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    checks["all_delivered_native_file_and_directory_links_resolve"] = len(links) == 7 and all(Path(path).exists() for path in links)
    checks["broken_native_link_is_rejected_without_rendering"] = not all(Path(path).exists() for path in links[:-1] + [str(OWN / "missing_native_link_regression_001")])
    for name in ["PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md"]:
        text = (docroot / name).read_text(encoding="utf-8")
        checks[name + "_current_R145_failure_and_parent_state_text_present"] = "R145" in text[:1600] and "22/50" in text[:1800] and ("关闭0" in text[:1800] or "失败0" in text[:1800])
    for path in [Path(__file__), Path(__file__).with_name("review_frustum_negative.py")]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    checks["both_closure_python_sources_parse"] = True
    fees = sum(row["child_elapsed_seconds"] for row in leases)
    checks["all11_measured_child_costs_match_current_state"] = fees == state["R145_all11_measured_lease_child_seconds"]
    result = {"status": "negative_candidate_post_exit_publication_passed" if all(checks.values()) else "failed_verification",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "actual_Windows_instances": identities,
        "unique_actual_instances": len(set((x["pid"], x["creation_time"]) for x in identities)),
        "all11_measured_lease_child_seconds": fees, "old_R138_measured_family_seconds": review["old_R138_family_measured_child_seconds"],
        "known_family_measured_seconds": fees + review["old_R138_family_measured_child_seconds"],
        "complete_research_wall_cost_not_measured": True, "accepted_count_contribution": 0, "parent_count": 22,
        "native_parent_goal_still_active_and_50_not_complete": True,
        "publisher_identity": publisher, "publisher_requires_external_actual_exit_check": True,
        "metadata_publication_seconds_before_write": time.perf_counter() - started}
    save(OWN / "FINAL_NEGATIVE_PUBLICATION_002.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "instances": result["unique_actual_instances"], "seconds": fees}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
