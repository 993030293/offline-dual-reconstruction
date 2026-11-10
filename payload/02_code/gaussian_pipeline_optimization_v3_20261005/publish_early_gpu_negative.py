"""Observe the completed negative review and check native text publication."""

import json
import msvcrt
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001/bootstrap_dag_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json
from gpu_query_node import remaining


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    began = time.perf_counter()
    output = OWN / "FINAL_SCREEN_PUBLICATION_001.json"
    if output.exists():
        raise FileExistsError("A final publication observation must be fresh")
    review = read(OWN / "r0/RESULT.json")
    leases = [read(OWN / name) for name in ("GENERIC_LEASE_001.json", "FIRST_PAIR_LEASE_001.json", "REVIEW_LEASE_001.json")]
    instances = [review["own_process"]]
    for lease in leases:
        for prefix in ("guard", "child"):
            instances.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    live = remaining(instances)
    state = read(DOC / "CURRENT_STATE.json")
    texts = {name: (DOC / name).read_text(encoding="utf-8") for name in
        ("R142_GPU_BOOTSTRAP_DAG_TASK.md", "R142_GPU_BOOTSTRAP_DAG_RESULTS.md", "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")}
    links = []
    for match in re.finditer(r"\]\((C:/[^)]+)\)", texts["R142_GPU_BOOTSTRAP_DAG_RESULTS.md"]):
        path = Path(match.group(1))
        links.append({"path": str(path), "exists_or_this_pending_publication": path.exists() or path == output})
    with (ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    owner = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/LOCK_OWNER.json")
    total = 75.030935 + sum(lease["child_elapsed_seconds"] for lease in leases)
    checks = {"review_original10_checks_preserved": len(review["checks"]) == 10 and all(review["checks"].values()),
        "actual_review_and_all_lease_instances_exited": not live,
        "all_three_actual_leases_returned0_and_released": all(lease["returncode"] == 0 and lease.get("lease_released_utc") for lease in leases),
        "actual_OS_exclusive_lock_acquired_and_released": True,
        "current_owner_released": owner["status"] == "released" and bool(owner.get("lease_released_utc")),
        "parent22_unique_accepted_receipts_and_no_active_candidate": state["additional_accepted_count"] == 22
            and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 22
            and len(state["acceptance_receipts"]) == len(set(state["acceptance_receipts"])) == 22 and not state["active_candidates"],
        "R142_negative_not_accepted": state["R142_new_optimization_count"] == 0 and state["R142_first_complete_reduction"] < .05,
        "known_old_generic_complete_and_review_computational_fees_kept": abs(total - state["R142_measured_family_computational_envelopes_seconds"]) < 1e-9,
        "report_links_target_current_files": bool(links) and all(row["exists_or_this_pending_publication"] for row in links),
        "native_text_scope_agrees": all("R142" in text for text in texts.values())
            and "未运行" in texts["R142_GPU_BOOTSTRAP_DAG_RESULTS.md"] and "不是三对正式速度" in texts["R142_GPU_BOOTSTRAP_DAG_RESULTS.md"]}
    if not all(checks.values()):
        raise AssertionError("A post-exit, state or native text publication guard failed")
    atomic_json(output, {"status": "negative_review_post_exit_and_native_text_publication_verified",
        "checks": checks, "observed_actual_review_and_lease_instances": instances, "actual_live_instances": live,
        "reviewed_native_text_characters": {name: len(text) for name, text in texts.items()}, "report_links": links,
        "known_measured_family_computational_envelopes_seconds": total,
        "last_publication_unmeasured_and_not_full_research_wall_time": True,
        "publisher_owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "publisher_own_shutdown_observation_is_separate": True,
        "new_optimization_count": 0, "parent_count": 22, "goal_still_active": True,
        "publication_before_last_serialization_seconds": time.perf_counter() - began})
    print(json.dumps({"status": "negative_review_post_exit_and_native_text_publication_verified", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
