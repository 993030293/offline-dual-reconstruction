"""Check post-exit resources, current text evidence and the unchanged accepted count."""

import datetime
import json
import msvcrt
import os
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "ud001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A current JSON evidence object contains a duplicate key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    output = OWN / "POST_EXIT_CLOSURE_001.json"
    if output.exists(): raise FileExistsError("The post-exit closure must preserve its first result")
    identities, leases = [], []
    for name in ("p0", "p1", "c0", "r0", "s0", "r1"):
        lease, guard = read(OWN / (name + "_lease.json")), read(OWN / (name + "_guard.json"))
        leases.append(lease)
        identities.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}, *guard["owned_process_identities"]])
        identity_path = OWN / name / "PROCESS_IDENTITY.json"
        if identity_path.exists(): identities.append(read(identity_path))
        if guard["status"] != "completed" or guard["returncode"] != 0:
            raise AssertionError("A registered dataset screen guard did not complete")
    for name in ("c0", "s0"):
        pair = read(OWN / name / "RESULT.json")
        identities.extend(row["owner"] for row in pair["rows"])
        identities.extend(row["owner"] for row in pair["rows"][1]["result"]["trace"]["workers"])
        for mode in ("native", "cached"):
            identities.append(read(OWN / name / mode / "PROCESS_IDENTITY.json"))
    identities = list({(row["pid"], row["creation_time"]): row for row in identities}.values())
    state = read(DOC / "CURRENT_STATE.json")
    names = ["R123_DATASET_ROUTE_TASK.md", "R123_DATASET_CONFIGURATION_DEBUG_TASK.md", "R123_COMPLETE_DATASET_COST_TASK.md",
        "R123_SHARED_MAP_TASK.md", "R123_COMPLETE_DATASET_SCREEN_RESULTS.md", "R123_SHARED_MAP_SCREEN_RESULTS.md",
        "CURRENT_MATCH_SELECTION_NOVELTY_AUDIT.md", "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md"]
    texts = {name: (DOC / name).read_text(encoding="utf-8") for name in names}
    links = []
    for name, text in texts.items():
        for target in re.findall(r"\]\((C:/[^)]+)\)", text):
            if not Path(target).exists(): raise AssertionError("A current local evidence link does not resolve: " + target)
            links.append({"document": name, "target": target})
    review0, review1 = read(OWN / "r0/REVIEW.json"), read(OWN / "r1/REVIEW.json")
    current_owner = read(OUT / "coordination_001/LOCK_OWNER.json")
    checks = {
        "all_saved_actual56_or_other_instances_exited_not_inferred_from_file": all(closed(row) for row in identities),
        "six_registered_leases_returned_and_released": all(row["status"] == "child_exited" and row["returncode"] == 0 and row["lease_released_utc"] for row in leases),
        "both_separate_root_reviews_passed": all(review0["checks"].values()) and all(review1["checks"].values()),
        "current_shared_owner_released_after_real_return": current_owner["owner"] == "root_R123_shared_complete_dataset_review_r1" and current_owner["status"] == "released",
        "accepted24_unique_receipts_resolve_and_goal50_kept": state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 24
            and len(state["acceptance_receipts"]) == 24 and all((ROOT / name).is_file() for name in state["acceptance_receipts"]) and state["additional_target"] == 50,
        "active_rank_no_uncompleted_benchmark_or_new_acceptance": state["status"].startswith("active") and state["primary_optimize_submode"] == "rank"
            and state["active_candidates"] == [] and state["R123_dataset_route_new_optimization_count"] == 0,
        "current_source_report_negative_four_and_deferred_shared_bound_accurate": "49.2544509" in texts["R123_COMPLETE_DATASET_SCREEN_RESULTS.md"]
            and "30.9044200" in texts["R123_SHARED_MAP_SCREEN_RESULTS.md"] and "3.248761" in texts["R123_SHARED_MAP_SCREEN_RESULTS.md"]
            and "不是whole" in texts["R123_SHARED_MAP_SCREEN_RESULTS.md"],
        "current_source_report_scope_and_independent_review_boundary_kept": "取消后恢复" in texts["R123_SHARED_MAP_SCREEN_RESULTS.md"]
            and "不能称为未执行的三对" in texts["R123_SHARED_MAP_SCREEN_RESULTS.md"]
            and "不冒充独立代理审查" in texts["R123_COMPLETE_DATASET_SCREEN_RESULTS.md"],
        "current_frontier_text_contains24of50_and_latest_shared_result": all("24/50" in texts[name][:2500] and "3.248761" in texts[name][:2500]
            for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")),
        "all_current_local_markdown_links_resolve": bool(links),
    }
    if not all(checks.values()):
        atomic_json(OWN / "POST_EXIT_FAILED_TEXT_REVIEW_001.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A required dataset screen text or post-exit review failed")
    descriptor = os.open(OUT / "coordination_001/exclusive_benchmark.lock", os.O_RDWR)
    with os.fdopen(descriptor, "r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        checks["OS_resource_lock_actually_free"] = True
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    for row in read(OWN / "s0/cached/sampling_proof/SHARED_SEGMENTS_CLOSED.json")["headers"]:
        from multiprocessing import shared_memory
        try:
            segment = shared_memory.SharedMemory(name=row["name"], create=False)
        except FileNotFoundError:
            continue
        segment.close()
        raise AssertionError("A named shared map still exists after actual consumer exit")
    old = read(OUT / "rm_native001/FINAL_COST_NEGATIVE_001.json")
    known = old["total_old_and_new_measured_computational_envelopes_including_formal_windows_lower_bound_seconds"] + sum(row["child_elapsed_seconds"] for row in leases)
    publisher = psutil.Process()
    payload = {"status": "screens_closed_exact_four_process_negative_and_shared_single_pair_cost_deferred0",
        "checks": checks, "actual_saved_instance_count": len(identities), "instances": identities,
        "six_actual_lease_cost_seconds": [row["child_elapsed_seconds"] for row in leases],
        "known_family_fee_lower_bound_after_both_review_exit_seconds": known,
        "text_review_files": names, "resolved_local_links": links,
        "text_only_review_no_document_hash_or_screenshot": True, "root_self_review_not_independent_agent": True,
        "publisher": {"pid": publisher.pid, "creation_time": publisher.create_time()},
        "publisher_internal_before_result_seconds": time.perf_counter() - started,
        "publisher_startup_exit_and_unmetered_text_research_not_assumed_zero": True,
        "additional_accepted_count": 24, "additional_target": 50, "new_optimization_count": 0,
        "parent_goal_complete": False, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "actual_saved_instance_count", "known_family_fee_lower_bound_after_both_review_exit_seconds", "additional_accepted_count", "new_optimization_count")}))


if __name__ == "__main__":
    main()
