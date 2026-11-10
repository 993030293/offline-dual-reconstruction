"""Verify the original report/state text and the actual final publisher exit."""

import ast
import json
import msvcrt
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
DOC = ROOT/"research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("An authoritative JSON object repeats a key: "+key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def main():
    started = time.perf_counter()
    output = OUT/"FINAL_POST_EXIT_001.json"
    if output.exists():
        raise FileExistsError("The text/post-exit evidence must have a fresh output")
    final = read(OUT/"FINAL_COST_NEGATIVE_001.json")
    state = read(DOC/"CURRENT_STATE.json")
    identities = [final["publisher"]]
    owner = read(OUT.with_name("coordination_001")/"LOCK_OWNER.json")
    identities += [{"pid": owner["guard_pid"], "creation_time": owner["guard_creation_time"]},
                   {"pid": owner["child_pid"], "creation_time": owner["child_creation_time"]}]
    observed = []
    for row in identities:
        alive = False
        try:
            process = psutil.Process(row["pid"])
            alive = process.create_time() == row["creation_time"] and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        observed.append(dict(row, alive=alive))
    files = [DOC/name for name in ("R123_NATIVE_WINDOWS_SAMPLING_RESULTS.md", "R123_NATIVE_WINDOWS_SAMPLING_TASK.md", "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")]
    texts = {path: path.read_text(encoding="utf-8") for path in files}
    report = texts[files[0]]
    links = []
    for text in texts.values():
        for target in re.findall(r"\]\((C:/[^)]+)\)", text):
            links.append({"path": target, "exists": Path(target).exists()})
    receipts = []
    for path in state["acceptance_receipts"]:
        row = read(ROOT/path)
        counts = [row[key] for key in ("new_optimization_count", "additional_accepted_count", "accepted_count_contribution") if key in row]
        receipts.append({"path": path, "one_mechanism": 1 in counts and row["status"].startswith("accepted")})
    parsed = []
    for name in ("native_camera_sampling.py", "build_native_camera_sampling.py", "check_native_camera_sampling.py", "check_native_sampling_boundaries.py", "localize_native_camera_sampling.py", "run_native_sampling_campaign.py", "review_native_sampling_cost.py", "close_native_sampling_negative.py", "check_native_sampling_publication.py"):
        path = CODE/name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parsed.append(str(path))
    checks = {"all_nine_final_closure_checks_passed": len(final["checks"]) == 9 and all(final["checks"].values()),
              "actual_final_publisher_and_latest_lease_instances_exited": not any(r["alive"] for r in observed),
              "all22_current_actual_acceptance_receipts_count_one": len(receipts) == state["additional_accepted_count"] == 22 and all(r["one_mechanism"] for r in receipts),
              "R123_count0_closed_no_active_candidate": state["R123_native_new_optimization_count"] == final["new_optimization_count"] == 0 and state["active_candidates"] == [] and state["R123_native_status"].startswith("closed_negative"),
              "parent_goal_not_false_complete": not final["parent_goal_complete"] and state["additional_target"] == 50,
              "all_original_text_local_links_exist": bool(links) and all(r["exists"] for r in links),
              "report_actual_raw_and_cost_scope_explicit": all(value in report for value in ("9.552474%", "2.468254%", "2.535059%", "260", "21", "40", "404", "81", "8", "22/50")),
              "no_new_training_Bitmap_read_GT_unknown_objects_claim": "均未运行" in report and "不重跑" in report,
              "all_nine_current_source_texts_parse": len(parsed) == 9,
              "latest_OS_owner_released": owner["status"] == "released" and bool(owner.get("lease_released_utc"))}
    with (OUT.with_name("coordination_001")/"exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["OS_resource_lock_free_after_publication"] = True
    atomic_json(output, {"status": "text_state_and_post_exit_verified" if all(checks.values()) else "failed_verification",
        "review_kind": "root_direct_original_text_JSON_source_and_actual_process_review", "checks": checks,
        "actual_instances": observed, "read_original_texts": [str(p) for p in files], "local_links": links,
        "all22_actual_receipts": receipts, "parsed_source_texts": parsed, "text_and_post_exit_seconds": time.perf_counter()-started,
        "publisher": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "no_document_hash_OCR_or_screenshot": True, "new_optimization_count": 0, "parent_additional_accepted_count": 22,
        "parent_goal_complete": False, "negative_full_cost_upper_bound_unchanged_by_nonnegative_publication_fee": True})
    print(json.dumps({"status": "text_state_and_post_exit_verified" if all(checks.values()) else "failed_verification", "checks": len(checks), "parent_count": 22}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
