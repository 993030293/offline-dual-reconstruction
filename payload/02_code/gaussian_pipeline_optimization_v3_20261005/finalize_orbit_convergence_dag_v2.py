"""Close scoped text, ownership, current sources and cost without changing preview code."""

import datetime
import msvcrt
import os
import re
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import DIAG, DOC, OWN, ROOT, closed, identity, matches, read, save
from review_orbit_convergence_dag_v2 import V1, V2, cost, fee_passes


def main():
    started = time.perf_counter()
    writer = {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "source": identity(Path(__file__))}
    checks = {}
    rows, leases = [], []
    for folder, prefixes in [(DIAG, ["p0", "p1", "p2"]), (OWN, V1 + V2 + ["review_v2", "publisher_v2"])]:
        for prefix in prefixes:
            guard, lease = read(folder / f"{prefix}_GUARD.json"), read(folder / f"{prefix}_lease.json")
            rows.extend(guard.get("owned_process_identities", []))
            rows.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]}, {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
            leases.append(lease)
    checks["all_previous_owned_PID_births_ended_and_leases_released"] = all(closed(row) for row in rows) and all(row.get("lease_released_utc") for row in leases)
    publisher = read(OWN / "PUBLISHER_ACTUAL_POST_EXIT_V2_001.json")
    checks["actual_V2_review_publisher_exit0_and_sources_unchanged"] = publisher["actual_tool_exit_code"] == 0 and publisher["all_births_ended"] and read(OWN / "review_v2_GUARD.json")["returncode"] == 0 and matches(read(OWN / "SELF_REVIEW_V2_V6_001.json")["reviewer"]["source"])
    lock_path = OWN.parent / "coordination_001/exclusive_benchmark.lock"
    owner = read(lock_path.with_name("LOCK_OWNER.json"))
    with lock_path.open("r+b", buffering=0) as handle:
        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_acquired_released_and_latest_owner_released"] = owner["status"] == "released" and bool(owner.get("lease_released_utc"))
    current = read(DOC / "CURRENT_STATE.json")
    accepted = current["accepted_mechanisms"]
    checks["parent34_of50_active_single_R162_V2_and_34_corresponding_receipts"] = current["additional_accepted_count"] == len(accepted) == len(set(accepted)) == len(current["acceptance_receipts"]) == 34 and sum(row.startswith("R162") for row in accepted) == 1 and current["R162_new_optimization_count"] == 1 and current["additional_target"] == 50
    checks["all_current_acceptance_receipts_exist_and_current_R162_uses_V2"] = all((ROOT / name).is_file() for name in current["acceptance_receipts"]) and current["R162_helper"].endswith("orbit_convergence_dag_v2.js") and current["R162_acceptance"].endswith("ACCEPTANCE_V2_V6_001.json")
    unique = {}
    for prefix in ["g3", "g4", "c3", "c4", "c5", "r1"]:
        result = read(OWN / prefix / "RESULT.json")
        for row in result["inputs"] + result["sources"]:
            unique[row["path"]] = row
    checks["all_current_new_executed_sources_and_original_assets_unchanged"] = all(matches(row) for row in unique.values())
    report = (DOC / "R162_V2_ORBIT_CONVERGENCE_DAG_RESULTS.md").read_text(encoding="utf-8")
    checks["report_text_full_metrics_quality_lifecycle_failures_cost_and_claim_boundaries"] = all(word in report for word in ["34/50 active", "8.6339774", "3.3257057", "61.481186%", "9.520269%", "V1临时发布", "立即撤回", "全部保留", "1504", "41个cycle", "145个", "五种", "真实原DOM", "24项", "未计量文本研究工时不作0", "未部署", "未知物体完整重建"])
    latest = [(name, (DOC / name).read_text(encoding="utf-8").split("\n\n")[1]) for name in ["CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md"]]
    checks["all_three_latest_frontiers_use_V2_parent34_and_mark_V1_history"] = all("V2" in paragraph and "34/50 active" in paragraph and "V1" in paragraph for _, paragraph in latest)
    history = (DOC / "R162_ORBIT_CONVERGENCE_DAG_RESULTS.md").read_text(encoding="utf-8")
    next_task = (DOC / "NEXT_SORT_CAPACITY_COST_006_BRIEF.md").read_text(encoding="utf-8")
    checks["V1_report_explicitly_retracted_and_next006_only_source_cost_no_count"] = "历史V1记录，接纳已撤回" in history and "没有生成" in history and "当前R162 V2" in next_task and "尚未实现或执行" in next_task and "新增0" in next_task
    future = {str(OWN / "FINAL_COST_V2_V6_002.json"), str(OWN / "FINAL_POST_EXIT_V2_V6_001.json")}
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    checks["current_report_absolute_links_exist_except_two_explicit_final_exit_outputs"] = bool(links) and all(Path(name).is_file() or str(Path(name)) in future for name in links)
    checks["private_output_under2GiB_and_disk_reserve20GiB"] = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) < 2 * 1024**3 and psutil.disk_usage(str(ROOT)).free >= 20 * 1024**3
    extra = read(OWN / "review_v2_lease.json")["child_elapsed_seconds"] + read(OWN / "publisher_v2_lease.json")["child_elapsed_seconds"] + time.perf_counter() - started
    fees = cost(extra)
    checks["all_old_new_fees_net2percent_and_whole_guards_remain"] = fee_passes(fees)
    save(OWN / "FINALIZER_V2_TEXT_RESOURCE_CLOSURE_001.json", {
        "status": "V2_text_resource_source_cost_closure_passed_pending_actual_writer_exit" if all(checks.values()) else "V2_closure_failed",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks,
        "writer": writer, "process_identity_count": len(rows), "unique_current_sources_assets": len(unique),
        "absolute_report_links": len(links), "text_read_directly_no_document_hash_or_OCR": True,
        "review_is_independent_agent": False, "cost_before_actual_finalizer_exit": fees,
        "new_optimization_count": 0, "parent": 34, "target": 50, "remaining": 16,
        "production_default_service_new_training_mapper_LIO_GT_cross_hardware": "not_run"})
    print({"checks": checks, "net_before_actual_exit": fees["net_complete_stage_reduction_under_registered_six_side_model"]}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The final source, text, resource or fee closure failed")


if __name__ == "__main__":
    main()
