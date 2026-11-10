"""Close a read-only complete GPU upload diagnostic without accepting an optimization."""

import datetime
import msvcrt
import os
import re
from pathlib import Path

import psutil

from review_orbit_convergence_dag import DOC, OWN as ORBIT, ROOT, closed, identity, matches, read, save


def main():
    output = ORBIT.parent / "oup006"
    result = read(output / "p2/RESULT.json")
    checks = {"all9_original_complete_quality_checks": len(result["checks"]) == 9 and all(result["checks"].values())}
    checks["both_contexts_same_R162_V2_all21_views_and_native_aliases_drained"] = all(len(episode["views"]) == 21 and episode["orbitDag"]["converged"] == 7 and episode["orbitDag"]["nativeUpdateUnchanged"] and episode["disposed"]["orderingUpload"]["originalAliasRestored"] for episode in result["episodes"])
    measurements = []
    for episode in result["episodes"]:
        measurements.append({"context_label": episode["hooks"], "complete_context_seconds": episode["complete_context_to_actual_close_ms"] / 1000,
                             "complete_upload_totals": episode["disposed"]["orderingUpload"]["totals"]})
    checks["recorded_complete_upload_bytes_positive_and_only_potential_not_applied"] = all(row["complete_upload_totals"]["orderingSubmittedBytes"] > 0 and row["complete_upload_totals"]["unmodifiedTailBytesPotential"] > 0 for row in measurements) and all(episode["views"][-1]["capture"]["orderingUpload"]["unmodifiedOriginalArguments"] for episode in result["episodes"])
    checks["all_original_assets_vendor_models_and_executed_sources_unchanged"] = all(matches(row) for row in result["inputs"] + result["sources"])
    snapshots = [{"path": str(output / "p2/sources" / Path(row["path"]).name), "bytes": row["bytes"], "sha256": row["sha256"]} for row in result["sources"]]
    checks["all_executed_source_snapshots_match_actual_sources"] = all(matches(row) for row in snapshots)
    rows, leases = [], []
    for prefix in ["p0", "p1", "p2"]:
        guard, lease = read(output / f"{prefix}_GUARD.json"), read(output / f"{prefix}_lease.json")
        rows.extend(guard.get("owned_process_identities", []))
        rows.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]}, {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
        leases.append(lease)
    checks["all_prior_owned_PID_births_ended_leases_released"] = all(closed(row) for row in rows) and all(row.get("lease_released_utc") for row in leases)
    checks["two_failed_sources_logs_fees_preserved_actual_exit_codes1_1_0"] = [row["returncode"] for row in leases] == [1, 1, 0] and all((output / name).is_file() for name in ["P0_ROUTE_BINDING_FAILURE_001.json", "P1_OUTPUT_ARGUMENT_FAILURE_001.json", "P0_FAILED_PROFILE_SOURCE_001.cjs", "P1_FAILED_COMMAND_SOURCE_001.cjs", "p0.log", "p1.log"])
    lock = output.parent / "coordination_001/exclusive_benchmark.lock"
    owner = read(lock.with_name("LOCK_OWNER.json"))
    with lock.open("r+b", buffering=0) as handle:
        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_acquired_released_owner_released"] = owner["status"] == "released" and bool(owner.get("lease_released_utc"))
    checks["actual_registered_resources_bounded_original_GPU_snapshots"] = all(read(output / f"{prefix}_GUARD.json")["elapsed_seconds"] < 180 and read(output / f"{prefix}_GUARD.json")["peak_tree_RSS_bytes"] < 4 * 1024**3 for prefix in ["p0", "p1", "p2"]) and all(row["usedMiB"] <= 12288 and row["freeMiB"] >= 2048 for row in [result["gpuBefore"], result["gpuAfter"]])
    state = read(DOC / "CURRENT_STATE.json")
    checks["parent34_unchanged_no_new_accepted_R163_or_diagnostic"] = state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == 34 and not any(row.startswith("R163") for row in state["accepted_mechanisms"])
    report = (DOC / "VIEWER_ORDERING_COST_006_RESULTS.md").read_text(encoding="utf-8")
    checks["direct_report_text_numbers_scope_failures_and_claim_boundaries"] = all(word in report for word in ["新增0", "181796864", "181731328", "138739712", "没有实际省去", "完整计量使用dispose前统计", "p0/p1", "ArrayBuffer transfer", "未实现前缀上传", "原GPU调用"])
    links = re.findall(r"\]\((C:/[^)]+)\)", report)
    checks["report_links_exist_except_own_closure_output"] = all(Path(name).is_file() or Path(name) == output / "SOURCE_COST_CLOSURE_001.json" for name in links)
    payload = {"status": "source_and_complete_native_cost_audit_closed0_pending_actual_self_writer_exit" if all(checks.values()) else "source_cost_audit_failed0",
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks, "measurements": measurements,
               "writer": {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "source": identity(Path(__file__))},
               "historical_owned_process_identity_count": len(rows), "executed_source_snapshot_count": len(snapshots),
               "known_actual_diagnostic_child_seconds": sum(row["child_elapsed_seconds"] for row in leases),
               "new_optimization_count": 0, "parent": 34, "target": 50, "remaining": 16,
               "root_self_review_is_independent_agent": False, "candidate_R163_implemented_or_executed": False,
               "new_training_production_service_GT_unknown_object_whole_chain": "not_run"}
    save(output / "SOURCE_COST_CLOSURE_001.json", payload)
    print({"checks": checks, "diagnostic_actual_seconds": payload["known_actual_diagnostic_child_seconds"]}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The source, original cost, text or ownership audit failed")


if __name__ == "__main__":
    main()
