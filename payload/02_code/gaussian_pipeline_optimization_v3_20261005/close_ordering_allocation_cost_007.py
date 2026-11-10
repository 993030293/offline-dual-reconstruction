"""Close the read-only allocation-request attribution and retain every failed observer branch."""

import datetime
import os
import shutil
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, identity, matches, read, save

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    began = time.perf_counter()
    result = read(OWN / "p3/RESULT.json")
    identities = list(result["sources"] + result["inputs"])
    snapshots, processes, fees = [], [], []
    for tag in ["p0", "p2"]:
        failure = read(OWN / tag / "FAILURE.json")
        identities.extend(failure["sources"] + failure["inputs"])
    unique = {(row["path"], row["sha256"]): row for row in identities}
    checks = {"all_current_normal_and_failed_sources_and_inputs_unchanged": all(matches(row) for row in unique.values()),
              "all9_original_complete_preview_checks_passed": len(result["checks"]) == 9 and all(result["checks"].values())}
    for tag in ["p0", "p2", "p3"]:
        for file in (OWN / tag / "sources").glob("*"):
            if file.name.startswith(("EXECUTED_", "COMPILED_")):
                continue
            original = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005" / file.name
            if not original.exists():
                continue
            left, right = identity(file), identity(original)
            if (left["bytes"], left["sha256"]) != (right["bytes"], right["sha256"]):
                raise AssertionError("A frozen observer or source snapshot changed")
            snapshots.append(left)
    failed_sources = OWN / "failure_sources_001"
    failed_sources.mkdir()
    for name in ["profile_ordering_allocation_cost_007_v2.cjs", "profile_ordering_allocation_cost_007_v3.cjs"]:
        original = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005" / name
        archive = failed_sources / name
        shutil.copyfile(original, archive)
        if identity(archive)["sha256"] != identity(original)["sha256"]:
            raise AssertionError("A failed source snapshot was not preserved")
        snapshots.append(identity(archive))
    checks["normal_and_failed_executed_source_archives_retained"] = bool(snapshots)
    for tag in ["p0", "p1", "p2", "p3"]:
        guard, lease = read(OWN / (tag + "_GUARD.json")), read(OWN / (tag + "_lease.json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != (0 if tag == "p3" else 1):
            raise AssertionError("An actual diagnostic or failure exit was not retained")
        processes.extend(guard["owned_process_identities"])
        processes.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                          {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
        fees.append({"scope": tag, "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
    checks["all24_previous_owned_births_ended_and_all4_leases_released"] = len(processes) == 24 and all(closed(row) for row in processes)
    measurements = []
    for episode in result["episodes"]:
        allocation = episode["disposed"]["orderingAllocation"]
        totals, rows, native = allocation["totals"], allocation["events"], episode["flight"]["after"]["totals"]
        if totals["actualNativeAllocationAtReadbackCalls"] != len(rows) or totals["verifiedNativeNewArrayBytes"] != sum(row["bytes"] for row in rows):
            raise AssertionError("The allocation-event request accounting changed")
        if totals["actualNativeAllocationAtReadbackCalls"] != native["readbackCalls"] or native["readbackCalls"] != native["sortStarts"]:
            raise AssertionError("The source-guaranteed actual readback boundary was not exhaustive")
        if totals["verifiedNativeNewArrayBytes"] - native["workerOrderingBytes"] != rows[0]["bytes"] or len(rows) - native["observedWorkerCalls"] != 1:
            raise AssertionError("The constructor-worker observation gap was not explicit and exact")
        if allocation["active"] or not allocation["disposed"] or not allocation["originalAliasRestored"] or totals["nativeErrors"]:
            raise AssertionError("An allocation-observer native alias or error contract changed")
        if not all(row["originalReadPause"] == 1 for row in rows) or allocation["retainedOriginalImageDataBytes"] != 2162688:
            raise AssertionError("The original pause or retained restoration image changed")
        measurements.append({"original_context_hooks_label": episode["hooks"], "native_allocation_request_calls": len(rows),
                             "native_allocation_request_bytes": totals["verifiedNativeNewArrayBytes"],
                             "maximum_requests": sum(row["bytes"] == totals["maximumAllocationBytes"] for row in rows),
                             "original_image_data_bytes": allocation["retainedOriginalImageDataBytes"],
                             "whole_observer_window_seconds_not_candidate_comparison": episode["complete_context_to_actual_close_ms"] / 1000})
    checks["all_complete_source_request_events_counts_bytes_and_native_worker_gap_exact"] = len(measurements) == 2
    checks["original_readPause_and_all_final_native_aliases_preserved"] = True
    native_source = (ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor/spark.module.js").read_text(encoding="utf-8")
    checks["native_Module_sharing_already_present_not_new_optimization"] = all(term in native_source for term in ["var WASM_MODULE = WebAssembly.compile(spark_rs_bg_default);", "WASM_MODULE.then((module)", 'name: "init-wasm"'])
    checks["first_wrong_phase_and_later_final_alias_failures_preserved"] = "native allocation dimensions" in str(read(OWN / "p0/FAILED_EPISODE_0.json")["errors"]) and len(read(OWN / "p2/FAILED_EPISODE_0.json")["completed_views"]) == 21
    current = read(DOC / "CURRENT_STATE.json")
    checks["parent_count_mechanisms_receipts_remain34_active_R163_and007_count0"] = current["additional_accepted_count"] == len(current["accepted_mechanisms"]) == len(current["acceptance_receipts"]) == 34 and current["status"] == "active_first_measured_algorithm_pass" and current["R163_new_optimization_count"] == 0
    report = (DOC / "VIEWER_ORDERING_ALLOCATION_COST_007_RESULTS.md").read_text(encoding="utf-8")
    checks["direct_report_text_zero_count_and_source_requests_not_performance_explicit"] = all(term in report for term in ["新增0", "175308800", "83次", "不是OS", "不能将差异当优化", "失败保留"])
    task = (DOC / "R164_ORDERING_SCRATCH_REUSE_TASK.md").read_text(encoding="utf-8")
    checks["next_task_preregistered_work10_whole_and_all_input_error_recovery_protection"] = all(term in task for term in ["任何R164实现", "全数组fill(0)", "净减少≥10%", "中位≤1.01", "2048MiB", "single chat/no subagent"])
    checks["no_R164_candidate_or_new_reconstruction_training_service_execution"] = not (OWN / "R164").exists()
    receipt = {"status": "read_only_complete_original_allocation_request_cost_closed0", "checks": checks,
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "measurements": measurements,
               "source_identity_count": len(unique), "source_snapshot_count": len(snapshots), "old_process_birth_count": len(processes),
               "known_all_actual_diagnostic_seconds": sum(row["seconds"] for row in fees), "fee_components": fees,
               "actual_self_review_seconds_before_write": time.perf_counter() - began, "not_total_research_cost_or_allocation_wall_time": True,
               "new_optimization_count": 0, "parent": 34, "remaining": 16, "target": 50, "goal_status": "active",
               "root_self_review_is_independent_agent": False,
               "writer": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()}}
    save(OWN / "SOURCE_COST_CLOSURE_001.json", receipt)
    print({"checks": checks, "measurements": measurements, "count": 0}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A separate source, cost, resource, or original-text review failed")


if __name__ == "__main__":
    main()
