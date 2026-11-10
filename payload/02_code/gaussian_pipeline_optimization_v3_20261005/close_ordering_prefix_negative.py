"""Review the failed whole-cost gate separately and close sources, fees, and owned processes."""

import datetime
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, identity, matches, read, save

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006"
OWN = BASE / "R163"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def pixels(capture):
    width, height = capture["dimensions"]
    return width * height * 4


def payload_fees():
    rows = [{"scope": "preregistered_unknown_historical_GPU_payload_allowance", "bytes": 2048 * 1024**2}]
    for directory in [BASE / "p2"] + [OWN / tag for tag in ["c0", "c1", "c2", "c4", "c5", "c6"]]:
        result = read(directory / "RESULT.json")
        for episode in result["episodes"]:
            captures = [episode["first"], episode["flight"]["capture"]] + [row["capture"] for row in episode["views"]]
            rows.append({"scope": str(directory) + "_" + str(episode["hooks"]) + "_actual_ordering_submission",
                         "bytes": episode["disposed"]["orderingUpload"]["totals"]["orderingSubmittedBytes"]})
            rows.append({"scope": str(directory) + "_" + str(episode["hooks"]) + "_all23_QA_GPU_read_payload",
                         "bytes": sum(map(pixels, captures))})
    generic = read(OWN / "g0/RESULT.json")
    rows.append({"scope": "g0_conservative_initialization_and_component_GPU_writes", "bytes": generic["extraGPUWriteBytes"]})
    rows.append({"scope": "g0_actual_component_GPU_reads", "bytes": generic["extraGPUReadBytes"]})
    for tag in ["x0", "x1", "x2"]:
        for file in sorted((OWN / tag).glob("CONTEXT_*.json")):
            result = read(file)
            captures = [result["restoredFirst"]["capture"]] + [row["capture"] for row in result["views"]]
            for name in ["initialCapture", "preLossPose0"]:
                if name in result:
                    captures.append(result[name]["capture"] if "capture" in result[name] else result[name])
            rows.append({"scope": str(file) + "_actual_ordering_submission",
                         "bytes": result["disposed"]["timing"]["orderingUpload"]["totals"]["orderingSubmittedBytes"]})
            rows.append({"scope": str(file) + "_all_QA_reads_plus_two_unstored_preloss_captures_conservative_max",
                         "bytes": sum(map(pixels, captures)) + 2 * max(map(pixels, captures))})
    rows.append({"scope": "g1_failed_and_g2_GL_handle_allocations_only_no_pixel_or_buffer_payload_API_in_executed_component",
                 "bytes": 0, "time_and_resource_cost_not_zero": True})
    return rows


def main():
    started = time.perf_counter()
    checks, identities, snapshots, processes, times = {}, [], [], [], []
    normal = {tag: read(OWN / tag / "RESULT.json") for tag in ["c0", "c1", "c2", "c4", "c5", "c6"]}
    normal["g0"] = read(OWN / "g0/RESULT.json")
    normal["g2"] = read(OWN / "g2/RESULT.json")
    normal["original006"] = read(BASE / "p2/RESULT.json")
    negative_contexts = [read(file) for tag in ["x0", "x1", "x2"] for file in (OWN / tag).glob("CONTEXT_*.json")]
    for result in normal.values():
        identities.extend(result["sources"] + result["inputs"])
    for tag in ["x0", "x1", "x2"]:
        failure = read(OWN / tag / "FAILURE.json")
        identities.extend(failure["sources"] + failure["inputs"])
    unique = {(row["path"], row["sha256"]): row for row in identities}
    checks["all_current_executed_sources_and_input_identities_exact"] = all(matches(row) for row in unique.values())
    for result_tag, result in normal.items():
        folder = BASE / "p2" if result_tag == "original006" else OWN / result_tag
        for row in result["sources"]:
            archive = folder / "sources" / Path(row["path"]).name
            if archive.exists():
                current = identity(archive)
                snapshots.append(current)
                if (current["bytes"], current["sha256"]) != (row["bytes"], row["sha256"]):
                    raise AssertionError("An executed source snapshot no longer matches its source identity")
    checks["all_available_executed_source_snapshots_exact"] = bool(snapshots)
    failed_sources = OWN / "failure_sources_001"
    failed_sources.mkdir()
    for name in ["run_ordering_prefix_complete_v2.cjs", "gl_resource_generation_generic_v2.js", "run_gl_resource_generation_generic_v2.cjs"]:
        source = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005" / name
        archive = failed_sources / name
        shutil.copyfile(source, archive)
        snapshots.append(identity(archive))
        if identity(source)["sha256"] != identity(archive)["sha256"]:
            raise AssertionError("A failed source archive copy was not preserved")
    for archive in (OWN / "g1/sources").glob("*"):
        source = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005" / archive.name
        if source.exists():
            if identity(source)["sha256"] != identity(archive)["sha256"]:
                raise AssertionError("The original failed GL component source changed")
            snapshots.append(identity(archive))
    checks["failed_GL_component_and_prebrowser_driver_sources_preserved"] = True
    for directory, tags in [(BASE, ["p0", "p1", "p2"]), (OWN, ["g0", "g1", "g2", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "x0", "x1", "x2"])]:
        for tag in tags:
            lease = read(directory / (tag + "_lease.json"))
            guard = read(directory / (tag + "_GUARD.json"))
            if not lease.get("lease_released_utc") or lease["status"] != "child_exited":
                raise AssertionError("An earlier GPU lease was not actually released")
            processes.extend(guard["owned_process_identities"])
            processes.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                              {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
            times.append({"scope": str(directory / (tag + "_lease.json")), "seconds": lease["child_elapsed_seconds"],
                          "actual_controller_returncode": lease["returncode"], "actual_owned_child_returncode": guard["returncode"]})
    checks["all_earlier_owned_process_births_ended_and_leases_released"] = all(closed(row) for row in processes)
    checks["all_g0_62_real_GL_and_g2_22_resource_and_error_contracts_passed"] = all(normal["g0"]["checks"].values()) and all(normal["g2"]["checks"].values()) and len(normal["g0"]["cases"]) == 62
    checks["V1_three_and_V2_three_complete_original_quality_pairs_preserved"] = all(all(normal[tag]["checks"].values()) for tag in ["c0", "c1", "c2", "c4", "c5", "c6"])
    checks["V2_shared_generation_prerequisite_and_all_aliases_restored_both_routes"] = all(episode["disposed"]["generation"]["originalAliasesRestored"] for tag in ["c4", "c5", "c6"] for episode in normal[tag]["episodes"])
    trace = read(OWN / "x2/CONTEXT_0.json")["disposed"]["generation"]
    checks["native_context_GL_failure_not_masked_and_three_recorded_stale_deletes"] = trace["enabled"] is False and trace["totals"]["knownStaleDeleteCalls"] == 3 and trace["totals"]["skippedKnownStaleDeletes"] == 0 and trace["totals"]["unknownDeleteCalls"] == 0
    checks["x0_x1_x2_failed_native_recovery_not_relabelled_success"] = all(not read(OWN / tag / "CONTEXT_0.json")["checks"]["native_GL_errors_and_page_errors_empty"] for tag in ["x0", "x1", "x2"])
    checks["g1_actual_guard_failure_and_c3_actual_prebrowser_binding_failure_retained"] = read(OWN / "g1_GUARD.json")["status"] == "failed_controller" and read(OWN / "c3_lease.json")["returncode"] == 1 and not (OWN / "c3/RESULT.json").exists()
    baseline, candidate, ratios, baseline_work, candidate_work = [], [], [], [], []
    for tag in ["c4", "c5", "c6"]:
        left = next(row for row in normal[tag]["episodes"] if not row["hooks"])
        right = next(row for row in normal[tag]["episodes"] if row["hooks"])
        baseline.append(left["complete_context_to_actual_close_ms"] / 1000)
        candidate.append(right["complete_context_to_actual_close_ms"] / 1000)
        ratios.append(candidate[-1] / baseline[-1])
        baseline_work.append(left["disposed"]["orderingUpload"]["totals"]["orderingSubmittedBytes"])
        candidate_work.append(right["disposed"]["orderingUpload"]["totals"]["orderingSubmittedBytes"])
    checks["registered_each_limit_passed_but_whole_median_above_1point01_rejected"] = max(ratios) <= 1.1 and statistics.median(ratios) > 1.01
    current = read(DOC / "CURRENT_STATE.json")
    checks["parent_count_and_accepted_mechanisms_and_receipts_still34_goal_active"] = current["additional_accepted_count"] == 34 and len(current["accepted_mechanisms"]) == len(current["acceptance_receipts"]) == 34 and current["status"] == "active" and current["R163_new_optimization_count"] == 0
    checks["new_lifecycle_public_and_full_recovery_not_executed_after_failed_guard"] = not any((OWN / tag / "RESULT.json").exists() for tag in ["public_v2", "lifecycle_v2", "context_v2", "fresh_recovery_v2"])
    task = (DOC / "R163_ORDERING_ACTIVE_PREFIX_TASK.md").read_text(encoding="utf-8")
    checks["original_text_work10_whole1point01_and_recovery_contracts_retained"] = "净工作减少必须≥10%" in task and "whole中位≤1.01" in task and "不能通过降低门槛跳过" in task
    report = (DOC / "R163_ORDERING_ACTIVE_PREFIX_RESULTS.md").read_text(encoding="utf-8")
    checks["result_text_failure_count0_unrun_and_no_speed_claim_explicit"] = all(term in report for term in ["计0", "1.415679", "未执行", "不是速度收益", "34/50"])
    fee_rows = payload_fees()
    B, C, fee = statistics.median(baseline_work), statistics.median(candidate_work), sum(row["bytes"] for row in fee_rows)
    elapsed = time.perf_counter() - started
    receipt = {"status": "closed_negative_registered_whole_cost_failed0", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "checks": checks, "formal_V2_baseline_seconds": baseline, "formal_V2_candidate_seconds": candidate,
               "raw_whole_ratios": ratios, "raw_whole_median_ratio": statistics.median(ratios),
               "complete_GPU_ordering_bytes_B": baseline_work, "complete_GPU_ordering_bytes_C": candidate_work,
               "raw_median_GPU_ordering_submission_reduction": (B - C) / B,
               "known_GPU_payload_fee_components": fee_rows, "known_GPU_payload_fee_bytes": fee,
               "net_work_upper_bound_before_unexecuted_required_recovery_and_lifecycle_fees": (B - C) / (B + fee / 6),
               "known_all_actual_job_seconds": sum(row["seconds"] for row in times), "actual_time_components": times,
               "additional_metadata_wall_conservative_allowance_seconds": 100.0, "metadata_allowance_is_not_total_research_time_or_GPU_bytes": True,
               "actual_self_review_seconds_before_write": elapsed, "no_speed_or_final_work_acceptance_claim": True,
               "source_identity_count": len(unique), "source_snapshot_count": len(snapshots), "old_process_birth_count": len(processes),
               "new_optimization_count": 0, "parent": 34, "target": 50, "remaining": 16, "goal_status": "active",
               "root_self_review_is_independent_agent": False,
               "writer": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()}}
    save(OWN / "NEGATIVE_SOURCE_COST_CLOSURE_001.json", receipt)
    print({"checks": checks, "raw_whole_median_ratio": receipt["raw_whole_median_ratio"], "count": 0}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A separate negative review, source, resource, or text check failed")


if __name__ == "__main__":
    main()
