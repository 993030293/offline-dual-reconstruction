"""Review complete scratch-request work, original behavior and actual lifecycle evidence separately."""

import datetime
import json
import os
import statistics
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, identity, matches, read, save

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164"
DIAG = OWN.parent
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def capture_bytes(capture):
    width, height = capture["dimensions"]
    return width * height * 4 + 32


def episode_capture_bytes(episode):
    return sum(capture_bytes(row) for row in [episode["first"], *[v["capture"] for v in episode["views"]], episode["flight"]["capture"]])


def scratch_requests(snapshot):
    totals = snapshot["totals"]
    return totals["newArrayRequestBytes"] + totals["oracleNewArrayRequestBytes"] + totals["retainedCopyNewArrayBytes"]


def main():
    began = time.perf_counter()
    tags = ["g0", "c0", "c1", "c2", "c3", "x0", "l0", "p0", "f0"]
    normal = {tag: read(OWN / tag / "RESULT.json") for tag in tags if tag != "c0"}
    failure = read(OWN / "c0/FAILURE.json")
    failed_episode = read(OWN / "c0/FAILED_EPISODE_0.json")
    sources, snapshots, processes, execution_fees, request_fees = [], [], [], [], []
    for tag, result in [*normal.items(), ("c0", failure)]:
        sources.extend(result["sources"] + result["inputs"])
        lease, guard = read(OWN / f"{tag}_lease.json"), read(OWN / f"{tag}_GUARD.json")
        if lease.get("returncode") != (1 if tag == "c0" else 0) or not lease.get("lease_released_utc"):
            raise AssertionError("A real measured process exit or failure lease is missing")
        if guard["peak_tree_RSS_bytes"] > 4 * 1024 ** 3 or guard["elapsed_seconds"] > 180:
            raise AssertionError("A registered owned-process resource bound failed")
        processes.extend(guard["owned_process_identities"])
        processes.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                          {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
        execution_fees.append({"scope": "R164_" + tag, "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
        for file in (OWN / tag / "sources").iterdir():
            if not file.is_file():
                continue
            if file.name == "SPARK-LICENSE.txt":
                native_license = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor/SPARK-LICENSE.txt"
                if file.read_text(encoding="utf-8") != native_license.read_text(encoding="utf-8"):
                    raise AssertionError("The directly read native MIT notice is incomplete")
                continue
            snapshot = identity(file)
            candidates = [row for row in result["sources"] if file.name == Path(row["path"]).name or file.name.endswith("_" + Path(row["path"]).name)]
            if candidates and not any(snapshot["sha256"] == row["sha256"] for row in candidates):
                raise AssertionError("An executed or failed source snapshot differs from its pinned source")
            snapshots.append(snapshot)
    unique = {(row["path"], row["sha256"]): row for row in sources}
    checks = {
        "all_current_native_vendor_model_sources_and_input_identities_exact": all(matches(row) for row in unique.values() if not row["path"].lower().endswith(".txt")),
        "all_actual_recorded_process_births_ended_and_all9_leases_released": all(closed(row) for row in processes),
        "all20_real_sort32_boundary_capacity_transfer_and_full_array_cases_passed": len(normal["g0"]["cases"]) == 20 and all(normal["g0"]["checks"].values()),
        "first_failed_worker_alias_binding_and_complete21_cost_preserved": len(failed_episode["completed_views"]) == 21 and "owned sort-worker hook changed" in str(failure),
        "all_three_NEW_complete_pairs_all9_original_quality_contracts_passed": all(len(normal[tag]["checks"]) == 9 and all(normal[tag]["checks"].values()) for tag in ["c1", "c2", "c3"]),
    }
    baseline, candidate, baseline_seconds, candidate_seconds, ratios = [], [], [], [], []
    for tag in ["c1", "c2", "c3"]:
        result = normal[tag]
        pair = [next(e for e in result["episodes"] if e["hooks"] == enabled) for enabled in [False, True]]
        for side, episode in enumerate(pair):
            snapshot = episode["disposed"]["orderingScratch"]
            totals, events = snapshot["totals"], snapshot["events"]
            if totals["allocationRequests"] != len(events) or totals["newArrayRequestBytes"] != sum(row["bytes"] for row in events if row["origin"] == "new"):
                raise AssertionError("Allocation-request source accounting is not exhaustive")
            if totals["newArrayRequests"] + totals["reusedRequests"] != totals["allocationRequests"] or not all(row["readPause"] == 1 for row in events):
                raise AssertionError("Native allocation count or the original pause changed")
            if totals["firstNativeWorkerCallsUnobserved"] != 1 or totals["fullOrderingComparisons"] != totals["allocationRequests"] - 1 or totals["fullReadbackComparisons"] != totals["fullOrderingComparisons"] or totals["activeSplatsComparisons"] != totals["fullOrderingComparisons"] or totals["fullZeroInputComparisons"] != totals["fullOrderingComparisons"]:
                raise AssertionError("The exhaustive real-worker comparison or explicit first-constructor native route changed")
            if not snapshot["fullOriginalMethodInverseExact"] or not snapshot["privateNativeConstructorBound"] or snapshot["active"] or snapshot["activeLease"] is not None or not all(snapshot[key] for key in ["originalSortAliasRestored", "originalReadbackAliasRestored", "originalWorkerAliasRestored"]):
                raise AssertionError("A complete native method, ownership or alias contract failed")
            (candidate if side else baseline).append(totals["newArrayRequestBytes"])
            (candidate_seconds if side else baseline_seconds).append(episode["complete_context_to_actual_close_ms"] / 1000)
            request_fees.append({"scope": tag + ("_C" if side else "_B") + "_validation_oracles_and_original_RGBA", "bytes": totals["oracleNewArrayRequestBytes"] + totals["retainedCopyNewArrayBytes"] + episode_capture_bytes(episode)})
        ratios.append(candidate_seconds[-1] / baseline_seconds[-1])
    checks["all_formal_full_zero_readback_sort_image_data_and_original_native_methods_exact"] = True
    checks["three_NEW_AB_BA_AB_orders_preserved"] = [normal[tag]["episodes"][0]["hooks"] for tag in ["c1", "c2", "c3"]] == [False, True, False]
    checks["no_reused_original_texture_data_or_changed_GPU_upload_arguments"] = all(
        e["disposed"]["orderingScratch"]["protectedOriginalImageDataCount"] == 2 and
        e["flight"]["after"]["orderingUpload"]["unmodifiedOriginalArguments"] and
        e["disposed"]["orderingScratch"]["totals"]["newArrayRequests"] == 4
        for tag in ["c1", "c2", "c3"] for e in normal[tag]["episodes"] if e["hooks"])
    native_module = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor/spark.module.js"
    native_text = native_module.read_text(encoding="utf-8")
    method = native_text.split("\tasync driveSort() {", 1)[1].split("\tensureLodWorker() {", 1)[0]
    allocation = "const ordering = new Uint32Array(this.maxSplats);"
    recycling = "this.sorting = false;\n\t\tthis.setDirty();"
    checks["native_method_allocation_and_recycle_sites_and_exported_dependencies_pinned"] = method.count(allocation) == 1 and method.count(recycling) == 1 and "uploadU32DataTextureRows: () => uploadU32DataTextureRows" in native_text and "Readback" in native_text.rsplit("export {", 1)[1]
    contexts = normal["x0"]["cases"]
    control = read(OWN / "x0/CONTEXT_NO_LOSS_REFERENCE.json")
    checks["same_state_no_loss_reference_and_both_actual_Gaussian_restorations_all8_checks"] = len(contexts) == 2 and all(len(c["checks"]) == 8 and all(c["checks"].values()) for c in [control, *contexts])
    checks["both_actual_loss_restore_full_original_image_data_and_first_stable_GL0"] = all(c["restoration"]["sameOriginalImageData"] and c["restoration"]["originalImageDataHashExact"] and c["restoredFirst"]["capture"]["glError"] == 0 and len(c["views"]) == 21 and c["disposed"]["timing"]["generation"]["totals"]["skippedKnownStaleDeletes"] == 3 for c in contexts)
    life = normal["l0"]["cases"]
    checks["two_actual_original_GL_callback_errors_forwarded_and_aliases_restored"] = len([c for c in life if c["kind"] == "fault"]) == 2 and all(c["observed"]["fault"]["calls"] == 2 and c["faultAliasRestored"] and c["errors"] for c in life if c["kind"] == "fault")
    checks["two_actual_native_worker_RPC_errors_forwarded_and_no_scratch_lease"] = len([c for c in life if c["kind"] == "workerFault"]) == 2 and all(c["observed"]["fault"]["calls"] == 1 and "Unknown worker RPC" in str(c["errors"]) and c["restored"]["timing"]["orderingScratch"]["activeLease"] is None for c in life if c["kind"] == "workerFault")
    checks["two_real_page_closes_cancelled_pending_continuous_W_waits"] = len([c for c in life if c["kind"] == "cancel"]) == 2 and all(c["actualPageClosed"] and not c["cancelled"]["resolved"] and c["actualOriginalWInputStillHeldAtClose"] for c in life if c["kind"] == "cancel")
    checks["fresh_original_complete_pair_after_faults_and_cancellations_all9_checks"] = all(normal["f0"]["checks"].values()) and all(e["views"] and len(e["views"]) == 21 for e in normal["f0"]["episodes"])
    public = normal["p0"]
    checks["current_original1016_public360_generic128_component_all1504_exact_not_GT"] = all(public["checks"].values()) and [r["denominator"] for r in public["values"]["summaries"]] == [1016, 360, 128] and all(r["denominator"] == r["exact"] for r in public["values"]["summaries"])
    request_fees.append({"scope": "g0_actual_native_and_sort32_oracle_arrays", "bytes": normal["g0"]["totalKnownNewArrayRequestBytes"] + capture_bytes(normal["g0"]["initial"])})
    request_fees.append({"scope": "c0_failed_alias_binding_original_arrays_oracles_and_32_capture_upper", "bytes": scratch_requests(failed_episode["actualState"]["orderingScratch"]) + 32 * capture_bytes(failed_episode["completed_views"][-1]["capture"]), "RGBA_count_is_conservative_source_bound_not_claimed_observed": True})
    for index, case in enumerate([control, *contexts]):
        request_fees.append({"scope": "x0_" + str(index) + "_all_original_arrays_oracles_and26_RGBA", "bytes": scratch_requests(case["disposed"]["timing"]["orderingScratch"]) + 26 * capture_bytes(case["initialCapture"]) + 64})
    for case in life:
        observed = case["restored"]["timing"]["orderingScratch"] if "restored" in case else case["observed"]["orderingScratch"]
        # Closed continuous views can complete another native request before process destruction.
        bound = 16 * 540672 * 4 if case["kind"] == "cancel" else 0
        request_fees.append({"scope": "l0_" + case["kind"] + "_" + str(case["enabled"]), "bytes": scratch_requests(observed) + sum(capture_bytes(v["capture"]) for v in case["completedViews"]) + bound, "cancel_tail_is_conservative16_complete_large_requests_each": case["kind"] == "cancel"})
    request_fees.append({"scope": "p0_native_preview_arrays_and_oracles", "bytes": scratch_requests(public["disposed"]["timing"]["orderingScratch"])})
    for e in normal["f0"]["episodes"]:
        request_fees.append({"scope": "f0_full_recovery_" + str(e["hooks"]), "bytes": scratch_requests(e["disposed"]["orderingScratch"]) + episode_capture_bytes(e)})
    diagnostic = read(DIAG / "p3/RESULT.json")
    for e in diagnostic["episodes"]:
        request_fees.append({"scope": "007_p3_original_" + str(e["hooks"]), "bytes": e["disposed"]["orderingAllocation"]["totals"]["verifiedNativeNewArrayBytes"] + episode_capture_bytes(e)})
    p2 = read(DIAG / "p2/FAILED_EPISODE_0.json")
    request_fees.append({"scope": "007_p2_failed_final_alias_complete_original_arrays_and32_RGBA_upper", "bytes": p2["actualState"]["orderingAllocation"]["totals"]["verifiedNativeNewArrayBytes"] + 32 * capture_bytes(p2["completed_views"][-1]["capture"])})
    request_fees.append({"scope": "preregistered_unknown_earlier007_and_shared_generation_development_array_work_allowance", "bytes": 2048 * 1024 ** 2, "not_claimed_actual_historical_array_counter": True})
    allocation_fee = sum(row["bytes"] for row in request_fees)
    surcharge = allocation_fee / 6
    raw_b, raw_c = statistics.median(baseline), statistics.median(candidate)
    net_work = (raw_b - raw_c) / (raw_b + surcharge)
    prior007 = read(DIAG / "SOURCE_COST_CLOSURE_002.json")
    prior163 = read(DIAG.parent / "oup006/R163/NEGATIVE_SOURCE_COST_CLOSURE_002.json")
    # Count every declared old execution fee again, conservatively, for the time guard only.
    old_seconds = 16.129845500108786 + 257.50154079985805 + 100.0
    execution_fees.append({"scope": "all_closed007_and_R163_shared_prerequisite_execution_and_prior100_metadata_allowance", "seconds": old_seconds, "source_receipts": [str(DIAG / "SOURCE_COST_CLOSURE_002.json"), str(DIAG.parent / "oup006/R163/NEGATIVE_SOURCE_COST_CLOSURE_002.json")], "old_R163_negative_sort_trials_conservatively_included0": True})
    execution_fees.append({"scope": "current_known_text_KB_and_root_post_exit_execution_allowance", "seconds": 100.0, "not_total_unmeasured_research_ROI": True})
    seconds_fee = sum(row["seconds"] for row in execution_fees) + time.perf_counter() - began
    adjusted_ratios = [(c + seconds_fee / 6) / (b + seconds_fee / 6) for b, c in zip(baseline_seconds, candidate_seconds)]
    checks["registered_net_complete_array_request_work_at_least10percent"] = net_work >= .1
    checks["registered_raw_whole_median_at_most1point01_each_at_most1point1"] = statistics.median(ratios) <= 1.01 and max(ratios) <= 1.1
    checks["registered_symmetric_all_fee_whole_median_and_each_kept"] = statistics.median(adjusted_ratios) <= 1.01 and max(adjusted_ratios) <= 1.1
    checks["historical_negative_source_cost_closures_present_without_reacceptance"] = prior007["new_optimization_count"] == 0 and prior163["new_optimization_count"] == 0
    master = read(DOC / "CURRENT_STATE.json")
    checks["parent_still_active34_unique_until_actual_review_and_publication_exit"] = master["additional_accepted_count"] == len(master["accepted_mechanisms"]) == len(set(master["accepted_mechanisms"])) == 34 and master["status"] == "active_first_measured_algorithm_pass"
    checks["all_original_files_and_models_not_production_edits_output_bounded"] = sum(p.stat().st_size for p in OWN.rglob("*") if p.is_file()) < 2 * 1024 ** 3 and psutil.disk_usage(ROOT).free >= 20 * 1024 ** 3
    result = {"status": "separate_root_review_passed0_pending_actual_review_exit_and_final_publication" if all(checks.values()) else "separate_root_review_failed0",
              "checks": checks, "review_kind": "separate_root_self_review_not_independent_agent", "sources": list(unique.values()),
              "source_snapshots": snapshots, "ended_historical_process_births": processes,
              "array_work_cost": {"fee_components": request_fees, "all_known_and_registered_allowance_bytes": allocation_fee,
                                  "six_side_common_surcharge_bytes": surcharge, "raw_new_ordering_requests_B_C_bytes": [raw_b, raw_c],
                                  "raw_request_work_reduction": 1 - raw_c / raw_b, "net_request_work_reduction": net_work,
                                  "scope": "complete_preview_new_ordering_Uint32_scratch_requests_not_OS_memory_RSS_GPU_FPS_or_total_CPU_work"},
              "execution_time": {"fee_components": execution_fees, "known_execution_and_registered_allowance_seconds": seconds_fee,
                                 "formal_B_seconds": baseline_seconds, "formal_C_seconds": candidate_seconds,
                                 "raw_whole_ratios": ratios, "raw_whole_median_ratio": statistics.median(ratios),
                                 "adjusted_whole_ratios": adjusted_ratios, "adjusted_whole_median_ratio": statistics.median(adjusted_ratios),
                                 "net_speed_reduction": (statistics.median(baseline_seconds) - statistics.median(candidate_seconds)) / (statistics.median(baseline_seconds) + seconds_fee / 6),
                                 "speed_optimization_accepted": False, "unmeasured_research_text_time_not_zero_no_total_ROI": True},
              "review_process": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "new_optimization_count": 0,
              "full_unknown_object_GT_cross_hardware_or_training_verified": False, "production_or_services_modified": False}
    save(OWN / "SOURCE_COST_REVIEW_001.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "net_work": net_work, "net_speed": result["execution_time"]["net_speed_reduction"], "cost_bytes": allocation_fee, "source_ids": len(unique), "birth_records": len(processes)}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
