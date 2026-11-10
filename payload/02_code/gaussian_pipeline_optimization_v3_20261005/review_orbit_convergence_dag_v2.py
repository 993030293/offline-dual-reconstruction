"""Separately review V2 navigation, complete preview, recovery and all historical fees."""

import datetime
import os
import statistics
import time
from pathlib import Path

import psutil

from review_orbit_convergence_dag import DIAG, DOC, OWN, ROOT, closed, identity, matches, read, save
from review_orbit_convergence_dag import cost as historical_cost


V1 = ["public_inputs", "g0", "g1", "c0", "c1", "c2", "r0", "review", "review2", "review3", "review4", "publisher"]
V2 = ["g2", "g3", "g4", "c3", "c4", "c5", "r1"]
EXACT = ["rgba_sha256", "position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"]


def cost(extra_seconds=0.0):
    components = historical_cost()["components"]
    components = [row for row in components if row["scope"] != "actual_separate_review_publisher_and_following_tail_before_decision"]
    for prefix in ["review4", "publisher"] + V2:
        path = OWN / f"{prefix}_lease.json"
        lease = read(path)
        if not lease.get("lease_released_utc"):
            raise AssertionError("An executed historical or V2 fee has no actual lease release")
        components.append({"scope": "V1_" + prefix if prefix in ["review4", "publisher"] else "V2_" + prefix,
                           "seconds": lease["child_elapsed_seconds"], "receipt": str(path)})
    components.extend([
        {"scope": "V2_metadata_conservative_allowance_including_known_actual_query_and_post_exit_commands", "seconds": 15.0,
         "known_latest_measured_subset_seconds": sum([.2305321, .1567988, .2018975, .2131485, .2000508, .2739511, .257842, .214662, .2303793]),
         "not_total_text_research_time": True},
        {"scope": "V2_actual_separate_review_publisher_and_finalizer_seconds", "seconds": float(extra_seconds)},
    ])
    native, candidate = [], []
    for prefix in ["c3", "c4", "c5"]:
        episodes = read(OWN / prefix / "RESULT.json")["episodes"]
        native.append(next(row for row in episodes if not row["hooks"])["complete_context_to_actual_close_ms"] / 1000)
        candidate.append(next(row for row in episodes if row["hooks"])["complete_context_to_actual_close_ms"] / 1000)
    total = sum(row["seconds"] for row in components)
    surcharge = total / 6
    baseline, optimized = statistics.median(native), statistics.median(candidate)
    raw_ratios = [right / left for left, right in zip(native, candidate)]
    adjusted_ratios = [(right + surcharge) / (left + surcharge) for left, right in zip(native, candidate)]
    return {"components": components, "known_extra_execution_seconds_with_declared_allowances": total,
            "six_side_common_seconds_surcharge": surcharge, "formal_native_seconds": native,
            "formal_candidate_seconds": candidate, "formal_raw_median_seconds": {"native": baseline, "candidate": optimized},
            "raw_complete_stage_reduction": 1 - optimized / baseline,
            "net_complete_stage_reduction_under_registered_six_side_model": (baseline - optimized) / (baseline + surcharge),
            "raw_whole_median_ratio": statistics.median(raw_ratios), "each_original_whole_ratios": raw_ratios,
            "adjusted_whole_median_ratio": statistics.median(adjusted_ratios), "each_adjusted_whole_ratios": adjusted_ratios,
            "scope": "V2_complete_registered_preview_context_initial_all21_fixed_views_native_flight_to_actual_close_not_full_reconstruction",
            "V1_results_and_provisional_publication_fees_included_not_reused_as_fresh_V2": True,
            "old_unrecorded_text_research_time_not_zero_no_total_research_ROI": True}


def fee_passes(fees):
    return (fees["net_complete_stage_reduction_under_registered_six_side_model"] >= .02
            and fees["raw_whole_median_ratio"] <= 1.01 and fees["adjusted_whole_median_ratio"] <= 1.01
            and max(fees["each_original_whole_ratios"] + fees["each_adjusted_whole_ratios"]) <= 1.1)


def main():
    started = time.perf_counter()
    reviewer = {"pid": os.getpid(), "creation_time": psutil.Process().create_time(),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source": identity(Path(__file__))}
    checks = {}
    task = (DOC / "R162_V2_NAVIGATION_INTENT_TASK.md").read_text(encoding="utf-8")
    checks["preregistered_V2_navigation_lifecycle_and_all_old_fees_same_family_maximum1"] = all(word in task for word in ["净2%", "V1费用全部计入", "三", "cancelDiscrete", "单chat", "同族最多1"])
    g3, g4 = read(OWN / "g3/RESULT.json"), read(OWN / "g4/RESULT.json")
    checks["complete1016_public360_generic128_all1504_marker_and_first_actual_inputs_exact"] = all(g3["checks"].values()) and [row["denominator"] for row in g3["summaries"]] == [1016, 360, 128] and all(row["exact"] == row["denominator"] for row in g3["summaries"])
    checks["all5_overwritten_cancelled_intent_single_original_updates_exact"] = g3["checks"]["all5_cancelled_or_overwritten_intent_routes_original_single_update_exact"]
    checks["all11_structural_unsupported_and186_cycles_bounds_original_routes_retained"] = g3["unsupportedExact"] == 11 and g3["routes"] == {"converged": 1318, "cycle": 41, "bound": 145}
    checks["real_DOM_play_navigation_control_start_fraction_cancels_before_next_RAF_and_new_fixed_RGBA_exact"] = all(g4["checks"].values()) and len(g4["native"]["cases"]) == len(g4["candidate"]["cases"]) == 4
    results = [read(OWN / prefix / "RESULT.json") for prefix in ["c3", "c4", "c5", "r1"]]
    checks["three_new_fresh_complete_pairs_and_fresh_full_recovery_all_original_checks"] = all(all(result["checks"].values()) for result in results)
    protocols = [read(OWN / prefix / "PROTOCOL.json") for prefix in ["c3", "c4", "c5"]]
    checks["fresh_AB_BA_AB_orders_symmetric_original_hooks_and_original_viewport"] = [row["registered_context_orders"] for row in protocols] == [["native", "candidate"], ["candidate", "native"], ["native", "candidate"]] and all(row["symmetric_original_cost_hooks"] and row["viewport"] == [960, 600] and row["deviceScaleFactor"] == 1 for row in protocols)
    reference = next(row for row in results[0]["episodes"] if not row["hooks"])
    exact = True
    for result in results:
        for episode in result["episodes"]:
            exact &= len(episode["views"]) == 21 and all(episode["first"][key] == reference["first"][key] for key in EXACT)
            for view, expected in zip(episode["views"], reference["views"]):
                exact &= view["name"] == expected["name"] and all(view["capture"][key] == expected["capture"][key] for key in EXACT)
    checks["all8_fresh_contexts_initial_and21_RGBA_full_camera_and_populations_exact"] = bool(exact)
    manifest = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/baseline/ASSET_MANIFEST.json")
    populations = [manifest["scene"]["numSplats"], manifest["box"]["numSplats"]]
    checks["original_lazy_initial_scene_only_then_loaded_both_full_populations_preserved"] = reference["first"]["count"] == [populations[0], 0] and next(row for row in reference["views"] if row["name"] == "switch_box")["capture"]["count"] == populations
    checks["complete_original1016_path_not_reduced"] = len(read(ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/baseline/cases/packed/camera_path.json")) == 1016
    episodes = [episode for result in results for episode in result["episodes"]]
    checks["actual_original_hardware_pixels_and_nonempty_views"] = all("RTX 5060 Ti" in episode["hardware"]["renderer"] and all(view["capture"]["glError"] == 0 for view in episode["views"]) and any(view["capture"]["nonuniformPixels"] for view in episode["views"]) for episode in episodes)
    checks["only7_discrete_updates_aliases_unchanged_drained_no_pending_intent"] = all(episode["orbitDag"]["nativeUpdateUnchanged"] and episode["orbitDag"]["disposed"] and not episode["orbitDag"]["pending"] and episode["orbitDag"]["converged"] == (7 if episode["hooks"] else 0) for episode in episodes)
    events = read(OWN / "r1/RECOVERY_EVENTS.json")["recoveryEvidence"]
    faults, cancellations = ([row for row in events if row["kind"] == kind] for kind in ["fault", "cancel"])
    checks["two_actual_original_callback_errors_original_UI_and_restored_aliases"] = len(faults) == 2 and all(row["observed"]["fault"]["calls"] >= 2 and row["restored"]["nativeFaultAliasRestored"] and row["restored"]["timing"]["active"] == 0 and row["actualContextClosed"] for row in faults)
    checks["two_actual_page_closes_cancel_unfinished_views_after_real_frame_progress"] = len(cancellations) == 2 and all(len(row["completedViews"]) == 2 and row["observed"]["snapshot"]["frame"] > row["before"]["frame"] and not row["originalPendingWait"]["resolved"] and row["actualPageClosed"] and row["actualContextClosed"] for row in cancellations)
    all_identities, snapshots = [], []
    for prefix in ["g0", "g1", "c0", "c1", "c2", "r0", "g3", "g4", "c3", "c4", "c5", "r1"]:
        result = read(OWN / prefix / "RESULT.json")
        all_identities.extend(result["inputs"] + result["sources"])
        snapshots.extend({"path": str(OWN / prefix / "sources" / Path(row["path"]).name), "bytes": row["bytes"], "sha256": row["sha256"]} for row in result["sources"])
    public = read(OWN / "public_inputs001/RESULT.json")
    all_identities.extend(public["inputs"] + [public["source"]])
    checks["all_V1_V2_current_original_data_vendor_models_and_executed_sources_unchanged"] = all(matches(row) for row in all_identities)
    checks["all_V1_V2_executed_source_snapshots_match_recorded_bytes"] = all(matches(row) for row in snapshots)
    checks["every_new_complete_executed_viewer_has_V2_cancel_and_actual_V2_helper_identity"] = all("if (useOrbitDag) orbitDag.cancelDiscrete();" in (OWN / prefix / "sources/EXECUTED_VIEWER.js").read_text(encoding="utf-8") and any(Path(row["path"]).name == "orbit_convergence_dag_v2.js" for row in read(OWN / prefix / "RESULT.json")["sources"]) for prefix in ["c3", "c4", "c5", "r1"])
    process_rows, leases, guards = [], [], []
    for folder, prefixes in [(DIAG, ["p0", "p1", "p2"]), (OWN, V1 + V2)]:
        for prefix in prefixes:
            guard, lease = read(folder / f"{prefix}_GUARD.json"), read(folder / f"{prefix}_lease.json")
            process_rows.extend(guard.get("owned_process_identities", []))
            process_rows.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]}, {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
            leases.append(lease); guards.append(guard)
    checks["all_saved_historical_and_new_PID_births_ended_all_leases_released"] = all(closed(row) for row in process_rows) and all(row.get("lease_released_utc") for row in leases)
    checks["all_registered_Windows_guards_below4GiB180s"] = all(guard["peak_tree_RSS_bytes"] < 4 * 1024**3 and guard["elapsed_seconds"] < 180 for guard in guards)
    checks["original_GPU_whole_device_snapshots_below12GiB_and_free2GiB"] = all(snapshot["usedMiB"] <= 12 * 1024 and snapshot["freeMiB"] >= 2 * 1024 for result in results for snapshot in [result["gpuBefore"], result["gpuAfter"]])
    checks["private_output_below2GiB_disk_reserve20GiB"] = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) < 2 * 1024**3 and psutil.disk_usage(str(ROOT)).free >= 20 * 1024**3
    failed_files = ["REVIEW_READER_FAILURE_001.json", "REVIEW_WINDOWS_ARGUMENT_FAILURE_002.json", "REVIEW_LAZY_DENOMINATOR_FAILURE_003.json", "V2_G2_WRAPPER_FAILURE_001.json"]
    checks["all_original_failed_branches_and_V1_publication_kept_as_history"] = all((OWN / name).is_file() for name in failed_files) and read(OWN / "ACCEPTANCE_V6_001.json")["parent_after"] == 34 and read(OWN / "g2_lease.json")["returncode"] == 1
    state = read(DOC / "CURRENT_STATE.json")
    checks["current_parent33_and_R162_V1_retracted_count0_no_duplicate"] = state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == 33 and not any(row.startswith("R162") for row in state["accepted_mechanisms"]) and state["R162_new_optimization_count"] == 0 and state["R162_V1_late_navigation_boundary_not_fixed_by_lower_speed_gate"]
    fees = cost()
    checks["V6_net2percent_and_raw_adjusted_whole_guards_all_V1_V2_fees"] = fee_passes(fees)
    payload = {"status": "V2_separate_root_self_review_passed_pending_actual_exit_and_publication0" if all(checks.values()) else "V2_failed_root_self_review0",
               "checks": checks, "check_count": len(checks), "reviewer": reviewer, "review_is_independent_agent": False,
               "identity_rows": len(all_identities), "source_snapshot_count": len(snapshots), "process_identity_count": len(process_rows),
               "cost_before_review_exit": fees, "V1_retraction_and_failed_sources_costs_preserved": True,
               "new_optimization_count": 0, "elapsed_seconds_before_actual_exit": time.perf_counter() - started}
    save(OWN / "SELF_REVIEW_V2_V6_001.json", payload)
    print({"checks": checks, "net": fees["net_complete_stage_reduction_under_registered_six_side_model"]}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The separate V2 root review failed an original or navigation lifecycle gate")


if __name__ == "__main__":
    main()
