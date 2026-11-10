"""Separately review actual complete preview evidence without changing the algorithm."""

import datetime
import hashlib
import json
import msvcrt
import os
import statistics
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001"
DIAG = OWN.parent / "viewer_cost_005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def matches(row):
    current = identity(row["path"])
    return row["bytes"] == current["bytes"] and row["sha256"] == current["sha256"]


def closed(row):
    try:
        return psutil.Process(int(row["pid"])).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def cost(extra_seconds=0.0):
    components = []
    for prefix in ["p0", "p1", "p2"]:
        receipt = read(DIAG / f"{prefix}_lease.json")
        components.append({"scope": "original_diagnostic_" + prefix, "seconds": receipt["child_elapsed_seconds"], "receipt": str(DIAG / f"{prefix}_lease.json")})
    for prefix in ["public_inputs", "g0", "g1", "c0", "c1", "c2", "r0"]:
        receipt = read(OWN / f"{prefix}_lease.json")
        components.append({"scope": prefix, "seconds": receipt["child_elapsed_seconds"], "receipt": str(OWN / f"{prefix}_lease.json")})
    for name in ["SOURCE_AUDIT_LAUNCH_FAILURE_001.json", "SOURCE_AUDIT_PARSE_FAILURE_002.json"]:
        receipt = read(DIAG / name)
        components.append({"scope": name, "seconds": receipt["actual_tool_wall_seconds"], "receipt": str(DIAG / name)})
    for name in ["review", "review2", "review3"]:
        if (OWN / f"{name}_lease.json").exists():
            failed_review = read(OWN / f"{name}_lease.json")
            if failed_review.get("returncode") != 1 or not failed_review.get("lease_released_utc"):
                raise AssertionError("The failed reviewer fee or actual close was not preserved")
            components.append({"scope": "failed_" + name, "seconds": failed_review["child_elapsed_seconds"], "receipt": str(OWN / f"{name}_lease.json")})
    components.extend([
        {"scope": "successful_source_audit_tool_wall_conservative_upper_from_observed_subsecond_return", "seconds": 1.0, "actual_tool_chunk": "e4bd34", "guard_seconds_included": read(DIAG / "source_audit_003_GUARD.json")["elapsed_seconds"]},
        {"scope": "current_KB_and_dedup_query_batch_wall_including_b39505", "seconds": 4.5},
        {"scope": "known_other_metadata_commands_individual_elapsed_conservative_sum", "seconds": sum([.2709493, .263977, .1681856, .2225621, .2152771, .2196263, .2395118, .2489499, .3680989, .2741738, .203844, .3513363, .3951817, .3671412, .2869947, .3398492, .1818731, .693257, .3190219, .2347, .2427317, .407972, .284848, .7303, .1690853, .2655372, .2506756, .3269314, .2403804, .2242016])},
        {"scope": "registered_metadata_allowance_not_total_measured_research_time", "seconds": 15.0},
        {"scope": "registered_post_exit_allowance_updated_if_actual_larger", "seconds": 5.0},
        {"scope": "actual_separate_review_publisher_and_following_tail_before_decision", "seconds": float(extra_seconds)},
    ])
    native, candidate, paired_ratios = [], [], []
    for prefix in ["c0", "c1", "c2"]:
        episodes = read(OWN / prefix / "RESULT.json")["episodes"]
        left = next(row for row in episodes if not row["hooks"])["complete_context_to_actual_close_ms"] / 1000
        right = next(row for row in episodes if row["hooks"])["complete_context_to_actual_close_ms"] / 1000
        native.append(left)
        candidate.append(right)
        paired_ratios.append(right / left)
    total = sum(row["seconds"] for row in components)
    surcharge = total / 6
    baseline = statistics.median(native)
    optimized = statistics.median(candidate)
    adjusted_ratios = [(right + surcharge) / (left + surcharge) for left, right in zip(native, candidate)]
    return {"components": components, "known_extra_execution_seconds_with_declared_allowances": total,
            "six_side_common_seconds_surcharge": surcharge,
            "formal_native_seconds": native, "formal_candidate_seconds": candidate,
            "formal_raw_median_seconds": {"native": baseline, "candidate": optimized},
            "raw_complete_stage_reduction": 1 - optimized / baseline,
            "net_complete_stage_reduction_under_registered_six_side_model": (baseline - optimized) / (baseline + surcharge),
            "raw_whole_median_ratio": statistics.median(paired_ratios),
            "each_original_whole_ratios": paired_ratios,
            "adjusted_whole_median_ratio": statistics.median(adjusted_ratios),
            "each_adjusted_whole_ratios": adjusted_ratios,
            "scope": "complete_registered_preview_context_all21_fixed_views_and_native_flight_to_actual_close_not_full_reconstruction",
            "old_unrecorded_text_research_time_not_zero_no_total_research_ROI": True}


def main():
    started = time.perf_counter()
    reviewer = {"pid": os.getpid(), "creation_time": psutil.Process().create_time(),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": identity(Path(__file__))}
    checks = {}
    task = (DOC / "R162_ORBIT_CONVERGENCE_DAG_TASK.md").read_text(encoding="utf-8")
    checks["preregistered_complete2percent_and_whole_guards_and_all_failures"] = all(value in task for value in ["净2%", "≤1.01", "≤1.1", "4096", "39.8992747", "17.2310101", "17.740855", "360", "1016"])
    g0, g1 = read(OWN / "g0/RESULT.json"), read(OWN / "g1/RESULT.json")
    checks["generic_full1016_public360_generic128_and_all11_unsupported_exact"] = all(g0["checks"].values()) and [row["denominator"] for row in g0["summaries"]] == [1016, 360, 128] and all(row["exact"] == row["denominator"] for row in g0["summaries"])
    checks["separate_supplement_all1504_marker_and_actual_preflight_inputs_unchanged"] = all(g1["checks"].values()) and g1["checks"]["all1504_marker_inputs_unchanged"] and g1["checks"]["all1504_original_inputs_unchanged_before_first_actual_update"]
    checks["unsupported186_not_removed_or_claimed_converged"] = g1["routes"] == {"converged": 1318, "cycle": 41, "bound": 145}
    results = [read(OWN / prefix / "RESULT.json") for prefix in ["c0", "c1", "c2", "r0"]]
    checks["all_three_fresh_complete_pairs_and_fresh_recovery_quality_checks"] = all(all(row["checks"].values()) for row in results)
    orders = [read(OWN / prefix / "PROTOCOL.json")["registered_context_orders"] for prefix in ["c0", "c1", "c2"]]
    checks["fresh_AB_BA_AB_orders_and_symmetric_native_hooks"] = orders == [["native", "candidate"], ["candidate", "native"], ["native", "candidate"]] and all(read(OWN / prefix / "PROTOCOL.json")["symmetric_original_cost_hooks"] for prefix in ["c0", "c1", "c2"])
    keys = ["rgba_sha256", "position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"]
    reference = next(row for row in results[0]["episodes"] if not row["hooks"])
    exact = True
    for result in results:
        for episode in result["episodes"]:
            exact &= len(episode["views"]) == 21
            exact &= all(episode["first"][key] == reference["first"][key] for key in keys)
            for view, expected in zip(episode["views"], reference["views"]):
                exact &= view["name"] == expected["name"] and all(view["capture"][key] == expected["capture"][key] for key in keys)
    checks["all_four_native_candidate_initial_and21_full_camera_RGBA_populations_exact"] = bool(exact)
    manifest = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/baseline/ASSET_MANIFEST.json")
    populations = [manifest["scene"]["numSplats"], manifest["box"]["numSplats"]]
    checks["original_lazy_initial_scene_only_and_complete_loaded_populations_match_manifest"] = reference["first"]["count"] == [populations[0], 0] and next(row for row in reference["views"] if row["name"] == "switch_box")["capture"]["count"] == populations
    checks["original_complete_path_denominator1016_not_reduced"] = len(read(ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/baseline/cases/packed/camera_path.json")) == 1016
    checks["actual_hardware_and_fixed_original_pixel_grids_and_nonempty_views"] = all(episode["hardware"]["renderer"].find("RTX 5060 Ti") >= 0 and any(view["capture"]["nonuniformPixels"] for view in episode["views"]) for result in results for episode in result["episodes"])
    checks["candidate_only_discrete_idle_routes_original_aliases_restored"] = all(episode["orbitDag"]["nativeUpdateUnchanged"] and episode["orbitDag"]["disposed"] and not episode["orbitDag"]["pending"] and episode["orbitDag"]["converged"] == (7 if episode["hooks"] else 0) for result in results for episode in result["episodes"])
    events = read(OWN / "r0/RECOVERY_EVENTS.json")["recoveryEvidence"]
    faults = [row for row in events if row["kind"] == "fault"]
    cancellations = [row for row in events if row["kind"] == "cancel"]
    checks["two_actual_native_callback_errors_original_UI_and_aliases_restored"] = len(faults) == 2 and all(row["observed"]["fault"]["calls"] >= 2 and row["restored"]["nativeFaultAliasRestored"] and row["restored"]["timing"]["active"] == 0 for row in faults)
    checks["two_real_page_closes_after_partial_views_cancelled_unfinished_waits"] = len(cancellations) == 2 and all(len(row["completedViews"]) == 2 and row["observed"]["snapshot"]["frame"] > row["before"]["frame"] and not row["originalPendingWait"]["resolved"] and row["actualPageClosed"] and row["actualContextClosed"] for row in cancellations)
    all_identities = []
    for result in results + [g0, g1, read(DIAG / "p2/RESULT.json")]:
        all_identities.extend(result["inputs"])
        all_identities.extend(result["sources"])
    public = read(OWN / "public_inputs001/RESULT.json")
    all_identities.extend(public["inputs"] + [public["source"]])
    checks["all_original_current_data_assets_vendor_and_executed_sources_unchanged"] = all(matches(row) for row in all_identities)
    source_copies = []
    for prefix in ["g0", "g1", "c0", "c1", "c2", "r0"]:
        result = read(OWN / prefix / "RESULT.json")
        for row in result["sources"]:
            archive = OWN / prefix / "sources" / Path(row["path"]).name
            source_copies.append({"path": str(archive), "bytes": row["bytes"], "sha256": row["sha256"]})
    checks["every_executed_source_snapshot_matches_recorded_source_bytes"] = all(matches(row) for row in source_copies)
    process_rows, leases = [], []
    for folder, prefixes in [(DIAG, ["p0", "p1", "p2"]), (OWN, ["public_inputs", "g0", "g1", "c0", "c1", "c2", "r0"])]:
        for prefix in prefixes:
            guard = read(folder / f"{prefix}_GUARD.json")
            lease = read(folder / f"{prefix}_lease.json")
            process_rows.extend(guard.get("owned_process_identities", []))
            process_rows.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]}, {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
            leases.append(lease)
    for prefix in ["002", "003"]:
        process_rows.extend(read(DIAG / f"source_audit_{prefix}_GUARD.json").get("owned_process_identities", []))
    checks["all_saved_owned_PID_births_ended_and_all_leases_released"] = all(closed(row) for row in process_rows) and all(row.get("lease_released_utc") for row in leases)
    checks["registered_resources_all_bounded_guards_below4GiB_180s"] = all(read(folder / f"{prefix}_GUARD.json")["peak_tree_RSS_bytes"] < 4 * 1024**3 and read(folder / f"{prefix}_GUARD.json")["elapsed_seconds"] < 180 for folder, prefixes in [(DIAG, ["p0", "p1", "p2"]), (OWN, ["public_inputs", "g0", "g1", "c0", "c1", "c2", "r0"])] for prefix in prefixes)
    checks["private_output_disk_size_below2GiB_each_and_reserve20GiB"] = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) < 2 * 1024**3 and psutil.disk_usage(str(ROOT)).free >= 20 * 1024**3
    current = read(DOC / "CURRENT_STATE.json")
    checks["mechanism_dedup_and_parent33_not_prematurely_published"] = current["additional_accepted_count"] == 33 and len(current["accepted_mechanisms"]) == 33 and not any(row.startswith("R162") for row in current["accepted_mechanisms"])
    fees = cost()
    checks["registered_complete_net2percent_and_whole_protection_with_current_fees"] = fees["net_complete_stage_reduction_under_registered_six_side_model"] >= .02 and fees["adjusted_whole_median_ratio"] <= 1.01 and fees["raw_whole_median_ratio"] <= 1.01 and max(fees["each_original_whole_ratios"] + fees["each_adjusted_whole_ratios"]) <= 1.1
    payload = {"status": "separate_root_self_review_passed_pending_actual_review_exit_and_publication0" if all(checks.values()) else "failed_root_self_review0",
               "checks": checks, "check_count": len(checks), "current_identity_rows_checked": len(all_identities),
               "source_snapshot_count": len(source_copies), "process_identity_count": len(process_rows),
               "reviewer": reviewer, "review_is_independent_agent": False, "cost_before_review_actual_exit": fees,
               "generic_original_marker_test_position_weakness_preserved_and_supplemented": True,
               "new_optimization_count": 0, "elapsed_seconds_before_publication": time.perf_counter() - started}
    save(OWN / "SELF_REVIEW_V6_002.json", payload)
    print(json.dumps({"checks": checks, "check_count": len(checks), "net": fees["net_complete_stage_reduction_under_registered_six_side_model"]}))
    if not all(checks.values()):
        raise AssertionError("The separate root review did not pass all original protections")


if __name__ == "__main__":
    main()
