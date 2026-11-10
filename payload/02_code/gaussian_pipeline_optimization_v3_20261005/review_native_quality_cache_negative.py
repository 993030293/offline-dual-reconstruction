"""Independently recheck the complete cache screen and close its negative branch."""

import argparse
from pathlib import Path
import statistics
import time

import psutil

import profile_quality_corner_cost as reference
from native_quality_cache_writer import BASE


def exited(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(BASE)
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    reference.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    (output / "EXECUTED_REVIEW_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    checks, identities = {}, []
    generic = reference.load(BASE / "g1/RESULT.json")
    first = reference.load(BASE / "g0/FAILURE.json")
    pilot = reference.load(BASE / "s0/RESULT.json")
    expected = reference.load(BASE.parent / "vector_gray_stats_001/s0/t0/vector/SCORES.json")["rows"]
    checks["generic15_all_original_interface_and_error_guards_passed"] = len(generic["checks"]) == 15 and all(generic["checks"].values())
    checks["g0_real_original_concurrent_PermissionError_and_audit_errno_failures_preserved"] = first["error_type"] == "PermissionError" and not first["checks"]["original_open_mkstemp_rename_audit_order_and_arguments_exact"] and not first["checks"]["actual_missing_directory_type_errno_winerror_preserved"]
    checks["three_complete_fixed128_AB_BA_AB_pairs_not_full7338"] = len(pilot["pairs"]) == 3 and [pair["order"] for pair in pilot["pairs"]] == [["current", "native"], ["native", "current"], ["current", "native"]]
    for pair in pilot["pairs"]:
        left, right = [BASE / "s0" / ("t" + str(pair["trial"])) / mode for mode in ("current", "native")]
        directories = [left / "quality", right / "quality", BASE.parent / "vector_gray_stats_001/s0/t0/vector/quality"]
        names = sorted(path.name for path in directories[0].iterdir())
        checks["pair" + str(pair["trial"]) + "_all128_scores_metadata_order_and_cache_bytes_exact"] = reference.load(left / "SCORES.json")["rows"] == reference.load(right / "SCORES.json")["rows"] == expected and len(names) == 128 and all(names == sorted(path.name for path in directory.iterdir()) for directory in directories) and all(len({(directory / name).read_bytes() for directory in directories}) == 1 for name in names)
        engine = pair["modes"]["native"]["cache_execution"]["engine"]
        checks["pair" + str(pair["trial"]) + "_actual128_durable_native_writes_and_unload"] = engine["native_calls"] == engine["actual_flushes"] == engine["closed_handles"] == engine["create_attempts"] == 128 and engine["DLL_unloaded"] and not engine["fallback_calls"] and not engine["active_calls"] and not engine["retained_output_bytes"]
    medians = {mode: statistics.median(pair["phase_seconds"][mode] for pair in pilot["pairs"]) for mode in ("current", "native")}
    reduction = 1 - medians["native"] / medians["current"]
    checks["original_stage_medians_and_negative_reduction_recomputed_exact"] = medians == pilot["medians"] and reduction == pilot["reduction"] and reduction < 0 and pilot["status"] == "closed_negative_complete128_below5percent"
    checks["all_generic_and_screen_frozen_sources_and_DLLs_current"] = all(row == reference.data_identity(Path(row["path"])) for result in (generic, pilot) for row in result["sources"])
    snapshots = []
    for folder, prefix in ((BASE / "g0", True), (BASE / "g1", True), (BASE / "s0", True), (BASE / "b0", False), (BASE / "b1", False)):
        result = reference.load(folder / ("FAILURE.json" if folder.name == "g0" else "RESULT.json"))
        for index, row in enumerate(result["sources"]):
            if Path(row["path"]).suffix not in (".py", ".cpp"):
                continue
            snapshot = folder / "source_snapshot" / ((str(index) + "_") if prefix else "")
            snapshot = snapshot.with_name(snapshot.name + Path(row["path"]).name) if prefix else snapshot / Path(row["path"]).name
            if not snapshot.is_file():
                raise FileNotFoundError("A failed or successful executed source snapshot is missing: " + str(snapshot))
            identity = reference.data_identity(snapshot)
            snapshots.append(identity["bytes"] == row["bytes"] and identity["sha256"] == row["sha256"])
    checks["first_build_failed_generic_and_current_executed_source_snapshots_exact"] = all(snapshots)
    fees = []
    for label, expected_code in (("B0", 0), ("B1", 0), ("G0", 1), ("G1", 0), ("S0", 0), ("R0", 1)):
        lease = reference.load(BASE / ("LEASE_" + label + ".json"))
        guard = reference.load(BASE / ("GUARD_" + label + ".json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc") or lease["returncode"] != expected_code or guard["returncode"] != expected_code:
            raise ValueError("An owned first, failed or successful task was not drained and released")
        identities.extend(guard["owned_process_identities"])
        identities.append({"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]})
        fees.append({"label": label, "actual_child_seconds": lease["child_elapsed_seconds"], "returncode": expected_code})
        checks[label + "_actual_time_and_RSS_within_preregistered_budget"] = guard["elapsed_seconds"] <= 120 and guard["peak_tree_RSS_bytes"] <= 4 * 1024**3
    for label in ("b0", "b1"):
        rows = reference.load(BASE / label / "BUILD_TELEMETRY_CLOSE.json")["owned_instances"]
        identities.extend({"pid": row["pid"], "creation_time": row["creation_time"]} for row in rows)
        checks[label + "_only_exact_owned_compiler_telemetry_closed"] = bool(rows) and all(row["exact_instance_exited"] and row["compiler_creation_time"] <= row["creation_time"] and Path(row["exe"]).name == "vctip.exe" for row in rows)
    unique = {(row["pid"], row["creation_time"]): row for row in identities}
    checks["all_saved_owned_instances_actual_PID_ctime_exited"] = all(exited(row) for row in unique.values())
    current = reference.load(reference.ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["parent27_unique_acceptances_unchanged_and_R151_not_counted"] = current["additional_accepted_count"] == len(current["accepted_mechanisms"]) == len(current["acceptance_receipts"]) == 27 and not any(name.startswith("R151") for name in current["accepted_mechanisms"])
    if not all(checks.values()):
        reference.atomic_json(output / "FAILURE.json", {"checks": checks, "accepted_count_contribution": 0})
        raise ValueError("A mandatory negative screen or resource closure check failed")
    reference.atomic_json(output / "RESULT.json", {"status": "closed_negative_R151_complete128_native_cache_slower_root_self_review_passed", "checks": checks,
        "review_kind": "separate_root_self_review_not_independent_agent", "medians": medians, "reduction": reduction,
        "first_failed_successful_measured_fees": fees, "known_candidate_seconds_before_this_review": sum(row["actual_child_seconds"] for row in fees),
        "all_saved_owned_instances_exited": list(unique.values()), "node_seconds": time.perf_counter()-started,
        "full7338_recovery_public_external_GT_training_production_not_run": True, "accepted_count_contribution": 0,
        "parent_goal_active": True, "text_research_and_unmetered_tail_seconds": "unknown_not_assumed_zero"})
    print({"status": "R151_negative_closed", "checks": len(checks), "reduction": reduction, "accepted_count_contribution": 0}, flush=True)


if __name__ == "__main__":
    main()
