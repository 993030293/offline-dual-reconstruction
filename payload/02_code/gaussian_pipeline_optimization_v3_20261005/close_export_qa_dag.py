"""Audit saved revisions, full outputs, measured fees and actual owned process exit."""

import time
STARTED = time.perf_counter()

import datetime
import json
import msvcrt
import os
from pathlib import Path
import shutil
import statistics

import psutil

from profile_export_qa_dag import ROOT, BASE
from runtime import atomic_json, data_identity

CODE = Path(__file__).resolve().parent
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def gone(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    destination = BASE / "COST_REVIEW_PENDING_001.json"
    if destination.exists():
        raise FileExistsError("A closure cannot overwrite its previous audit")
    reviewer_source = data_identity(Path(__file__))
    saved_reviewer = BASE / "REVIEWER_SOURCE_001.py"
    if saved_reviewer.exists():
        raise FileExistsError("The audit requires its own immutable source receipt")
    shutil.copyfile(Path(__file__), saved_reviewer)
    profile = read(BASE / "d0/RESULT.json")
    campaign = read(BASE / "c0/RESULT.json")
    first_failure = read(BASE / "g0/FAILURE.json")
    supplement = read(BASE / "g1/RESULT.json")
    contracts = [(path, read(path)) for path in sorted(BASE.rglob("CONTRACT.json"))]
    old_checker = next(row for row in read(BASE / "g0/CONTRACT.json")["sources"] if Path(row["path"]).name == "check_export_qa_dag.py")
    identities, obsolete, snapshots = {}, [], []
    for path, contract in contracts:
        for row in contract.get("sources", []) + contract.get("inputs", []) + contract.get("map_inputs", []):
            if row == old_checker:
                obsolete.append({"record": str(path), "identity": row})
            else:
                if row["path"] in identities and identities[row["path"]] != row:
                    raise ValueError("A supposedly frozen native source or input has conflicting recorded identities")
                identities[row["path"]] = row
        folder = path.parent / "source_snapshot"
        if folder.exists():
            for index, row in enumerate(contract["sources"]):
                copy = folder / (f"{index:03d}_" + Path(row["path"]).name)
                snapshots.append({"path": str(copy), "match": data_identity(copy)["sha256"] == row["sha256"]})
    identity_checks = [data_identity(row["path"]) == row for row in identities.values()]
    instances, leases, guards = {}, [], []
    for path in sorted(BASE.glob("*LEASE*.json")):
        lease = read(path)
        leases.append({"path": str(path), "released": bool(lease.get("lease_released_utc")),
                       "returncode": lease["returncode"], "child_elapsed_seconds": lease["child_elapsed_seconds"]})
        for role in ("guard", "child"):
            row = {"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]}
            instances[(row["pid"], row["creation_time"])] = {**row, "gone": gone(row)}
    for path in sorted(BASE.rglob("*GUARD*.json")):
        guard = read(path)
        guards.append({"path": str(path), "status": guard["status"], "returncode": guard["returncode"],
            "peak_tree_RSS_bytes": guard["peak_tree_RSS_bytes"], "bounded": guard["elapsed_seconds"] <= guard["maximum_seconds"]
                and guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"]})
        for row in guard["owned_process_identities"]:
            instances[(row["pid"], row["creation_time"])] = {**row, "gone": gone(row)}
    worker_closures = []
    for path in sorted(BASE.rglob("CLOSURE.json")):
        row = read(path)
        worker_closures.append({"path": str(path), "closed": row["closed"], "status": row["status"]})
        if "identity" in row:
            identity = row["identity"]
            instances[(identity["pid"], identity["creation_time"])] = {**identity, "gone": gone(identity)}
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    state = read(DOC / "CURRENT_STATE.json")
    old_receipts = [read(ROOT / path) for path in state["acceptance_receipts"]]
    pairs = campaign["pairs"]
    serial_wholes = [row["nodes"]["serial"]["whole_seconds"] for row in pairs]
    dag_wholes = [row["nodes"]["dag"]["whole_seconds"] for row in pairs]
    checks = {
        "native_complete_profile_and_all_original_protections": profile["status"] == "complete_unchanged_export_profile_protected" and all(profile["checks"].values()),
        "three_preregistered_fresh_interleaved_complete_pairs": len(pairs) == 3 and [row["order"] for row in pairs] == [["serial", "dag"], ["dag", "serial"], ["serial", "dag"]],
        "three_full1046_bytes_four_scientific_JSON_comparisons": all(row["comparison"]["complete_byte_pairs"] == 1046 and row["comparison"]["complete_JSON_pairs"] == 4 for row in pairs),
        "six_original_entry_denominators_quality_bindings_and_workers": all(all(read(BASE / f"c0/t{index}/{mode}/RESULT.json")["checks"].values()) for index in range(3) for mode in ("serial", "dag")),
        "original_whole_each_pair_and_median_guards": all(row["whole_ratio"] <= 1.1 for row in pairs) and statistics.median(dag_wholes) <= statistics.median(serial_wholes),
        "g0_original_public_type_failure_and_nine_passed_fixtures_preserved": first_failure["status"] == "failed_DAG_supplements_no_acceptance"
            and "camera format or full point membership" in first_failure["error"] and all(first_failure["checks_before_failure"].values()) and len(obsolete) == 1,
        "g1_all_generic_public_native_compatibility_errors_cancel_recovery": supplement["status"] == "complete_DAG_generic_external_errors_cancellation_recovery_passed" and all(supplement["checks"].values()),
        "public360_native_OPENCV_not_misrepresented_training_admission": next(row for row in supplement["records"] if row.get("fixture", "").startswith("existing_Oxford"))["strict_current_PINHOLE_training_admission"] == "not_passed_original_native_OPENCV_remains",
        "all_current_native_sources_inputs_private_maps_exact": all(identity_checks),
        "all_executed_source_revisions_snapshots_preserved": len(snapshots) == 162 and all(row["match"] for row in snapshots),
        "all_recorded_actual_owned_Windows_processes_exited": all(row["gone"] for row in instances.values()),
        "all_five_leases_released_actual_OS_lock_free": len(leases) == 5 and all(row["released"] for row in leases) and read(lock.with_name("LOCK_OWNER.json"))["status"] == "released",
        "all_eleven_real_guards_time_RSS_bounded_expected_one_failure": len(guards) == 11 and all(row["bounded"] for row in guards)
            and sum(row["status"] == "failed_child" and row["returncode"] == 1 for row in guards) == 1
            and sum(row["status"] == "completed" and row["returncode"] == 0 for row in guards) == 10,
        "all_worker_closures_preserved_and_closed": all(row["closed"] for row in worker_closures),
        "old20_unique_accepted_receipts_current5percent_unchanged": state["additional_accepted_count"] == 20
            and len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 20
            and all(row["status"].startswith("accepted") for row in old_receipts) and state["minimum_complete_stage_speed_reduction_current"] == .05,
        "all_actual_owned_output_and_disk_bounds": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) < 30 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    node_seconds = sum(serial_wholes) + sum(dag_wholes)
    campaign_elapsed = read(BASE / "PAIR_LEASE_001.json")["child_elapsed_seconds"] + read(BASE / "PAIRS_REMAINING_LEASE_001.json")["child_elapsed_seconds"]
    campaign_overhead = campaign_elapsed - node_seconds
    if campaign_overhead < 0:
        raise ValueError("Measured campaign overhead cannot be negative")
    other = [row for row in leases if "PAIR_LEASE" not in Path(row["path"]).name and "PAIRS_REMAINING_LEASE" not in Path(row["path"]).name]
    probe_seconds = .4489001 + .4972787 + .2729114 + 1.4585149 + .8752773 + .2128729
    fees_before_closing_envelope = sum(row["child_elapsed_seconds"] for row in other) + campaign_overhead + probe_seconds
    medians = {kind: {mode: statistics.median(row["nodes"][mode][kind + "_seconds"] for row in pairs) for mode in ("serial", "dag")}
               for kind in ("phase", "whole")}
    tree_RSS = {mode: statistics.median(read(BASE / f"c0/t{index}/{mode}_GUARD.json")["peak_tree_RSS_bytes"] for index in range(3)) for mode in ("serial", "dag")}
    atomic_json(destination, {"status": "root_self_review_passed_complete_cost_publication_pending", "checks": checks,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "current_identity_count": len(identities), "source_snapshots": snapshots,
        "obsolete_checker_revision": obsolete, "windows_instances": list(instances.values()), "leases": leases, "guards": guards,
        "worker_closures": worker_closures, "raw_medians": medians, "median_sampled_process_tree_RSS_bytes": tree_RSS,
        "complete_campaign_overhead_seconds": campaign_overhead, "recorded_nonoverlapping_probe_tool_seconds": probe_seconds,
        "all_known_measured_fees_before_this_closure_and_publication_seconds": fees_before_closing_envelope,
        "closure_before_receipt_seconds": time.perf_counter() - STARTED,
        "reviewer_source": reviewer_source, "saved_reviewer_source": str(saved_reviewer),
        "closer_identity": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "new_optimization_count": 0, "self_review_not_independent_agent_review": True,
        "unmeasured_research_text_reading_and_agent_planning_not_formal_algorithm_cost": True})
    print(json.dumps({"status": "root_self_review_passed_complete_cost_publication_pending", "checks": checks,
        "fee_seconds_before_closure": fees_before_closing_envelope, "median_seconds": medians, "Windows_instances": len(instances)}), flush=True)


if __name__ == "__main__":
    main()
