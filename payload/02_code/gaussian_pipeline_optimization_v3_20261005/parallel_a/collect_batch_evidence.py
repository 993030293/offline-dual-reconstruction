"""Consolidate original terminal receipts without relabeling pending acceptance."""

import argparse
import datetime
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/parallel_a_001"
DOC = ROOT/"research/2026-10-05-gaussian-optimization-round3-50/parallel_001/a"
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def location(path):
    return str(Path(path).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--own-live-lease", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    p2, p4, x2 = (read(OUT/name/"RESULT.json") for name in ("p2s", "p4", "x2"))
    audits = read(OUT/"source_audit_001/RESULT.json")
    recovery = read(OUT/"recovery_002/RESULT.json")
    if p2["status"] != "accepted_pending_monitor" or p4["status"] != "rejected_speed" or x2["status"] != "external_component_passed":
        raise ValueError("A complete measured disposition changed or remains unfinished")
    if recovery["status"] != "passed" or not all(recovery["checks"].values()):
        raise ValueError("The final current-entry recovery guard did not pass")
    source_checks = []
    for label, records in (("PA02_current", p2["sources"]), ("PA04_current", p4["sources"]),
            ("PA02_Oxford", x2["sources"]), ("current_recovery", recovery["sources"])):
        for identity in records:
            source_checks.append({"scope": label, "identity": identity, "current_exact": data_identity(Path(identity["path"])) == identity})
    for key in ("PA01", "PA03"):
        for row in audits[key]["sources"]:
            source_checks.append({"scope": key, "identity": row["identity"], "current_exact": data_identity(Path(row["identity"]["path"])) == row["identity"]})
    if not all(row["current_exact"] for row in source_checks):
        raise ValueError("A current source differs from a required terminal receipt")
    leases, processes, alive = [], [], []
    for path in sorted(OUT.rglob("*_lease.json")):
        if path.resolve() == args.own_live_lease.resolve():
            continue
        row = read(path)
        leases.append({"path": location(path), "owner": row.get("owner"), "returncode": row.get("returncode"),
            "wait_seconds": row.get("wait_seconds"), "released": bool(row.get("lease_released_utc")), "release_utc": row.get("lease_released_utc")})
    for path in sorted(OUT.rglob("*_process.json")):
        row = read(path)
        if not row.get("status") in ("completed", "failed_child"):
            continue
        identities = row.get("owned_process_identities", [])
        for identity in identities:
            try:
                process = psutil.Process(identity["pid"])
                if abs(process.create_time()-identity["creation_time"]) < .001 and process.is_running():
                    alive.append({"receipt": location(path), "identity": identity})
            except psutil.NoSuchProcess:
                pass
        processes.append({"path": location(path), "status": row["status"], "returncode": row.get("returncode"),
            "elapsed_seconds": row["elapsed_seconds"], "peak_windows_tree_RSS_bytes": row["peak_tree_RSS_bytes"]})
    if not all(row["released"] for row in leases) or alive:
        raise ValueError("A prior exact A lease or Windows process still owns work")
    gpu = []
    for path in sorted((OUT/"p4").rglob("INTEGER_SIFT_TRACE.json")):
        row = read(path)["trace"]
        gpu.append({"path": location(path), "shutdown": row.get("shutdown"), "returncode": row.get("returncode")})
    if len(gpu) != 6 or not all(row["returncode"] == 0 and row["shutdown"]["status"] == "stopped" for row in gpu):
        raise ValueError("An original full matching worker did not report its own stopped shutdown")
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    disk = psutil.disk_usage(str(ROOT))
    if size > 30*1024**3 or disk.free < 20*1024**3:
        raise ValueError("The batch crossed its frozen disk or headroom guard")
    rows = []
    for key in ("PA01", "PA03"):
        rows.append({"id": key, "decision": "rejected", "basis": audits[key]["reason"],
            "implemented": False, "three_current_pairs": "not_applicable_pre_execution_duplicate",
            "receipt": location(OUT/("source_audit_001/"+key+"_REJECTION.json")), "counted": False})
    rows.insert(1, {"id": "PA02", "decision": "accepted", "acceptance_scope": "local_effect_confirmed_external_body_pose_component_passed",
        "independent_review": "pending", "central_promoted": False, "implemented": True, "three_current_pairs": 3,
        "current_result": location(OUT/"p2s/RESULT.json"), "current_median": p2["median"], "current_reductions": p2["reductions"],
        "external_result": location(OUT/"x2/RESULT.json"), "external_median": x2["median"], "external_reductions": x2["reductions"],
        "whole_entry_bounded_poll_resolution_seconds": .5, "whole_net_speed_not_established_at_this_resolution": True,
        "superseded_incomplete_output_cost": location(OUT/"p2/RESULT.json"), "counted": False})
    rows.append({"id": "PA04", "decision": "rejected", "basis": "Both preregistered complete read phase reduction gates failed",
        "implemented": True, "three_current_pairs": 3, "current_result": location(OUT/"p4/RESULT.json"),
        "current_median": p4["median"], "current_reductions": p4["reductions"],
        "external": "not_run_local_speed_failed_stop_rule", "counted": False})
    evidence = {"schema_version": 1, "task_id": "batch_a_001", "thread_id": "01a10c61-9180-7b33-b15d-d414c526b640",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "evidence_complete_independent_review_pending",
        "candidate_count": 4, "local_accepted_candidates": 1, "rejected_candidates": 3, "main50_counter_modified": False,
        "main_goal_not_completed_by_this_batch": True, "native_goal_status": "active", "candidates": rows,
        "input_denominators": {"RAW": 7338, "eligible": 6797, "initial": 1002, "pairs": 11583,
            "admitted_cameras": 1016, "admitted_points": 75767, "admitted_observations": 959120, "development": 24, "excluded": 48},
        "source_identity_checks": source_checks, "prior_actual_leases": leases, "prior_terminal_processes": processes,
        "prior_exact_owned_Windows_processes_still_live": alive, "original_GPU_worker_shutdown_receipts": gpu,
        "Windows_RSS_scope": "Owned Windows process tree only; not full WSL RSS", "disk_artifact_bytes": size,
        "disk_free_bytes": disk.free, "maximum_prior_windows_tree_RSS_bytes": max(row["peak_windows_tree_RSS_bytes"] for row in processes),
        "own_final_audit_lease": location(args.own_live_lease), "own_final_audit_must_release_before_completion": True,
        "recovery_result": location(OUT/"recovery_002/RESULT.json"),
        "unverified": ["New object full pipeline", "Oxford full camera image selection and SfM/GS",
            "ATE/RPE/TLS without independent reference", "Full new training", "Production code integration",
            "Forced guard kill and WSL descendant recovery", "Formal central promotion before parent audit"],
        "failed_or_superseded": ["pa02_oracle_001 failed SciPy fixture input shape; corrected fixture only in 002",
            "p2 initial complete pairs omitted original selector output tail; superseded by p2s",
            "PA04 full speed gates failed; no further parameter trials",
            "PA01 and PA03 rejected duplicate mechanisms before new execution"],
        "earlier_R105": {"belongs_to_this_batch": False, "restarted": False,
            "handoff_receipt": location(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/wp001/RESULT.json"),
            "speed_failed_count_zero_and_no_A_batch_lease_verified": True}}
    atomic_json(args.output/"RESULT.json", evidence)
    atomic_json(DOC/"RESULTS.json", evidence)
    print({"status": evidence["status"], "candidates": [(row["id"], row["decision"]) for row in rows],
        "leases": len(leases), "exact_owned_live": len(alive), "artifact_bytes": size}, flush=True)


if __name__ == "__main__":
    main()
