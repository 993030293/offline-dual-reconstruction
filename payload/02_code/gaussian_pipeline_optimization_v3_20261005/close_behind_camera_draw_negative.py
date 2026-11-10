"""Close the actual browser screen while preserving all prior instrumentation failures."""

import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/behind_camera_draw_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gone(pid, creation_time):
    try:
        return psutil.Process(pid).create_time() != creation_time
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    destination = BASE / "NEGATIVE_CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError("A negative closure must retain its unique version")
    names = ["g0", "g1", "g2", "p0", "g3", "p1"]
    results = {name: read(BASE / name / "RESULT.json") for name in names}
    latest = results["p1"]
    current = {row["path"]: row for row in latest["sources"] + latest["inputIdentities"]}
    snapshots = []
    for name, result in results.items():
        for row in result["sources"]:
            copy = BASE / name / "sources" / Path(row["path"]).name
            snapshots.append({"run": name, "path": str(copy), "equal": digest(copy) == row["sha256"]})
    leases = []
    instances = {}
    fees = []
    for path in sorted(BASE.glob("*_LEASE_*.json")):
        lease = read(path)
        leases.append({"path": str(path), "released": bool(lease.get("lease_released_utc")), "returncode": lease["returncode"]})
        fees.append({"path": str(path), "seconds": lease["child_elapsed_seconds"], "includes_nested_guard_and_browser_once": True})
        for role in ("guard", "child"):
            key = (lease[role + "_pid"], lease[role + "_creation_time"])
            instances[key] = {"pid": key[0], "creation_time": key[1], "gone": gone(*key)}
    guards = []
    for path in sorted(BASE.glob("*_GUARD_*.json")):
        guard = read(path)
        guards.append({"path": str(path), "status": guard["status"], "peak_tree_RSS_bytes": guard["peak_tree_RSS_bytes"],
                       "bounded": guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"]
                       and guard["elapsed_seconds"] <= guard["maximum_seconds"]})
        for row in guard["owned_process_identities"]:
            key = (row["pid"], row["creation_time"])
            instances[key] = {**row, "gone": gone(*key)}
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    state = read(DOC / "CURRENT_STATE.json")
    pair = latest["pairs"][0]
    checks = {
        "latest_complete124_each_and_all1122_native_guards": latest["status"] == "screen_negative_no_promotion"
        and len(latest["episodes"]) == 2 and all(len(row["captures"]) == 124 for row in latest["episodes"])
        and len(latest["checks"]) == 1122 and all(row["passed"] for row in latest["checks"]),
        "original_work_gate_failed_without_alternate_acceptance": pair["reduction"] < 0.2,
        "generic51_both_complete_models_full_half_and_native_button_regression": results["g3"]["status"] == "generic_passed"
        and len(results["g3"]["generic"]["checks"]) == 51
        and {row["count"] for row in results["g3"]["generic"]["actual"]} == {526886, 60451}
        and all(row["passed"] for row in results["g3"]["generic"]["checks"])
        and all(row["passed"] for row in results["g3"]["checks"]),
        "initial_small_fixture_and_private_button_failures_preserved": results["g0"]["status"] == "failed_verification"
        and "heterogeneous_native_bounds" in results["g0"]["error"]
        and results["p0"]["status"] == "failed_verification" and "Timeout" in results["p0"]["error"],
        "all_current_source_and_complete_native_input_identities": all(digest(row["path"]) == row["sha256"] for row in current.values()),
        "all_historical_executed_source_versions_preserved": all(row["equal"] for row in snapshots),
        "all_recorded_windows_instances_gone": all(row["gone"] for row in instances.values()),
        "all_six_leases_released_and_os_lock_free": len(leases) == 6 and all(row["released"] for row in leases)
        and read(lock.with_name("LOCK_OWNER.json"))["status"] == "released",
        "all_actual_time_rss_and_postrun_disk_budgets": len(guards) == 6 and all(row["bounded"] for row in guards)
        and sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) < 30 * 1024**3
        and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        "parent20_count_and_prior20_receipts_unchanged": state["additional_accepted_count"] == 20
        and len(set(state["accepted_mechanisms"])) == 20 and len(set(state["acceptance_receipts"])) == 20
        and all((ROOT / path).is_file() for path in state["acceptance_receipts"]),
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    closure_seconds = time.perf_counter() - started
    result = {"status": "closed_negative_original20percent_draw_gate_no_promotion", "checks": checks,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "new_optimization_count": 0, "parent_additional_accepted_count": 20,
              "single_chat_self_review_not_independent_agent": True, "measured_pair": pair,
              "raw_whole_screen_reduction": 1 - pair["candidateSeconds"] / pair["nativeSeconds"],
              "current_identity_count": len(current), "source_snapshot_count": len(snapshots),
              "snapshots": snapshots, "recorded_windows_instances": list(instances.values()),
              "leases": leases, "guards": guards, "measured_fees_including_failures": fees,
              "closure_seconds": closure_seconds,
              "failed_root_closure_envelope_seconds": read(BASE / "FAILED_CLOSURE_TOOL_001.json")["measured_tool_envelope_seconds"],
              "all_known_actual_nonoverlapping_fees_seconds": sum(row["seconds"] for row in fees) + closure_seconds
              + read(BASE / "FAILED_CLOSURE_TOOL_001.json")["measured_tool_envelope_seconds"],
              "closer_identity": {"pid": os.getpid(), "creation_time": psutil.Process().create_time(),
                                  "own_exit_requires_separate_outer_verification": True},
              "gpu_evidence": "sampled_whole_device_only_not_owned_peak_or_gpu_time",
              "not_run": ["three_fresh_formal_pairs", "extra_keyboard_pointer_wheel_path_playback_controls",
                          "additional_permissions_cancellation_injection_and_new_context_recovery",
                          "new_training", "independent_GT", "unseen_object_complete_flow", "production_deployment"],
              "text_publication_after_this_fee_boundary_not_claimed_as_included": True}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "checks": len(checks), "recorded_windows_instances": len(instances),
                      "fees_seconds": result["all_known_actual_nonoverlapping_fees_seconds"], "count": 0}))


if __name__ == "__main__":
    main()
