"""Close the bounded complete evaluator diagnosis without accepting an algorithm."""

import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/evaluation_workspace_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows_path(name):
    return Path("C:/" + name[7:]) if name.startswith("/mnt/c/") else Path(name)


def gone(pid, creation_time):
    try:
        return psutil.Process(pid).create_time() != creation_time
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    destination = BASE / "DIAGNOSTIC_CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError("The profile closure requires a fresh receipt")
    profiles = {name: read(BASE / name / "PROFILE.json") for name in ("d0", "d1")}
    contracts = {name: read(BASE / name / "CONTRACT.json") for name in profiles}
    snapshots = []
    for name, contract in contracts.items():
        for index, source in enumerate(contract["sources"]):
            copy = BASE / name / "source_copies" / (f"{index:02d}_" + Path(source["path"]).name)
            snapshots.append({"path": str(copy), "equal": digest(copy) == source["sha256"]})
    current = {row["path"]: row for row in contracts["d1"]["sources"] + contracts["d1"]["inputs"]}
    for row in profiles["d1"]["native_sources"]:
        path = windows_path(row["path"])
        current[str(path)] = {**row, "path": str(path)}
    instances, leases, guards, linux_resources = {}, [], [], []
    for path in sorted(BASE.glob("PROFILE_LEASE_*.json")):
        lease = read(path)
        leases.append({"path": str(path), "released": bool(lease.get("lease_released_utc")),
                       "returncode": lease["returncode"], "child_elapsed_seconds": lease["child_elapsed_seconds"]})
        for role in ("guard", "child"):
            key = (lease[role + "_pid"], lease[role + "_creation_time"])
            instances[key] = {"pid": key[0], "creation_time": key[1], "gone": gone(*key)}
    guard_paths = list(BASE.glob("PROFILE_GUARD_*.json")) + list(BASE.glob("d*/WINDOWS_GUARD.json"))
    for path in sorted(guard_paths):
        guard = read(path)
        guards.append({"path": str(path), "status": guard["status"], "peak_tree_RSS_bytes": guard["peak_tree_RSS_bytes"],
                       "bounded": guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"]
                       and guard["elapsed_seconds"] <= guard["maximum_seconds"]})
        for row in guard["owned_process_identities"]:
            key = (row["pid"], row["creation_time"])
            instances[key] = {**row, "gone": gone(*key)}
    for name in profiles:
        linux_resources.append(read(BASE / name / "measurement/WSL_RESOURCE.json"))
    probe = """import json,psutil,sys
rows=json.loads(sys.argv[1]); result=[]
for row in rows:
    try: absent=psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess: absent=True
    result.append(dict(pid=row['pid'],creation_time=row['creation_time'],gone=absent))
print(json.dumps(result))
"""
    linux = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", probe, json.dumps(linux_resources)],
        check=True, capture_output=True, text=True, timeout=30)
    linux_instances = json.loads(linux.stdout)
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    state = read(DOC / "CURRENT_STATE.json")
    events = profiles["d1"]["cuda_stream_interval_events"]
    timings = {}
    for row in events:
        timings[row["name"]] = timings.get(row["name"], 0.0) + row["elapsed_ms"]
    measurement = read(BASE / "d1/measurement/RESULT.json")
    output_bytes = sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file())
    checks = {
        "first_CUDA_collection_failure_preserved_not_zero_gpu_cost": profiles["d0"]["status"] == "failed_verification"
            and profiles["d0"]["checks"]["actual_CUDA_kernel_events_available"] is False
            and all(value for key, value in profiles["d0"]["checks"].items() if key != "actual_CUDA_kernel_events_available"),
        "corrected_complete_native_entry_all_seven_guards": profiles["d1"]["status"] == "complete_native_profile_passed"
            and all(profiles["d1"]["checks"].values()) and read(BASE / "d1/RESULT.json")["new_optimization_count"] == 0,
        "actual48_renders_four48_property_calls_240_stream_intervals": all(value == 48 for value in profiles["d1"]["calls"].values())
            and len(events) == 240,
        "all_current_complete_inputs_and_sources_intact": all(digest(row["path"]) == row["sha256"] for row in current.values()),
        "all_ten_executed_source_snapshots_preserved": len(snapshots) == 10 and all(row["equal"] for row in snapshots),
        "all_recorded_Windows_and_Linux_instances_gone": all(row["gone"] for row in instances.values())
            and len(linux_instances) == 2 and all(row["gone"] for row in linux_instances),
        "both_leases_released_original_lock_free": len(leases) == 2 and all(row["released"] for row in leases)
            and read(lock.with_name("LOCK_OWNER.json"))["status"] == "released",
        "all_Windows_Linux_time_RSS_disk_budgets": len(guards) == 4 and all(row["bounded"] for row in guards)
            and all(row["status"] == "completed" and row["peak_tree_RSS_bytes"] <= row["maximum_RSS_bytes"]
                    and row["elapsed_seconds"] <= row["maximum_seconds"] for row in linux_resources)
            and output_bytes < 30 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        "original_actual_CUDA_reserved_peak_bounded": measurement["peak_CUDA_reserved_bytes"] <= 12 * 1024**3,
        "parent20_unique_mechanisms_receipts_and_current5percent_guard": state["additional_accepted_count"] == 20
            and len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 20
            and state["minimum_complete_stage_speed_reduction_current"] == 0.05,
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    elapsed = time.perf_counter() - started
    receipt = {"status": "closed_protected_profile_candidate_implementation_deferred_no_performance_acceptance", "checks": checks,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cuda_stream_interval_totals_ms": timings,
        "nested_intervals_must_not_be_added_to_render": True, "profile_not_formal_performance_or_upper_bound": True,
        "corrected_complete_child_seconds": measurement["complete_child_seconds"], "current_identity_count": len(current),
        "snapshots": snapshots, "windows_instances": list(instances.values()), "linux_instances": linux_instances,
        "leases": leases, "guards": guards, "Linux_resources": linux_resources, "output_bytes": output_bytes,
        "recorded_hardware_probe_tool_envelope_seconds": 3.3960952,
        "closure_seconds": elapsed,
        "all_known_nonoverlapping_diagnostic_fees_seconds": sum(row["child_elapsed_seconds"] for row in leases) + elapsed + 3.3960952,
        "closer_identity": {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "outer_exit_check_required": True},
        "not_run": ["new_render_or_model_cache_implementation", "three_formal_pairs", "algorithm_fault_and_new_recovery",
                    "new_training", "GT", "new_object_whole_pipeline", "production_or_preview_changes"],
        "new_optimization_count": 0, "parent_additional_accepted_count": 20, "single_root_self_review_not_independent": True}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "fees_seconds": receipt["all_known_nonoverlapping_diagnostic_fees_seconds"],
                      "windows_instances": len(instances), "linux_instances": len(linux_instances)}))


if __name__ == "__main__":
    main()
