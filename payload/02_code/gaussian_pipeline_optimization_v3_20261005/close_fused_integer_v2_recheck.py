"""Review the complete recheck, retain exact-precision failures, and close owned resources."""

import datetime
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from close_evaluation_workspace_profile import digest, gone, read
from screen_fused_integer_v2 import bindings_restored

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fm001"
OUT = BASE / "v2_screen_002"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("The complete scientific fields changed")
        return [row for key in left if key != "image_reference" for row in differences(left[key], right[key], path + "/" + key)]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("The complete scientific shape changed")
        return [row for index, (a, b) in enumerate(zip(left, right)) for row in differences(a, b, path + "/" + str(index))]
    if left == right:
        return []
    if type(left) not in (int, float) or type(right) not in (int, float):
        raise ValueError("A non-numeric scientific field changed: " + path)
    return [{"field": path, "original": left, "current": right, "absolute_delta": abs(left - right)}]


def main():
    started = time.perf_counter()
    destination = BASE / "V2_RECHECK_CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError("A complete recheck closure requires a fresh receipt")
    result = read(OUT / "RESULT.json")
    candidate = read(OUT / "fused/RESULT.json")
    original = read(BASE / "c0/t0/b/RESULT.json")
    node = read(OUT / "n0/NODE.json")
    contract = read(OUT / "CONTRACT.json")
    binding = read(OUT / "fused/BINDING_RESTORATION.json")
    cases = [({"worker_restored": True, "argv_restored": True, "counted": False}, True),
             ({"worker_restored": True, "argv_restored": True, "counted": True}, True),
             ({"worker_restored": False, "argv_restored": True, "counted": False}, False),
             ({"worker_restored": True, "argv_restored": False, "counted": False}, False),
             ({"worker_restored": 1, "argv_restored": True}, False)]
    regression = [bindings_restored(value) is expected for value, expected in cases]
    try:
        bindings_restored({"worker_restored": True})
    except KeyError:
        regression.append(True)
    else:
        regression.append(False)
    current_controller = CODE / "screen_fused_integer_v2.py"
    snapshots = []
    for index, source in enumerate(contract["sources"]):
        copy = OUT / "source_copies" / (f"{index:02d}_" + Path(source["path"]).name)
        snapshots.append({"path": str(copy), "source": source["path"], "equal": digest(copy) == source["sha256"]})
    native_sources = [row for row in contract["sources"] if Path(row["path"]) != current_controller]
    left = read(original["localization"])["rows"]
    right = read(candidate["localization"])["rows"]
    diff = differences(left, right)
    pose_diff = [row for row in diff if any(field in row["field"] for field in ("/R_CW/", "/t_CW/", "/camera_center_m/"))]
    instances, leases, guards = {}, [], []
    for path in sorted(BASE.glob("V2_SCREEN_LEASE_*.json")):
        lease = read(path)
        leases.append({"path": str(path), "released": bool(lease.get("lease_released_utc")),
                       "returncode": lease["returncode"], "child_elapsed_seconds": lease["child_elapsed_seconds"]})
        for role in ("guard", "child"):
            key = (lease[role + "_pid"], lease[role + "_creation_time"])
            instances[key] = {"pid": key[0], "creation_time": key[1], "gone": gone(*key)}
    for path in sorted(BASE.glob("V2_SCREEN_GUARD_*.json")):
        guard = read(path)
        guards.append({"path": str(path), "status": guard["status"], "bounded": guard["elapsed_seconds"] <= guard["maximum_seconds"]
            and guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"], "peak_tree_RSS_bytes": guard["peak_tree_RSS_bytes"]})
        for row in guard["owned_process_identities"]:
            key = (row["pid"], row["creation_time"])
            instances[key] = {**row, "gone": gone(*key)}
    linux_resource = read(candidate["worker"]["resource_receipt"])
    prior_closure = read(BASE / "UNACCEPTED_CLOSURE_001.json")
    linux_rows = [linux_resource] + prior_closure["actual_Linux_now"]
    probe = """import json,psutil,sys
result=[]
for row in json.loads(sys.argv[1]):
    try: absent=psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess: absent=True
    result.append(dict(pid=row['pid'],creation_time=row['creation_time'],gone=absent))
print(json.dumps(result))
"""
    linux = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python",
        "-B", "-c", probe, json.dumps(linux_rows)], check=True, capture_output=True, text=True, timeout=30)
    linux_instances = json.loads(linux.stdout)
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    profile_closer = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/evaluation_workspace_001/DIAGNOSTIC_CLOSURE_001.json")["closer_identity"]
    state = read(DOC / "CURRENT_STATE.json")
    comparison = result["comparison"]
    checks = {
        "original_all24_all48_real_candidate_completed_and_drain": node["status"] == "completed" and node["returncode"] == 0
            and all(candidate["checks"].values()) and candidate["worker"]["returncode"] == 0,
        "original48_matching_feature_pixels_and_nonstrict_guards_passed": comparison["checks"]["all48_actual_match_outputs_exact"]
            and all(value for key, value in comparison["checks"].items() if key != "all24_complete_numeric_pose_centers_stats_exact"),
        "strict_numeric_difference_retained_no_posthoc_tolerance": result["status"] == "failed_scientific_protection"
            and comparison["checks"]["all24_complete_numeric_pose_centers_stats_exact"] is False and len(diff) > 0 and len(pose_diff) > 0,
        "metadata_counted_false_reader_bug_fixed_actual_binding_restore": binding["counted"] is False and bindings_restored(binding)
            and result["checks"]["both_original_private_bindings_restored"] is False and all(regression) and len(regression) == 6,
        "current_original_algorithm_and_full_input_identities": all(digest(row["path"]) == row["sha256"] for row in native_sources + contract["inputs"]),
        "all_executed_current_source_snapshots_preserved": all(row["equal"] for row in snapshots)
            and (BASE / "v2_screen_001/FAILED_CONTROLLER_SOURCE.py").is_file(),
        "all_new_recorded_Windows_and_four_Linux_instances_gone": all(row["gone"] for row in instances.values())
            and len(linux_instances) == 4 and all(row["gone"] for row in linux_instances),
        "both_new_leases_released_os_lock_free": len(leases) == 2 and all(row["released"] for row in leases)
            and read(lock.with_name("LOCK_OWNER.json"))["status"] == "released",
        "actual_windows_linux_cuda_disk_budgets_maintained": all(row["bounded"] for row in guards)
            and linux_resource["closed"] and linux_resource["peak_RSS_bytes"] <= linux_resource["maximum_RSS_bytes"]
            and linux_resource["elapsed_seconds"] <= linux_resource["maximum_seconds"]
            and candidate["worker"]["shutdown"]["peak_CUDA_reserved_bytes"] <= 4 * 1024**3
            and node["resource_peaks"]["GPU_total_used_bytes"] <= 12 * 1024**3
            and sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) <= 30 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        "prior_R139_closer_actual_exit_now_verified": gone(profile_closer["pid"], profile_closer["creation_time"]),
        "parent20_and_five_percent_policy_unchanged": state["additional_accepted_count"] == 20
            and len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 20
            and state["minimum_complete_stage_speed_reduction_current"] == 0.05,
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    elapsed = time.perf_counter() - started
    receipt = {"status": "closed_unaccepted_V2_actual_complete_candidate_strict_numeric_guard_failed_no_speed_acceptance",
        "checks": checks, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "new_optimization_count": 0,
        "strict_numeric_differences": diff, "pose_max_absolute_delta": max(row["absolute_delta"] for row in pose_diff),
        "binding_checker_regression_six": regression, "binding_actual": binding,
        "corrected_controller_source": {"path": str(current_controller), "sha256": digest(current_controller)},
        "source_snapshots": snapshots, "native_source_count": len(native_sources), "input_count": len(contract["inputs"]),
        "windows_instances": list(instances.values()), "linux_instances": linux_instances, "leases": leases, "guards": guards,
        "saved_control_stage_seconds": original["complete_distance_stage_seconds"], "candidate_stage_seconds": candidate["complete_distance_stage_seconds"],
        "saved_control_request_seconds": sum(row["total_prepare_IPC_kernel_seconds"] for row in original["worker"]["calls"]),
        "candidate_request_seconds": sum(row["total_prepare_IPC_kernel_seconds"] for row in candidate["worker"]["calls"]),
        "timing_scope": "Saved control versus current complete candidate is diagnostic, not fresh paired speed evidence",
        "prior_nonoverlapping_fees_seconds": prior_closure["first_saved_child_seconds"],
        "closure_seconds": elapsed,
        "all_known_actual_nonoverlapping_fees_seconds": prior_closure["first_saved_child_seconds"] + sum(row["child_elapsed_seconds"] for row in leases) + elapsed,
        "closer_identity": {"pid": os.getpid(), "creation_time": psutil.Process().create_time(), "requires_outer_exit_verification": True},
        "single_root_self_review_not_independent": True,
        "not_run": ["current_fresh_incumbent", "three_formal_pairs", "external_component", "new_fault_and_recovery",
                    "new_training", "new_object_whole_pipeline", "GT", "preview_or_production_changes"]}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "pose_max_delta": receipt["pose_max_absolute_delta"],
        "numeric_differences": len(diff), "fee_seconds": receipt["all_known_actual_nonoverlapping_fees_seconds"],
        "new_windows_instances": len(instances), "linux_instances": len(linux_instances)}))


if __name__ == "__main__":
    main()
