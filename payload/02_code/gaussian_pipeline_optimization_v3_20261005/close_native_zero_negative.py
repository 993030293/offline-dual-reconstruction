"""Audit the complete negative guard and close only the private experiment."""

import json
import msvcrt
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

import run_native_zero_feature_campaign as campaign
from runtime import atomic_json, data_identity

ROOT, CODE, OUT = campaign.ROOT, campaign.CODE, campaign.OUT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows_closed(row):
    try:
        process = psutil.Process(row["pid"])
        return process.create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    result = read(OUT / "c2/RESULT.json")
    pairs = result["pairs"]
    medians = {mode: statistics.median(pair["rows"][mode]["whole_seconds"] for pair in pairs) for mode in ("native", "direct")}
    comparisons = [campaign.compare(OUT / "c2" / ("t" + str(index)) / "native",
                                    OUT / "c2" / ("t" + str(index)) / "direct") for index in range(3)]
    original = read(OUT / "c2/t0/native/RESULT.json")
    source_copies = OUT / "c2/source_snapshot"
    current_source_rows = result["sources"]
    copies_exact = all(data_identity(source_copies / (f"{index:03d}_" + Path(row["path"]).name))["sha256"] == row["sha256"]
                       for index, row in enumerate(current_source_rows))
    inputs_current = all(data_identity(campaign.local(row["path"]))["sha256"] == row["sha256"]
                         and campaign.local(row["path"]).stat().st_size == row["bytes"] for row in original["inputs"])
    sources_current = all(data_identity(campaign.local(row["path"]))["sha256"] == row["sha256"]
                          for row in original["sources"])
    current_own = all(data_identity(row["path"]) == row for row in current_source_rows)
    windows = {}
    linux = {}
    for path in OUT.rglob("*.json"):
        if path.parent.name.endswith("_history") or path.name.startswith("CUDA_TRANSFER_"):
            continue
        try:
            row = read(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        for key in ("owner", "failed_owner", "failed_actual_owner", "failed_actual_node_owner"):
            if isinstance(row.get(key), dict) and set(row[key]) >= {"pid", "creation_time"}:
                item = row[key]
                windows[item["pid"], item["creation_time"]] = item
        for key in ("guard",):
            if isinstance(row.get(key), dict):
                for item in row[key].get("owned_process_identities", []):
                    windows[item["pid"], item["creation_time"]] = item
        if set(row) >= {"guard_pid", "guard_creation_time"}:
            item = {"pid": row["guard_pid"], "creation_time": row["guard_creation_time"]}
            windows[item["pid"], item["creation_time"]] = item
        if set(row) >= {"child_pid", "child_creation_time"}:
            item = {"pid": row["child_pid"], "creation_time": row["child_creation_time"]}
            windows[item["pid"], item["creation_time"]] = item
        if row.get("scope") == "WSL" and set(row) >= {"pid", "creation_time"}:
            item = {"pid": row["pid"], "creation_time": row["creation_time"]}
            linux[item["pid"], item["creation_time"]] = item
    for history in OUT.glob("c0/t1/direct_node/guard_history"):
        paths = sorted(history.glob("sample_*.json"))
        if paths:
            for item in read(paths[-1]).get("owned_process_identities", []):
                windows[item["pid"], item["creation_time"]] = item
    win_checks = [{**row, "closed": windows_closed(row)} for row in windows.values()]
    payload = json.dumps(list(linux.values()))
    code = "import json,psutil; rows=json.loads(" + repr(payload) + "); answer=[]\n"
    code += "for r in rows:\n try:\n  p=psutil.Process(r['pid']); alive=p.create_time()==r['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess:\n  alive=False\n answer.append(dict(r,closed=not alive))\nprint(json.dumps(answer))"
    linux_checks = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    leases = [read(path) for path in sorted(OUT.glob("*_lease.json"))]
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    ratio = medians["direct"] / medians["native"]
    checks = {"three_actual_fresh_alternating_complete_pairs": len(pairs) == 3 and result["orders"] == [["native", "direct"], ["direct", "native"], ["native", "direct"]],
              "all_original15_fields17_sources1036_inputs_and_two_files_recompared": all(all(row["checks"].values()) for row in comparisons),
              "current_all_original_sources_and_inputs_unchanged": inputs_current and sources_current,
              "current_all_owned_formal_sources_and_execution_copies_unchanged": current_own and copies_exact,
              "actual_normal_native_CUDA_payloads_each17274876_2727612": all(pair["rows"]["native"]["actual_native_HtoD_bytes"] == 17274876 and pair["rows"]["direct"]["actual_native_HtoD_bytes"] == 2727612 for pair in pairs),
              "original_whole_median_gate_failed_without_reinterpretation": result["status"] == "closed_negative_whole_median_guard0" and ratio > 1.01,
              "saved_complete_medians_exact": medians == result["whole_medians"],
              "all_formal_Linux16GiB300second_bounds_actual": all(read(OUT / "c2" / ("t" + str(index)) / mode / "WSL_RESOURCE.json")["maximum_RSS_bytes"] == 16 * 1024**3
                  and read(OUT / "c2" / ("t" + str(index)) / mode / "WSL_RESOURCE.json")["maximum_seconds"] == 300 for index in range(3) for mode in ("native", "direct")),
              "all_formal_real_guard_exit0_saved_before_final_journal": all(read(OUT / "c2" / ("t" + str(index)) / (mode + "_node") / "ACTUAL_GUARD_EXIT.json")["returncode"] == 0 for index in range(3) for mode in ("native", "direct")),
              "all_saved_actual_Windows_instances_closed": all(row["closed"] for row in win_checks),
              "all_saved_actual_Linux_instances_closed": all(row["closed"] for row in linux_checks),
              "all_private_leases_released": all(row.get("lease_released_utc") for row in leases),
              "OS_exclusive_lock_reacquired_and_released": True,
              "V1_CUPTI_failure_and_old_C0_C1_failures_retained": (OUT / "g0/PARTIAL.json").is_file() and (OUT / "c0/t1/direct.log").is_file() and (OUT / "c1/t0/native.log").is_file(),
              "current_V2_g2_four_native_oracles_passed": all(read(OUT / "g2/RESULT.json")["checks"].values()),
              "root_parent_count_still30_and_goal_not_marked_complete": read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")["additional_accepted_count"] == 30,
              "unexecuted_contract_public_fault_recovery_not_claimed": not (OUT / "g3/RESULT.json").exists() and not (OUT / "v3/RESULT.json").exists()}
    actual_bytes = sum(read(OUT / name)["extra_actual_HtoD_bytes"] for name in ("g1/RESULT.json", "g2/RESULT.json"))
    actual_bytes += sum(read(OUT / "c0/t0" / mode / "ZERO_TRANSFER_TRACE.json")["actual_native_HtoD_bytes"] for mode in ("native", "direct"))
    fees = {Path(path).name: read(path)["child_elapsed_seconds"] for path in sorted(OUT.glob("*_lease.json"))}
    atomic_json(OUT / "NEGATIVE_REVIEW_001.json", {"status": "passed_negative_closure" if all(checks.values()) else "failed_review",
        "checks": checks, "review_kind": "separate_root_self_review_not_independent_agent",
        "whole_medians": medians, "whole_ratio": ratio, "whole_slowdown": ratio - 1,
        "raw_initializer_successful_native_HtoD_reduction": 1 - 2727612 / 17274876,
        "known_private_lease_fees_seconds": fees, "formal_two_lease_windows_are_not_recharged_as_extra_cost": ["c2_first_lease.json", "c2_rest_lease.json"],
        "extra_actual_native_HtoD_bytes": actual_bytes, "old_failed_t1_conditional_work_budget_not_native_census": 17274876,
        "old_V1_source_request_budget_not_native_census": 2604,
        "original_or_new_formal_quality_speed_training_GT_production_not_completed": True,
        "not_run": ["g3_native_direction_capacity_error_extended_contracts", "public133020_360_component", "actual_full_consumer_fault_recovery", "final_net_work_acceptance", "new_training", "GT", "production_integration"],
        "Windows_instances": win_checks, "Linux_instances": linux_checks,
        "actual_review_seconds_before_publication": time.perf_counter() - started,
        "source": data_identity(Path(__file__)), "pid": os.getpid(), "creation_time": psutil.Process().create_time(),
        "new_optimization_count": 0, "parent_accepted_count": 30})
    if not all(checks.values()):
        raise AssertionError("A separate native zero-feature negative review failed")
    print({"status": "negative_closed0", "checks": len(checks), "whole_slowdown": ratio - 1,
           "native_work_reduction": 1 - 2727612 / 17274876, "new": 0})


if __name__ == "__main__":
    main()
