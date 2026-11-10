"""Audit the registered negative finite-peak component and prior process exits."""

import datetime
import json
import msvcrt
from pathlib import Path
import re
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fcp001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    target = OUT / "NEGATIVE_POST_EXIT_001.json"
    if target.exists():
        raise FileExistsError("A negative closing record must keep its original path")
    result = read(OUT / "g0/RESULT.json")
    lease, guard = read(OUT / "g0_lease.json"), read(OUT / "g0_GUARD.json")
    parent = read(DOC / "CURRENT_STATE.json")
    checks = {
        "all_original_generic_RAW_threshold_and_lifetime_conditions_passed": all(result["checks"].values()) and len(result["checks"]) == 15,
        "measured_complete_component_below_registered5percent": result["component_reduction_not_full_acceptance"] < .05 and result["status"] == "closed_negative_component_screen0",
        "no_pilot_full_or_acceptance_after_negative": result["new_optimization_count"] == 0 and not any((OUT / name).exists() for name in ("p0", "c0")),
        "all_executed_sources_and_both_libraries_current": all(data_identity(Path(row["path"])) == row for row in result["sources"] + [result["library"]]),
        "actual_original_guard_lease_completed0_and_released": guard["returncode"] == lease["returncode"] == 0 and bool(lease["lease_released_utc"]),
        "parent33_unique_and_no_R161_counted": parent["additional_accepted_count"] == len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 33 and not any(name.startswith("R161") for name in parent["accepted_mechanisms"]),
    }
    instances = guard["owned_process_identities"] + [{"pid": lease[key + "pid"], "creation_time": lease[key + "creation_time"]} for key in ("guard_", "child_")]
    alive = []
    for row in instances:
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                alive.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_actual_PID_birth_instances_ended"] = not alive
    shared = OUT.parent / "coordination_001"
    owner = read(shared / "LOCK_OWNER.json")
    checks["latest_resource_owner_released"] = owner["owner"] == "root_R161_fused_peak_generic" and owner["status"] == "released"
    with (shared / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_acquired_and_released"] = True
    report = (DOC / "R161_FUSED_CORNER_PEAK_RESULTS.md").read_text(encoding="utf-8")
    task = (DOC / "R161_FUSED_CORNER_PEAK_TASK.md").read_text(encoding="utf-8")
    checks["report_original_text_counts_scope_and_measured_numbers_verified"] = all(part in report for part in ("33/50", "0.273153%", "15项", "不运行128", "threshold后的response字节", "未独立测定"))
    checks["all_report_links_existing_or_own_final_receipt"] = all(Path(path).is_file() or Path(path) == target for path in re.findall(r"\]\((C:/[^)]+)\)", report))
    checks["registered_component_and_complete_quality_gates_kept"] = all(part in task for part in ("少>=5%", "each<=1.1", "净上界<2%", "不生成完整dilate", "原std::sort"))
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    atomic_json(target, {"status": "closed_negative_complete_corner_component0", "checks": checks,
        "actual_generic_tool_chunk": "e5cc43", "actual_generic_tool_exit_code": 0,
        "actual_generic_lease_seconds": lease["child_elapsed_seconds"], "owned_prior_instances": instances,
        "report_read_as_original_text": True, "independent_agent_review": False,
        "new_optimization_count": 0, "parent_accepted_count": 33, "goal_status": "active",
        "source": data_identity(Path(__file__)), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"checks": checks, "new_count": 0, "parent": 33}), flush=True)


if __name__ == "__main__":
    main()
