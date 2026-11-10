"""Audit the original text and exact negative component before closing its version."""

import datetime
import json
import msvcrt
from pathlib import Path
import re
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco002"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    target = OUT / "NEGATIVE_V2_POST_EXIT_001.json"
    if target.exists():
        raise FileExistsError("The negative closure needs a new path")
    result, lease, guard = read(OUT / "g0/RESULT.json"), read(OUT / "g0_lease.json"), read(OUT / "g0_GUARD.json")
    checks = {
        "all15_original_generic_priority_RAW_and_lifetime_checks": len(result["checks"]) == 15 and all(result["checks"].values()),
        "both_component_algorithms_below_preregistered5percent": all(value < .05 for value in result["component_reductions_not_full_acceptance"].values()) and not result["eligible_modes"] and result["chosen_mode_by_registered_single_selection"] is None,
        "no_pilot_full7338_or_acceptance_after_stop": result["status"] == "closed_negative_component_screen0" and result["new_optimization_count"] == 0 and not any((OUT / name).exists() for name in ("p0", "c0")),
        "all_current_executed_sources_and_library_exact": all(data_identity(Path(row["path"])) == row for row in result["sources"] + [result["library"]]),
        "actual_guard_and_lease_exited0_and_released": guard["returncode"] == lease["returncode"] == 0 and bool(lease["lease_released_utc"]),
        "V1_original_failure_receipts_and_final_review_retained": (OUT.parent / "lco001/c0/RESULT.json").is_file() and (OUT.parent / "lco001/NEGATIVE_V1_SELF_REVIEW_001.json").is_file() and read(OUT.parent / "lco001/FINAL_V1_POST_EXIT_001.json")["new_optimization_count"] == 0,
        "parent33_unique_no_R160_added": read(DOC / "CURRENT_STATE.json")["additional_accepted_count"] == 33 and not any(name.startswith("R160") for name in read(DOC / "CURRENT_STATE.json")["accepted_mechanisms"]),
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
    checks["latest_recorded_resource_owner_released"] = owner["owner"] == "root_R160_V2_exact_key_generic" and owner["status"] == "released"
    with (shared / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_acquired_and_released"] = True
    report = (DOC / "R160_EXACT_KEY_SORTING_V2_RESULTS.md").read_text(encoding="utf-8")
    task = (DOC / "R160_EXACT_KEY_SORTING_V2_TASK.md").read_text(encoding="utf-8")
    checks["report_original_text_scopes_counts_and_numbers_match"] = all(part in report for part in ("33/50", "1.638227%", "1.953974%", "15项", "chosen=null", "组件计时", "停止128"))
    checks["all_report_links_existing_or_own_final_receipt"] = all(Path(path).is_file() or Path(path) == target for path in re.findall(r"\]\((C:/[^)]+)\)", report))
    checks["preregistered_screen_and_quality_not_lowered_after_observation"] = all(part in task for part in ("少>=5%", "每对不慢于1.1", "净上界不足2%", "同属R160", "本版不得删除旧费用"))
    checks["first_closer_wrong_literal_and_executed_source_preserved"] = read(OUT / "NEGATIVE_V2_CLOSURE_FAILURE_001.json")["actual_tool_exit_code"] == 1 and (OUT / "negative_closer_source_001.py").is_file()
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    atomic_json(target, {"status": "closed_negative_exact_key_component_screen0", "checks": checks,
        "actual_generic_tool_chunk": "8b821a", "actual_generic_tool_exit_code": 0,
        "actual_generic_lease_seconds": lease["child_elapsed_seconds"], "owned_prior_instances": instances,
        "first_failed_closer_tool_wall_seconds": .2713795,
        "report_read_as_original_text": True, "independent_agent_review": False,
        "new_optimization_count": 0, "parent_accepted_count": 33, "goal_status": "active",
        "source": data_identity(Path(__file__)), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"checks": checks, "new_count": 0, "parent": 33}), flush=True)


if __name__ == "__main__":
    main()
