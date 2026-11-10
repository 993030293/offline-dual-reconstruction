"""Close the negative version only after its real reviewer process has exited."""

import datetime
import json
import msvcrt
from pathlib import Path
import re
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    target = OUT / "FINAL_V1_POST_EXIT_001.json"
    if target.exists():
        raise FileExistsError("The actual post-exit record cannot be overwritten")
    review = read(OUT / "NEGATIVE_V1_SELF_REVIEW_002.json")
    checks = {
        "separate_corrected_review24_checks_passed": len(review["checks"]) == 24 and all(review["checks"].values()),
        "executed_corrected_reviewer_source_current": review["source"] == data_identity(Path(__file__).with_name("review_lazy_corner_negative.py")),
        "parent33_unique_no_new_acceptance": read(DOC / "CURRENT_STATE.json")["additional_accepted_count"] == 33,
    }
    instances = review["Windows_saved_instances"] + [review["reviewer"]]
    for name, expected in (("v1_negative_review", 1), ("v1_negative_review_002", 0)):
        lease, guard = read(OUT / (name + "_lease.json")), read(OUT / (name + "_GUARD.json"))
        checks[name + "_actual_exit_and_release"] = lease["returncode"] == guard["returncode"] == expected and bool(lease["lease_released_utc"])
        instances += guard["owned_process_identities"]
        instances += [{"pid": lease[key + "pid"], "creation_time": lease[key + "creation_time"]} for key in ("guard_", "child_")]
    instances = list({(row["pid"], row["creation_time"]): row for row in instances}.values())
    alive = []
    for row in instances:
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                alive.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_recorded_previous_owner_and_review_PID_birth_instances_ended"] = not alive
    shared = OUT.parent / "coordination_001"
    owner = read(shared / "LOCK_OWNER.json")
    checks["latest_recorded_lease_released"] = owner["status"] == "released" and owner["owner"] == "root_R160_V1_negative_review_corrected"
    with (shared / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["actual_OS_lock_acquired_and_released"] = True
    report = (DOC / "R160_LAZY_CORNER_ORDER_V1_RESULTS.md").read_text(encoding="utf-8")
    links = [Path(path) for path in re.findall(r"\]\((C:/[^)]+)\)", report)]
    checks["report_original_text_and_all_link_targets_checked"] = all(part in report for part in ("33/50", "0.070974%", "0.060567%", "复核exit1", "第2/3")) and all(path.is_file() or path == target for path in links)
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    atomic_json(target, {"status": "closed_negative_actual_corrected_review_exited0_no_new_count",
        "checks": checks, "new_optimization_count": 0, "parent_accepted_count": 33,
        "goal_status": "active", "actual_corrected_review_tool_chunk": "4e94e2",
        "actual_corrected_review_tool_exit_code": 0, "actual_corrected_review_tool_wall_seconds": 5.0711032,
        "actual_failed_review_tool_chunk": "95b6fb", "actual_failed_review_tool_exit_code": 1,
        "Windows_saved_and_review_instances_ended": len(instances),
        "first_failed_review_lease_seconds": read(OUT / "v1_negative_review_lease.json")["child_elapsed_seconds"],
        "corrected_review_lease_seconds": read(OUT / "v1_negative_review_002_lease.json")["child_elapsed_seconds"],
        "original_report_read_as_text": True, "independent_agent_review": False,
        "remaining_formal_public_full_cancel_recovery_training_production": "not_run_after_registered_first_complete_negative",
        "source": data_identity(Path(__file__)),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"status": "closed_negative", "checks": checks, "ended_instances": len(instances), "new_count": 0}), flush=True)


if __name__ == "__main__":
    main()
