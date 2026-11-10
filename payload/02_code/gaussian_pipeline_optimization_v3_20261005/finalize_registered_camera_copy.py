"""Verify released benchmark ownership and publish a bounded negative-screen receipt."""

import json
import msvcrt
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json
from review_registered_camera_copy import same_instance


def main():
    started = time.perf_counter()
    own = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/registered_camera_copy_001"
    target = own / "FINAL_POST_EXIT_001.json"
    if target.exists():
        raise FileExistsError("A final screen receipt must not overwrite prior evidence")
    review = json.loads((own / "review_001/RESULT.json").read_text())
    reviewer = json.loads((own / "review_001/PROCESS_IDENTITY.json").read_text())
    leases = [json.loads((own / name).read_text()) for name in ("G0_LEASE.json", "P0_LEASE.json", "REVIEW_LEASE.json")]
    lock_path = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    lock_owner = json.loads(lock_path.with_name("LOCK_OWNER.json").read_text())
    with lock_path.open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            lock_free = True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    doc = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    task = (doc / "R82_REGISTERED_HOST_TRANSFER_TASK.md").read_text(encoding="utf-8")
    report = (doc / "R82_REGISTERED_HOST_TRANSFER_RESULTS.md").read_text(encoding="utf-8")
    state = json.loads((doc / "CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "separate_root_review13_true": len(review["checks"]) == 13 and all(review["checks"].values()),
        "reviewer_actual_instance_exited": not same_instance(reviewer["pid"], reviewer["creation_time"]),
        "all_three_leases_released_and_owned_windows_exited": all(row.get("returncode") == 0 and row.get("lease_released_utc")
            and not same_instance(row["guard_pid"], row["guard_creation_time"])
            and not same_instance(row["child_pid"], row["child_creation_time"]) for row in leases),
        "OS_lock_actually_available_and_last_owner_released": lock_free and lock_owner["status"] == "released"
            and lock_owner["owner"] == leases[-1]["owner"],
        "native_text_preserves_preregistered5percent_negative": "5%" in task and "4.925254%" in report and "216" in task and "贡献 0" in report,
        "native_text_distinguishes_not_run_and_same_family": "formal" in report and "checkpoint" in report
            and "已有未接受 R82" in report and "1016/12000" in report,
        "parent22_no_live_candidate_and_scoped_no_count": state["additional_accepted_count"] == 22
            and state["active_candidates"] == [] and state["R82_registered_new_optimization_count"] == 0,
    }
    writer = psutil.Process()
    receipt = {"status": "bounded_negative_screen_closed" if all(checks.values()) else "closure_failed",
        "checks": checks, "counted": False, "review_kind": "single_root_separate_self_review",
        "known_new_measured_lease_child_seconds": sum(row["child_elapsed_seconds"] for row in leases),
        "formal_complete_training_cost_not_computed": True, "three_leases": leases,
        "reviewer_actual_instance_exited": checks["reviewer_actual_instance_exited"],
        "writer_identity": {"pid": writer.pid, "creation_time": writer.create_time()},
        "writer_actual_exit_check_pending": True, "closure_seconds_before_receipt": time.perf_counter()-started}
    atomic_json(target, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != "three_leases"}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An actual resource or native text closure guard failed")


if __name__ == "__main__":
    main()
