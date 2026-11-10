"""Record a standalone native receipt check independently of a service query."""

import json
from pathlib import Path
import subprocess
import sys
import time

from run_open_image_handoff_campaign import OUT
from review_scene_batch_negative import read, windows_closed
from runtime import atomic_json, data_identity


def main():
    original = read(OUT / "FINAL_POST_EXIT_001.json")
    if not all(original["checks"].values()) or not windows_closed(original["actual_last_checker_own_process"]):
        raise ValueError("The actual post-exit checker did not complete its native checks")
    code = "import json,pathlib; r=json.loads(pathlib.Path(" + repr(str(OUT / "FINAL_POST_EXIT_001.json")) + ").read_text()); assert all(r['checks'].values()) and r['parent_count']==30 and r['new_optimization_count']==1 and r['remaining']==20; print(json.dumps(dict(native_checks=len(r['checks']),parent=30,new=1,remaining=20)))"
    started = time.perf_counter()
    child = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if child.returncode:
        raise RuntimeError("The standalone native result check failed: " + child.stderr)
    actual = json.loads(child.stdout)
    cost = read(OUT / "FINAL_COST_001.json")
    fee = cost["known_fee_plus_registered10_reserve_seconds"] + 1.4823142 + elapsed
    cost.update(status="all_scoped_prior_publishers_and_finalizer_exited_standalone_native_post_check_actual_exit0_parent30_active",
                known_fee_plus_registered10_reserve_seconds=fee,
                previous_combined_readonly_diagnostic_tool={"chunk_id": "601856", "returncode": 1, "seconds": 1.4823142, "native_post_checks_all_true": True, "separate_native_python_exit_not_captured_in_that_combined_command": True},
                actual_standalone_native_child_exit={"returncode": child.returncode, "seconds": elapsed, "result": actual})
    atomic_json(OUT / "FINAL_COST_001.json", cost)
    atomic_json(OUT / "FINAL_TOOL_EXIT_001.json", {"status": "actual_post_checker_instance_closed_and_standalone_native_child_exit0",
        "checks": {"all_original12_post_exit_checks_true": all(original["checks"].values()), "actual_checker_own_instance_exited": windows_closed(original["actual_last_checker_own_process"]), "actual_standalone_native_child_exit0": child.returncode == 0},
        "actual_native_child": actual, "native_returncode": child.returncode,
        "actual_child_seconds": elapsed, "previous_combined_tool_exit": 1, "previous_combined_tool_chunk": "601856",
        "service_availability_not_established_by_the_quiet_port_query": True,
        "no_viewer_service_started_or_stopped": True, "source_image_work_in_this_text_and_JSON_check": 0,
        "parent_count": 30, "new_optimization_count": 1, "remaining": 20, "goal_remains_active": True,
        "source": data_identity(Path(__file__)), "known_fee_plus_registered10_reserve_seconds": fee})
    print({"actual_native_returncode": child.returncode, "native_post_checks": actual["native_checks"], "parent": 30, "new": 1, "remaining": 20, "goal": "active"})


if __name__ == "__main__":
    main()
