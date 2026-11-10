"""Preserve the first reviewer failure and match the actual parent-state enum explicitly."""

from pathlib import Path

original = Path(__file__).with_name("close_ordering_prefix_negative.py")
source = original.read_text(encoding="utf-8")


def once(before, after):
    global source
    if source.count(before) != 1:
        raise AssertionError("The negative-review source boundary changed")
    source = source.replace(before, after)


once('current["status"] == "active"', 'current["status"] == "active_first_measured_algorithm_pass"')
source = source.replace("NEGATIVE_SOURCE_COST_CLOSURE_001.json", "NEGATIVE_SOURCE_COST_CLOSURE_002.json").replace("failure_sources_001", "failure_sources_002")
once('    checks["all_earlier_owned_process_births_ended_and_leases_released"]', '''    prior_review = read(OWN / "close1_lease.json")
    if prior_review["returncode"] != 1 or not prior_review.get("lease_released_utc"):
        raise AssertionError("The original reviewer failure or its fee was not preserved")
    processes.extend([read(OWN / "NEGATIVE_SOURCE_COST_CLOSURE_001.json")["writer"],
                      {"pid": prior_review["guard_pid"], "creation_time": prior_review["guard_creation_time"]},
                      {"pid": prior_review["child_pid"], "creation_time": prior_review["child_creation_time"]}])
    times.append({"scope": "actual_failed_negative_review_close1", "seconds": prior_review["child_elapsed_seconds"], "actual_controller_returncode": 1})
    checks["first_review_failed_parent_enum_check_preserved_without_scientific_gate_change"] = not read(OWN / "NEGATIVE_SOURCE_COST_CLOSURE_001.json")["checks"]["parent_count_and_accepted_mechanisms_and_receipts_still34_goal_active"]
    checks["all_earlier_owned_process_births_ended_and_leases_released"]''')
once('"run_gl_resource_generation_generic_v2.cjs"]:', '"run_gl_resource_generation_generic_v2.cjs", "close_ordering_prefix_negative.py", "close_ordering_prefix_negative_v2.py"]:')
exec(compile(source, str(Path(__file__)), "exec"), globals())
