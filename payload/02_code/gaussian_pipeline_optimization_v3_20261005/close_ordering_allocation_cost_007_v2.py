"""Use recorded process births without guessing a fixed diagnostic process count."""

from pathlib import Path

original = Path(__file__).with_name("close_ordering_allocation_cost_007.py")
source = original.read_text(encoding="utf-8")


def once(before, after):
    global source
    if source.count(before) != 1:
        raise AssertionError("The allocation-cost review source boundary changed")
    source = source.replace(before, after)


source = source.replace("SOURCE_COST_CLOSURE_001.json", "SOURCE_COST_CLOSURE_002.json").replace("failure_sources_001", "failure_sources_002")
once('    checks["all24_previous_owned_births_ended_and_all4_leases_released"] = len(processes) == 24 and all(closed(row) for row in processes)', '''    previous_review = read(OWN / "close1_lease.json")
    previous_receipt = read(OWN / "SOURCE_COST_CLOSURE_001.json")
    if previous_review["returncode"] != 1 or not previous_review.get("lease_released_utc"):
        raise AssertionError("The first reviewer failure and fee were not retained")
    processes.extend([previous_receipt["writer"],
                      {"pid": previous_review["guard_pid"], "creation_time": previous_review["guard_creation_time"]},
                      {"pid": previous_review["child_pid"], "creation_time": previous_review["child_creation_time"]}])
    fees.append({"scope": "failed_previous_review", "seconds": previous_review["child_elapsed_seconds"], "returncode": 1})
    checks["first_wrong_fixed_process_count_review_preserved"] = not previous_receipt["checks"]["all24_previous_owned_births_ended_and_all4_leases_released"]
    checks["all_recorded_previous_process_births_ended_and_all_leases_released"] = bool(processes) and all(closed(row) for row in processes)''')
exec(compile(source, str(Path(__file__)), "exec"), globals())
