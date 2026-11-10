"""Exercise real mutual exclusion and release after a failing child."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_with_resource_lease import validate_prior_owner


ROOT = Path(__file__).resolve().parents[2]
GUARD = Path(__file__).with_name("run_with_resource_lease.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs")
    output.mkdir(parents=True, exist_ok=False)
    started = output / "started.txt"
    done = output / "done.txt"
    common = [sys.executable, str(GUARD), "--wait-seconds", "10"]
    first = subprocess.Popen(common + ["--owner", "guard_oracle_first", "--receipt",
        str(output / "first.json"), "--", sys.executable, "-c",
        "from pathlib import Path; import sys,time; "
        "Path(sys.argv[1]).write_text('started'); time.sleep(0.35); "
        "Path(sys.argv[2]).write_text('done')", str(started), str(done)])
    deadline = time.monotonic() + 5
    while not started.exists():
        if time.monotonic() >= deadline or first.poll() is not None:
            raise RuntimeError("The first real lease child did not start")
        time.sleep(0.01)
    second = subprocess.run(common + ["--owner", "guard_oracle_second", "--receipt",
        str(output / "second.json"), "--", sys.executable, "-c",
        "from pathlib import Path; import sys; assert Path(sys.argv[1]).read_text() == 'done'",
        str(done)], check=False, timeout=10)
    first_code = first.wait(timeout=5)
    failing = subprocess.run(common + ["--owner", "guard_oracle_failure", "--receipt",
        str(output / "failure.json"), "--", sys.executable, "-c", "raise SystemExit(7)"],
        check=False, timeout=10)
    after = subprocess.run(common + ["--owner", "guard_oracle_after_failure", "--receipt",
        str(output / "after.json"), "--", sys.executable, "-c", "raise SystemExit(0)"],
        check=False, timeout=10)
    traces = {name: json.loads((output / (name + ".json")).read_text(encoding="utf-8"))
              for name in ("first", "second", "failure", "after")}
    abandoned = output / "abandoned_owner.json"
    abandoned.write_text(json.dumps({"status": "running", "guard_pid": -1}), encoding="utf-8")
    abandoned_rejected = False
    try:
        validate_prior_owner(abandoned)
    except RuntimeError as error:
        abandoned_rejected = "inspect its Windows and WSL descendants" in str(error)
    checks = {
        "two_actual_children_do_not_overlap": first_code == second.returncode == 0,
        "second_waited_for_first_real_child": traces["second"]["wait_seconds"] >= 0.25,
        "child_error_exitcode_preserved": failing.returncode == 7,
        "real_lock_released_after_failure": after.returncode == 0,
        "all_leases_released_and_all_owned_children_closed": all(
            "lease_released_utc" in value for value in traces.values()),
        "abandoned_owner_is_rejected_before_new_child_execution": abandoned_rejected,
    }
    result = {"status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "traces": traces, "optimization_count": 0}
    (output / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
