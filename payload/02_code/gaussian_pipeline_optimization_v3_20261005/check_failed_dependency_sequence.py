"""Verify that actual process or receipt failures never launch the next step."""

from pathlib import Path
import sys
import time

import run_inference_property_task as native
from checked_campaign_sequence import run_checked_sequence


def main():
    started = time.perf_counter()
    output = native.OUT / "s0"
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    for mode in ("exit", "receipt"):
        folder = output / mode
        folder.mkdir()
        marker, failed = folder / "SHOULD_NOT_EXIST", folder / "FAILED.json"
        first = {"name": "failed_prerequisite", "command": [sys.executable, "-c", "raise SystemExit(17)"]}
        if mode == "receipt":
            code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{\"status\":\"failed_whole_screen\"}',encoding='utf-8')"
            first = {"name": "receipt_failed_prerequisite", "command": [sys.executable, "-c", code, str(failed)],
                "result": str(failed), "accepted_statuses": ["complete_pair_protected"]}
        second = {"name": "must_not_start", "command": [sys.executable, "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('unexpected',encoding='utf-8')", str(marker)]}
        result = run_checked_sequence([first, second], folder / "RESULT.json")
        checks[mode + "_actual_failure_stops_before_next_process"] = (result["status"] == "stopped_at_failed_dependency"
            and result["submitted"] == 1 and result["remaining_steps_not_started"] == ["must_not_start"] and not marker.exists())
    native.atomic_json(output / "RESULT.json", {"status": "failed_dependency_regressions_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": [native.data_identity(path) for path in (Path(__file__), native.CODE / "checked_campaign_sequence.py")],
        "complete_seconds": time.perf_counter() - started, "scope": "workflow_bug_regression_not_a_new_algorithm_optimization",
        "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("A failed dependency launched a forbidden subsequent task")


if __name__ == "__main__":
    main()
