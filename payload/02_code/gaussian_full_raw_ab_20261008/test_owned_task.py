"""Verify successful, failing, and time-limited children through the adapted guard."""

import argparse
from pathlib import Path
import subprocess
import sys

from atomic_json_windows import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    rows = []
    for label, seconds, exitcode, budget, expected in (
        ("success", 0.1, 0, 3, "completed"),
        ("failed_child", 0.1, 7, 3, "failed_child"),
        ("time_guard", 5.0, 0, 1, "failed_controller")):
        command = [sys.executable, str(Path(__file__).with_name("run_owned_task.py")),
            "--receipt", str(args.output / (label + ".json")), "--log", str(args.output / (label + ".log")),
            "--maximum-seconds", str(budget), "--maximum-rss-gib", "1", "--", sys.executable, "-c",
            "import time,sys; time.sleep(" + str(seconds) + "); sys.exit(" + str(exitcode) + ")"]
        run = subprocess.run(command, capture_output=True, text=True)
        receipt = read_json(args.output / (label + ".json"))
        assert receipt["status"] == expected
        assert (run.returncode == 0) == (label == "success")
        assert receipt["receipt_publication_before_current_write"]["original_process_guard_unchanged"]
        rows.append({"label": label, "status": receipt["status"], "guard_returncode": run.returncode,
                     "child_returncode": receipt["returncode"], "elapsed_seconds": receipt["elapsed_seconds"]})
    atomic_json(args.output / "RESULT.json", {"status": "completed", "rows": rows,
        "RSS_guard_source_unchanged_not_separately_stress_tested": True})
    print(rows)


if __name__ == "__main__":
    main()
