"""Run dependent bounded steps only after explicit exit and receipt success."""

import json
from pathlib import Path
import subprocess
import time


def run_checked_sequence(steps, receipt):
    receipt = Path(receipt)
    if receipt.exists():
        raise FileExistsError("A dependent sequence receipt must be new")
    for step in steps:
        if not step["command"] or any(not isinstance(value, str) for value in step["command"]):
            raise ValueError("A concrete shell-free bounded command is required")
        if ("result" in step) != ("accepted_statuses" in step):
            raise ValueError("A result receipt requires its preregistered successful statuses")
    rows = []
    status = "completed"
    for step in steps:
        started = time.perf_counter()
        process = subprocess.Popen(step["command"])
        try:
            code = process.wait()
        except BaseException:
            process.wait()
            raise
        row = {"name": step["name"], "returncode": code, "seconds": time.perf_counter() - started,
               "command": step["command"], "success": code == 0}
        if row["success"] and "result" in step:
            actual = json.loads(Path(step["result"]).read_text(encoding="utf-8"))
            row["result_status"] = actual["status"]
            row["success"] = row["result_status"] in step["accepted_statuses"]
        rows.append(row)
        if not row["success"]:
            status = "stopped_at_failed_dependency"
            break
    result = {"status": status, "rows": rows, "submitted": len(rows), "planned": len(steps),
              "remaining_steps_not_started": [step["name"] for step in steps[len(rows):]], "new_optimization_count": 0}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result
