"""Inspect JSON receipts without denying Windows atomic replacement sharing."""

import argparse
import json
from pathlib import Path

from atomic_json_windows import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    fields = ("status", "pid", "process_birth", "current_case", "current_mode", "current_stage", "elapsed_seconds",
              "iteration", "target_iterations", "training_call_seconds", "gaussian_count", "error", "label")
    for path in args.paths:
        if not path.exists():
            print(json.dumps({"path": str(path), "status": "absent"}))
            continue
        value = read_json(path)
        row = {"path": str(path), **{name: value[name] for name in fields if name in value}}
        if "rows" in value:
            row["row_count"] = len(value["rows"])
        if "stages" in value:
            row["last_stage"] = {name: value["stages"][-1].get(name) for name in
                ("name", "stage", "status", "elapsed_seconds", "returncodes")} if value["stages"] else None
        print(json.dumps(row))


if __name__ == "__main__":
    main()
