"""Verify the complete original raw outputs under the common resource lease."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json
from raw_initial_invariants import compare_initial_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The full raw comparison receipt must be new")
    started = time.perf_counter()
    rows = {"pair": compare_initial_outputs(args.left, args.right),
        "current_to_left": compare_initial_outputs(args.reference, args.left),
        "current_to_right": compare_initial_outputs(args.reference, args.right)}
    result = {"status": "passed" if all(row["all_protected"] for row in rows.values()) else "failed_verification",
        "rows": rows, "verification_seconds": time.perf_counter()-started, "counted": False}
    atomic_json(args.output, result)
    if result["status"] != "passed":
        raise ValueError("A complete current original raw output changed")


if __name__ == "__main__":
    main()
