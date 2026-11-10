"""Protect every actual development feature, match, pixel and camera pose."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json
from compare_actual_localization import compare


def entry(root):
    results = list(root.rglob("H_LOCALIZATION.json"))
    if len(results) != 1 or results[0].parent.name != "localization":
        raise ValueError("Exactly one actual complete localization result is required")
    return results[0].parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    roots = [entry(path) for path in (args.left, args.right, args.reference)]
    results = []
    for a, b in ((0, 1), (2, 0), (2, 1)):
        results.append(compare(roots[a], roots[b], roots[a]/"NEIGHBOR_TRACE.json", roots[b]/"NEIGHBOR_TRACE.json"))
    passed = all(row["all_protected"] for row in results)
    atomic_json(args.output, {"status": "passed" if passed else "failed", "comparisons": results,
        "verification_seconds": time.perf_counter()-started, "counted": False})
    if not passed:
        raise ValueError("An actual complete current localization invariant failed")


if __name__ == "__main__":
    main()
