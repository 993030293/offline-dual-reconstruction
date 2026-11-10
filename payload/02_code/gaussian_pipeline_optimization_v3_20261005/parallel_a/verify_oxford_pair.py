"""Compare every original public pose and novelty output with exact equality."""

import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    with np.load(args.left/"DECISIONS.npz", allow_pickle=False) as a, np.load(args.right/"DECISIONS.npz", allow_pickle=False) as b:
        checks = {key: np.array_equal(a[key], b[key]) for key in ("timestamp_ns", "pose_xyz_xyzw", "novelty")}
        checks["same_complete_schema"] = a.files == b.files
    atomic_json(args.output, {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "verification_seconds": time.perf_counter()-started, "counted": False})
    if not all(checks.values()):
        raise ValueError("An original public raw pose or novelty output changed")


if __name__ == "__main__":
    main()
