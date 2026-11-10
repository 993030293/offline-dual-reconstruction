"""Compare native checkpoint prefixes without rerunning or promoting failed training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in (args.left, args.right)]
    rows = []

    def compare(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            same_shape = a.shape == b.shape
            rows.append({"path": path, "type": "tensor", "same_shape": same_shape, "left_shape": list(a.shape),
                "right_shape": list(b.shape), "exact": same_shape and torch.equal(a, b),
                "maximum_abs": float((a-b).abs().max()) if same_shape and a.numel() and a.is_floating_point() else None})
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            rows.append({"path": path, "type": "numpy", "exact": bool(np.array_equal(a, b))})
        elif isinstance(a, dict) and isinstance(b, dict):
            rows.append({"path": path+".keys", "type": "keys", "exact": list(a) == list(b)})
            for key in a.keys() & b.keys():
                compare(a[key], b[key], path+"."+str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            rows.append({"path": path+".length", "type": "length", "exact": len(a) == len(b)})
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+"."+str(index))
        else:
            rows.append({"path": path, "type": type(a).__name__, "exact": bool(a == b)})

    compare(left, right, "checkpoint")
    differences = [row for row in rows if not row["exact"]]
    result = {"status": "diagnostic_completed_not_optimization", "left": data_identity(args.left), "right": data_identity(args.right),
        "iteration": [left["iteration"], right["iteration"]], "all_rows": rows, "differences": differences,
        "prefix_already_different_before_crash": bool(differences), "causal_root_unconfirmed": True,
        "failed_quality_gate_unchanged": True, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "iteration", "differences", "causal_root_unconfirmed")}), flush=True)


if __name__ == "__main__":
    main()
