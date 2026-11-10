"""Reproduce native commit-event ordering at an existing oracle failure."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    activate_reference(args.runtime)
    import torch
    from scene.gaussian_model import GaussianModel
    from ply_serialization import vectorized_ply_model
    from streamed_gaussian_ply_oracle_002 import NAMES, streamed_ply_model
    classes = {"native": vectorized_ply_model(GaussianModel, True), "stream": streamed_ply_model(GaussianModel)}
    torch.manual_seed(127)
    values = [torch.randn(shape, device="cuda") for shape in ((7, 3), (7, 1, 3), (7, 15, 3), (7, 1), (7, 3), (7, 4))]
    rows = []
    for previous in (False, True):
        states = {}
        for mode, cls in classes.items():
            model = cls(3)
            for name, value in zip(NAMES, values): setattr(model, name, torch.nn.Parameter(value.clone()))
            folder = args.output / f"previous{int(previous)}_{mode}"
            folder.mkdir()
            if previous:
                model.save_ply(str(folder / "success/point_cloud.ply"))
            count_before = len(getattr(model, "ply_serialization_events", []))
            attr_before = hasattr(model, "ply_serialization_events")
            (folder / "native_oracle.ply").write_bytes(b"protected")
            error = None
            try: model.save_ply(str(folder / "point_cloud.ply"))
            except FileExistsError as caught: error = str(caught)
            states[mode] = {"events_before": count_before, "events_after": len(getattr(model, "ply_serialization_events", [])),
                "attribute_before": attr_before, "attribute_after": hasattr(model, "ply_serialization_events"),
                "error": error, "protected_oracle": (folder / "native_oracle.ply").read_bytes() == b"protected"}
        rows.append({"previous_success": previous, "states": states,
            "original_event_commit_guard_matches": states["native"] == states["stream"]})
    result = {"status": "original_event_commit_guard_regression_passed", "rows": rows,
        "both_original_event_guards_matched": len(rows) == 2 and all(row["original_event_commit_guard_matches"] for row in rows),
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("streamed_gaussian_ply_oracle_002.py"))],
        "new_optimization_count": 0}
    atomic_json(args.output / "RESULT.json", result)
    print(result, flush=True)
    if not result["both_original_event_guards_matched"]: raise AssertionError("The original event commit regression failed")


if __name__ == "__main__": main()
