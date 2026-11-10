"""Reproduce a reservation boundary with two interleaved native models."""

import json
from pathlib import Path
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/aw001/b0"
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(OUT / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=300, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from scene.gaussian_model import GaussianModel
        from inference_property_workspace import FIELDS, InferencePropertyWorkspace
        workspace = InferencePropertyWorkspace(GaussianModel, True, maximum_bytes=6000)
        models = [workspace.model_class(3) for _ in range(2)]
        shapes = ((17, 3), (17, 1, 3), (17, 15, 3), (17, 1), (17, 3), (17, 4))
        for model in models:
            for name, shape in zip(FIELDS, shapes):
                setattr(model, name, torch.nn.Parameter(torch.ones(shape, device="cuda")))
        caught = None
        with torch.no_grad():
            models[0].get_opacity
            models[1].get_opacity
            models[0].get_features
            try:
                models[1].get_features
            except AssertionError as error:
                caught = repr(error)
        workspace.close()
        result = {"status": "reproduced_missing_full_entry_reservation" if caught else "not_reproduced",
            "error": caught, "workspace": workspace.trace,
            "source": data_identity(CODE / "inference_property_workspace.py"),
            "scope": "two_17_row_native_models_interleaved_properties_6000_bytes_original_getters_succeed",
            "complete_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
        atomic_json(OUT / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "error": caught}), flush=True)
        if not caught:
            raise AssertionError("The source-audit budget failure was not reproduced")


if __name__ == "__main__":
    main()
