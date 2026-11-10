"""Measure native SSIM execution and gradients on all real calibrated development views."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from native_loss_execution import CachedWindowSSIM, GraphSSIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    import utils.loss_utils as losses
    localization = json.loads(args.localization.read_text())
    if len(localization["rows"]) != 24 or any(r["status"] != "localized" for r in localization["rows"]):
        raise ValueError("Every original real development view is required for native loss profiling")
    protocol = {"runtime": metadata, "model": data_identity(args.model), "localization": data_identity(args.localization),
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("native_loss_execution.py"))],
        "all24_real_predictions_and_reference_pixels": True, "trials": 3,
        "minimum_actual_loss_forward_and_backward_wall_reduction": .1,
        "kernel_and_graph_construction_in_first_trial_and_separate_cold_cost": True,
        "loss_max_abs": 1e-7, "gradient_atol": 2e-6, "gradient_rtol": 1e-4,
        "no_mixed_precision_or_new_loss_formula": True, "three_seed_training_quality_required": True,
        "maximum_cuda_bytes": 12*1024**3, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    torch.cuda.reset_peak_memory_stats()
    model = GaussianModel(3)
    model.load_ply(str(args.model))
    native_parser = argparse.ArgumentParser()
    group = PipelineParams(native_parser)
    pipeline = group.extract(native_parser.parse_args([]))
    intr = localization["PINHOLE_camera"]
    original = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = original.resized(round(original.width/2), round(original.height/2))
    background = torch.zeros(3, device="cuda")
    inputs = []
    for row in localization["rows"]:
        photo = Image.open(mounted_path(row["image_reference"])).convert("RGB")
        camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
            2*math.atan(original.width/(2*original.fx)), 2*math.atan(original.height/(2*original.fy)),
            None, photo, None, row["image_name"], 0, data_device="cpu")
        apply_to_camera(camera, scaled)
        with torch.no_grad():
            predicted = render(camera, model, pipeline, background)["render"]
        inputs.append((row["image_name"], predicted.detach(), camera.original_image.cuda()))
    methods = {"baseline": losses.ssim, "kernel": CachedWindowSSIM(losses), "graph": GraphSSIM(losses)}
    a, b = inputs[0][1].detach().clone().requires_grad_(True), inputs[0][2]
    losses.ssim(a, b).backward()
    torch.cuda.synchronize()
    rows, trials = [], []
    try:
        for trial in range(3):
            order = ("baseline", "kernel", "graph") if trial % 2 == 0 else ("graph", "kernel", "baseline")
            outputs, seconds = {}, {}
            for name in order:
                outputs[name] = []
                elapsed = 0.
                for image_name, predicted, reference in inputs:
                    value = predicted.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = methods[name](value, reference)
                    result.backward()
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter()-start
                    outputs[name].append((float(result.detach()), value.grad.detach().cpu().clone()))
                seconds[name] = elapsed
            checks = []
            for index, (image_name, _, _) in enumerate(inputs):
                original_loss, original_grad = outputs["baseline"][index]
                for name in ("kernel", "graph"):
                    value, gradient = outputs[name][index]
                    item = {"trial": trial, "name": image_name, "mode": name,
                        "loss_abs": abs(value-original_loss), "loss_bit_exact": value == original_loss,
                        "gradient_max_abs": float((gradient-original_grad).abs().max()),
                        "gradient_bit_exact": torch.equal(gradient, original_grad),
                        "gradient_guard": torch.allclose(gradient, original_grad, atol=2e-6, rtol=1e-4)}
                    checks.append(item["loss_abs"] <= 1e-7 and item["gradient_guard"])
                    rows.append(item)
            item = {"trial": trial, "actual_forward_backward_wall_seconds": seconds, "all24_value_and_gradient_guards": all(checks)}
            trials.append(item)
            atomic_json(args.output/"STATE.json", {"status": "measuring", "trials": trials, "counted": False})
            print(json.dumps(item), flush=True)
            if not all(checks):
                raise ValueError("A real native SSIM value or gradient failed its original precision guard")
        median = {mode: statistics.median(t["actual_forward_backward_wall_seconds"][mode] for t in trials) for mode in methods}
        results = {mode: {"median_seconds": median[mode], "reduction": 1-median[mode]/median["baseline"],
            "status": "stage_benefit_passed_full_training_pending" if 1-median[mode]/median["baseline"] >= .1 else "rejected_insufficient_benefit",
            "mechanism": "immutable_native_window" if mode == "kernel" else "native_forward_backward_CUDA_graph",
            "trace": methods[mode].trace, "counted": False} for mode in ("kernel", "graph")}
        peak = torch.cuda.max_memory_reserved()
        if peak > 12*1024**3:
            raise MemoryError("The native loss execution exceeded its declared CUDA budget")
        atomic_json(args.output/"RESULT.json", {"status": "real24_gradient_campaign_completed_training_pending",
            "results": results, "trials": trials, "all144_precision_checks": rows,
            "baseline_median_seconds": median["baseline"], "peak_CUDA_reserved_bytes": peak,
            "sources": protocol["sources"], "counted": False, "full_training_quality_pending": True})
    except BaseException as error:
        atomic_json(args.output/"STATE.json", {"status": "failed", "error": repr(error), "trials": trials,
            "precision_checks": rows, "counted": False})
        raise


if __name__ == "__main__":
    main()
