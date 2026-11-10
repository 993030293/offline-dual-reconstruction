"""Compare complete losses and prediction gradients on every actual development view."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from native_loss_execution import CachedWindowSSIM
from loss_algorithms import ConvolutionSSIM, CompiledL1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference_identity = activate_reference(args.runtime)
    import torch
    import utils.loss_utils as losses
    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from vectorized_gaussian_read import vectorized_gaussian_read
    identities = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("loss_algorithms.py"), args.model, args.localization)]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identities, "runtime": reference_identity,
        "all24_real_views": True, "trials": 3, "minimum_stage_reduction": .1,
        "loss_max_abs": 1e-7, "gradient_atol": 2e-6, "gradient_rtol": 1e-4,
        "precision_and_training_loss_definition_unchanged": True,
        "kernel_and_compile_first_cost_included_first_trial_and_reported": True,
        "warm_stage_not_a_cold_or_whole_training_speed_claim": True,
        "three_fresh12000_seed_pairs_required_before_training_acceptance": True,
        "common_CachedWindowSSIM_comparator_is_previous_unaccepted_Q57_for_isolating_new_algorithms": True,
        "maximum_cuda_bytes": 12*1024**3, "counted": False})
    localization = json.loads(args.localization.read_text())
    if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
        raise ValueError("The full actual24 development denominator is required")
    intrinsic = localization["PINHOLE_camera"]
    original = Pinhole(intrinsic["width"], intrinsic["height"], *intrinsic["params"])
    scaled = original.resized(round(original.width/2), round(original.height/2))
    native_parser = argparse.ArgumentParser()
    group = PipelineParams(native_parser)
    pipeline = group.extract(native_parser.parse_args([]))
    model = GaussianModel(3)
    with vectorized_gaussian_read(True):
        model.load_ply(str(args.model))
    background = torch.zeros(3, device="cuda")
    inputs = []
    for row in localization["rows"]:
        with Image.open(mounted_path(row["image_reference"])) as photo:
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
                2*math.atan(original.width/(2*original.fx)), 2*math.atan(original.height/(2*original.fy)),
                None, photo.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
        apply_to_camera(camera, scaled)
        with torch.no_grad():
            predicted = render(camera, model, pipeline, background)["render"]
        inputs.append((row["image_name"], predicted.detach(), camera.original_image.cuda()))
    torch.cuda.synchronize()
    common = CachedWindowSSIM(losses)
    mechanisms = {"R86_batched": ConvolutionSSIM(losses, "batched"),
        "R87_separable": ConvolutionSSIM(losses, "separable"), "R88_compiled_l1": CompiledL1()}
    results = {}
    for name, method in mechanisms.items():
        candidates = {"incumbent": losses.l1_loss if name == "R88_compiled_l1" else losses.ssim,
            "common_kernel": losses.l1_loss if name == "R88_compiled_l1" else common, "candidate": method}
        trials, numerical = [], []
        try:
            for trial in range(3):
                timings, values, peaks = {}, {}, {}
                order = list(candidates) if trial % 2 == 0 else list(reversed(candidates))
                for mode in order:
                    outputs, elapsed = [], 0.
                    torch.cuda.reset_peak_memory_stats()
                    for image_name, predicted, reference in inputs:
                        current = predicted.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        value = candidates[mode](current, reference)
                        value.backward()
                        torch.cuda.synchronize()
                        elapsed += time.perf_counter()-before
                        outputs.append((float(value.detach()), current.grad.detach().cpu().clone()))
                    timings[mode], values[mode], peaks[mode] = elapsed, outputs, torch.cuda.max_memory_allocated()
                guards = []
                for index, (image_name, _, _) in enumerate(inputs):
                    a, da = values["incumbent"][index]
                    b, db = values["candidate"][index]
                    row = {"trial": trial, "image_name": image_name, "loss_max_abs": abs(a-b),
                        "gradient_max_abs": float((da-db).abs().max()),
                        "gradient_guard": bool(torch.allclose(da, db, atol=2e-6, rtol=1e-4)),
                        "value_bit_exact": a == b, "gradient_bit_exact": bool(torch.equal(da, db))}
                    numerical.append(row)
                    guards.append(row["loss_max_abs"] <= 1e-7 and row["gradient_guard"])
                row = {"trial": trial, "seconds": timings, "cuda_allocated_peak_bytes": peaks, "all24_original_guards": all(guards)}
                trials.append(row)
                atomic_json(args.output/"STATE.json", {"status": "running", "active": name, "results": results, "trials": trials})
                print(json.dumps({"candidate": name, **row}), flush=True)
                if not all(guards):
                    raise ValueError("An original real pixel loss or gradient guard failed")
            median = {mode: statistics.median(row["seconds"][mode] for row in trials) for mode in candidates}
            reduction = 1-median["candidate"]/median["incumbent"]
            isolated = 1-median["candidate"]/median["common_kernel"]
            results[name] = {"status": "stage_benefit_passed_three_seed_training_pending" if min(reduction, isolated) >= .1 else "rejected_insufficient_benefit",
                "trials": trials, "median_seconds": median, "reduction": reduction, "new_algorithm_isolated_reduction": isolated,
                "numerical": numerical, "trace": method.trace, "counted": False}
        except Exception as error:
            results[name] = {"status": "failed_original_guard_or_execution", "error": repr(error),
                "traceback": traceback.format_exc(), "trials": trials, "numerical": numerical, "trace": method.trace, "counted": False}
        atomic_json(args.output/"RESULT.json", {"status": "candidate_profile_progress", "results": results})
    unchanged = identities == [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("loss_algorithms.py"), args.model, args.localization)]
    atomic_json(args.output/"RESULT.json", {"status": "all_registered_loss_candidates_profiled", "results": results,
        "sources_and_inputs_unchanged": unchanged, "development_frames": 24, "three_seed_training_executed": False,
        "additional_accepted_count": 0})
    print(json.dumps({"status": "all_registered_loss_candidates_profiled", "results": {
        name: {key: value for key, value in result.items() if key not in ("numerical", "traceback")} for name, result in results.items()}}), flush=True)
    if not unchanged:
        raise ValueError("A loss candidate source or actual input changed")


if __name__ == "__main__":
    main()
