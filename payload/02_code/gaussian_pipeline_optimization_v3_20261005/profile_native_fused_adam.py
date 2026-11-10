"""Screen installed fused Adam on complete saved-model training steps."""

import argparse
import copy
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path


NAMES = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fa001").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time(), "counted": False})
    runtime = activate_reference(args.runtime)
    import torch
    import torch.optim.adam as adam_module
    import diff_gaussian_rasterization
    import gsplat_adapter
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams, PipelineParams
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    if not torch.__version__.startswith("2.7.1+") or not torch.cuda.is_available():
        raise RuntimeError("The registered installed CUDA Adam runtime is required")
    torch.set_num_threads(4)
    torch.manual_seed(20261006)
    build = gsplat_adapter.activate_gsplat(args.build)
    localization = json.loads(args.localization.read_text(encoding="utf-8"))
    if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
        raise ValueError("The complete original development denominator is required")
    source_paths = [Path(__file__), Path(adam_module.__file__),
        Path(sys.modules[GaussianModel.__module__].__file__), Path(sys.modules[ssim.__module__].__file__),
        V1 / "runtime.py", V1 / "full_intrinsics.py", V1 / "gsplat_adapter.py",
        V2 / "vectorized_gaussian_read.py"]
    input_paths = [args.model, args.localization] + [mounted_path(row["image_reference"]) for row in localization["rows"]]
    identities = [data_identity(path) for path in source_paths + input_paths]
    atomic_json(args.output / "PROTOCOL.json", {"sources_inputs": identities, "runtime": runtime,
        "build": build, "torch": torch.__version__, "GPU": torch.cuda.get_device_name(),
        "SparseGaussianAdam_installed": hasattr(diff_gaussian_rasterization, "SparseGaussianAdam"),
        "orders": [["native", "fused"], ["fused", "native"], ["native", "fused"]],
        "steps_each_side": 216, "all24_development_cameras": True, "minimum_complete_step_reduction": .1,
        "kernel_first_call_in_first_side_cost": True, "original_loss_copy_and_synchronization": True,
        "stage_is_not_full_training_or_quality_acceptance": True, "counted": False})
    intr = localization["PINHOLE_camera"]
    native = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = native.resized(round(native.width / 2), round(native.height / 2))
    cameras = []
    for row in localization["rows"]:
        with Image.open(mounted_path(row["image_reference"])) as photo:
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T,
                np.asarray(row["t_CW"]), 2 * math.atan(native.width / (2 * native.fx)),
                2 * math.atan(native.height / (2 * native.fy)), None, photo.convert("RGB"),
                None, row["image_name"], 0, data_device="cpu")
        apply_to_camera(camera, scaled)
        cameras.append(camera)
    options_parser = argparse.ArgumentParser()
    opt_group, pipe_group = OptimizationParams(options_parser), PipelineParams(options_parser)
    options = options_parser.parse_args([])
    optimization, pipeline = opt_group.extract(options), pipe_group.extract(options)
    model = GaussianModel(3)
    with vectorized_gaussian_read(True):
        model.load_ply(str(args.model))
    model.pretrained_exposures = None
    model.exposure_mapping = {camera.image_name: i for i, camera in enumerate(cameras)}
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(len(cameras), 1, 1))
    model.spatial_lr_scale = 1.
    model.training_setup(optimization)
    snapshot = copy.deepcopy(model.capture())
    point_count = len(model.get_xyz)
    background = torch.zeros(3, device="cuda")
    free, _ = torch.cuda.mem_get_info()
    budget = min(12 * 1024**3, free - 2 * 1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient registered GPU headroom")
    trials = []
    for trial in range(3):
        sides, captured = {}, {}
        for mode in (("native", "fused") if trial % 2 == 0 else ("fused", "native")):
            model.restore(copy.deepcopy(snapshot), optimization)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            if mode == "fused":
                if model.optimizer.state:
                    raise ValueError("Only the registered identical empty optimizer state can be replaced")
                groups = [dict(group, foreach=False, fused=True) for group in model.optimizer.param_groups]
                model.optimizer = torch.optim.Adam(groups, lr=0., eps=1e-15, foreach=False, fused=True)
            model.update_learning_rate(12000)
            events, values = [], []
            for index in range(216):
                camera = cameras[index % 24]
                begin, after_backward, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                begin.record()
                package = gsplat_adapter.render(camera, model, pipeline, background)
                image = package["render"]
                if camera.alpha_mask is not None:
                    image *= camera.alpha_mask.cuda()
                target = camera.original_image.cuda()
                l1 = l1_loss(image, target)
                loss = .8 * l1 + .2 * (1 - ssim(image, target))
                loss.backward()
                after_backward.record()
                with torch.no_grad():
                    values.append(loss.item())
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                end.record()
                events.append((begin, after_backward, end))
                if torch.cuda.memory_reserved() > budget or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 32 * 1024**3:
                    raise MemoryError("The registered diagnostic memory bound was exceeded")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            gpu = {"render_copy_loss_backward_ms": sum(a.elapsed_time(b) for a, b, c in events),
                "original_scalar_sync_Adam_zero_grad_ms": sum(b.elapsed_time(c) for a, b, c in events)}
            complete = {name: {"parameter": getattr(model, name).detach().cpu(),
                **{key: model.optimizer.state[getattr(model, name)][key].detach().cpu()
                   for key in ("exp_avg", "exp_avg_sq", "step")}} for name in NAMES}
            captured[mode] = complete
            checks = {"all216_loss_values_finite": len(values) == 216 and all(math.isfinite(value) for value in values),
                "all_complete_parameters_and_moments_finite": all(bool(torch.isfinite(value).all())
                    for row in complete.values() for value in row.values()),
                "all_six_original_groups_and_point_count": len(model.optimizer.param_groups) == 6 and len(model.get_xyz) == point_count,
                "fused_flag_matches_registered_mode": all(bool(group["fused"]) == (mode == "fused") for group in model.optimizer.param_groups),
                "CUDA_budget": torch.cuda.max_memory_reserved() <= budget}
            sides[mode] = {"complete_step_seconds": elapsed, "GPU_event_components": gpu,
                "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "losses": values, "checks": checks}
            if not all(checks.values()):
                atomic_json(args.output / "FAILED_SIDE.json", {"trial": trial, "mode": mode, "side": sides[mode]})
                raise ValueError("A registered diagnostic denominator, finite-value or implementation guard failed")
        drift = {name: {key: {"bit_exact": bool(torch.equal(captured["native"][name][key], captured["fused"][name][key])),
            "max_abs": float((captured["native"][name][key] - captured["fused"][name][key]).abs().max())}
            for key in captured["native"][name]} for name in NAMES}
        row = {"trial": trial, "sides": sides, "complete_state_differences": drift,
            "max_loss_difference": max(abs(a - b) for a, b in zip(sides["native"]["losses"], sides["fused"]["losses"]))}
        trials.append(row)
        atomic_json(args.output / "STATE.json", {"status": "running", "trials": trials})
        print(json.dumps({"trial": trial, "seconds": {mode: value["complete_step_seconds"] for mode, value in sides.items()},
            "GPU_event_components": {mode: value["GPU_event_components"] for mode, value in sides.items()},
            "max_loss_difference": row["max_loss_difference"]}), flush=True)
        del captured
        torch.cuda.empty_cache()
    medians = {mode: statistics.median(row["sides"][mode]["complete_step_seconds"] for row in trials) for mode in ("native", "fused")}
    reduction = 1 - medians["fused"] / medians["native"]
    checks = {"all_three_registered_pairs": len(trials) == 3,
        "all_diagnostic_side_guards": all(all(side["checks"].values()) for row in trials for side in row["sides"].values()),
        "complete_sources_and_inputs_unchanged": identities == [data_identity(path) for path in source_paths + input_paths]}
    result = {"status": "diagnostic_speed_passed_full_numeric_quality_recovery_pending" if reduction >= .1 else "closed_negative_complete_mature_step_speed",
        "checks": checks, "trials": trials, "point_count": point_count, "median_seconds": medians,
        "complete_step_reduction": reduction, "whole_diagnostic_seconds": time.perf_counter() - started,
        "WSL_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "three_seed_full_training_executed": False, "native_complete_resume_executed": False,
        "new_optimization_count": 0, "scope": "saved_C212_model_216_step_screen_not_full_current_map_training"}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "point_count", "median_seconds", "complete_step_reduction", "whole_diagnostic_seconds")}), flush=True)
    if not all(checks.values()):
        raise ValueError("A complete diagnostic identity or resource guard failed")


if __name__ == "__main__":
    main()
