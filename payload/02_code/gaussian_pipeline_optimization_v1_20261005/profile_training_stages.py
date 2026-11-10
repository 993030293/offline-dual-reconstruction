"""Measure actual CUDA stages on a completed model without an optimization claim."""

import argparse
import json
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np

from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import camera_loader_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--gsplat-build", type=Path)
    parser.add_argument("--ssim-source", type=Path)
    parser.add_argument("--ssim-library", type=Path)
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--compact-image-cache", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads((args.model_run / "STATE.json").read_text())
    if state["status"] != "completed" or state["iteration"] != args.iterations:
        raise ValueError("Profiling requires the actually completed reference model")
    metadata = activate_reference(args.runtime)
    backend = None
    if args.gsplat_build:
        from gsplat_adapter import activate_gsplat
        backend = activate_gsplat(args.gsplat_build)
    import torch
    free, _ = torch.cuda.mem_get_info()
    gpu_budget = min(12*1024**3, free-2*1024**3)
    if gpu_budget <= 0:
        raise RuntimeError("Insufficient GPU headroom for an isolated profile")
    torch.cuda.reset_peak_memory_stats()
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    if args.gsplat_build:
        from gsplat_adapter import render
    from utils.loss_utils import l1_loss, ssim
    optional_ssim = None
    if args.ssim_source or args.ssim_library:
        if args.ssim_source is None or args.ssim_library is None:
            raise ValueError("Both isolated SSIM paths are required")
        extensions = list(args.ssim_library.glob("fused_ssim_cuda*.so"))
        if len(extensions) != 1:
            raise ValueError("The explicit SSIM library is missing or ambiguous")
        optional_ssim = {"source": data_identity(args.ssim_source/"fused_ssim/__init__.py"),
                         "extension": data_identity(extensions[0])}
        sys.path[:0] = [str(args.ssim_source), str(args.ssim_library)]
        from fused_ssim import fused_ssim
        def selected_ssim(prediction, target):
            return fused_ssim(prediction[None], target[None])
    else:
        selected_ssim = ssim
    if args.fused_adam:
        from optimizer_adapter import fused_adam_model
        GaussianModel = fused_adam_model(GaussianModel)

    native_parser = argparse.ArgumentParser()
    groups = [ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)]
    native = native_parser.parse_args([])
    for key, value in json.loads((args.runtime / "MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(native, key, value)
    native.source_path = str(args.dataset.resolve())
    native.model_path = str((args.model_run / "model").resolve())
    dataset, optimization, pipeline = [group.extract(native) for group in groups]
    random.seed(0)
    torch.manual_seed(0)
    gaussian = GaussianModel(3)
    with camera_loader_adapter(args.dataset, enabled=True):
        scene = Scene(dataset, gaussian, load_iteration=args.iterations, shuffle=False)
    cameras = scene.getTrainCameras()
    image_cache = None
    if args.compact_image_cache:
        from resident_image_cache import ResidentImageCache
        image_cache = ResidentImageCache(cameras, 4*1024**3, gpu_budget)
    if dataset.train_test_exp:
        raise ValueError("This diagnostic preserves the frozen disabled exposure policy")
    gaussian.pretrained_exposures = None
    gaussian.exposure_mapping = {camera.image_name: i for i, camera in enumerate(cameras)}
    gaussian._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(len(cameras), 1, 1))
    gaussian.training_setup(optimization)
    gaussian.update_learning_rate(args.iterations-1)
    background = torch.zeros(3, device="cuda")
    stages = ["render", "alpha_H2D_and_multiply", "ground_truth_H2D", "L1", "SSIM_and_loss", "backward", "Adam_and_zero_grad"]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(len(stages)+1)]
    rows = []
    started = time.monotonic()
    atomic_json(args.output / "CONTRACT.json", {"model_run": str(args.model_run), "iterations": args.iterations,
        "backend": backend, "maximum_loop_seconds": 300, "timed_steps": 60, "warmup_steps": 5,
        "optional_ssim": optional_ssim, "fused_adam": args.fused_adam,
        "compact_image_cache": args.compact_image_cache,
        "gpu_budget_bytes": gpu_budget, "ram_budget_bytes": 32*1024**3,
        "source": data_identity(Path(__file__)), "counted": False})
    for index in range(65):
        camera = cameras[(index*17) % len(cameras)]
        begin = time.monotonic()
        events[0].record()
        if image_cache:
            image_cache.activate(camera)
        package = render(camera, gaussian, pipeline, background)
        prediction = package["render"]
        events[1].record()
        if camera.alpha_mask is not None:
            prediction *= camera.alpha_mask.cuda()
        events[2].record()
        target = camera.original_image.cuda()
        events[3].record()
        l1 = l1_loss(prediction, target)
        events[4].record()
        loss = (1-optimization.lambda_dssim)*l1 + optimization.lambda_dssim*(1-selected_ssim(prediction, target))
        events[5].record()
        loss.backward()
        events[6].record()
        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)
        events[7].record()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-begin
        if image_cache:
            image_cache.release()
        if time.monotonic()-started > 300:
            raise TimeoutError("The bounded profiler exceeded its loop budget")
        if (torch.cuda.max_memory_reserved() > gpu_budget
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 32*1024**3):
            raise MemoryError("The bounded profiler exceeded its memory budget")
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Profiler encountered a non-finite loss")
        if index >= 5:
            rows.append({"trial": (index-5)//20, "camera": camera.image_name, "wall_ms": elapsed*1000,
                         "gpu_ms": {stage: events[i].elapsed_time(events[i+1]) for i, stage in enumerate(stages)}})
    result = {"status": "diagnostic_completed", "reference_runtime": metadata,
        "model": data_identity(args.model_run / f"model/point_cloud/iteration_{args.iterations}/point_cloud.ply"),
        "backend": backend, "iterations": args.iterations,
        "optional_ssim": optional_ssim, "fused_adam": args.fused_adam,
        "resident_image_cache": image_cache.finish() if image_cache else None,
        "maximum_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "source": data_identity(Path(__file__)), "camera_count": len(cameras), "gaussian_count": gaussian.get_xyz.shape[0],
        "same_world_and_full_K": True, "source_model_modified": False, "warmup_steps": 5,
        "timed_steps": 60, "in_memory_steps_with_no_densification": True, "counts_as_optimization": False,
        "boundary": "mature-model stage diagnostic only; excludes loading/saving/densification and full optimization trajectory",
        "stage_gpu_ms_p50_p95": {stage: np.quantile([row["gpu_ms"][stage] for row in rows], [.5, .95]).tolist() for stage in stages},
        "wall_ms_p50_p95": np.quantile([row["wall_ms"] for row in rows], [.5, .95]).tolist(),
        "trial_wall_ms_medians": [float(np.median([row["wall_ms"] for row in rows if row["trial"] == trial])) for trial in range(3)],
        "elapsed_seconds": time.monotonic()-started, "rows": rows}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "stage_gpu_ms_p50_p95", "wall_ms_p50_p95")}))


if __name__ == "__main__":
    main()
