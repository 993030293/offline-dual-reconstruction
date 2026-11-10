"""Train one isolated, frozen-input control or full-intrinsics experiment."""

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import traceback

import numpy as np

from full_intrinsics import camera_loader_adapter
from runtime import activate_reference, atomic_json, data_identity


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection", choices=["legacy", "full"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--maximum-seconds", type=int, default=1800)
    parser.add_argument("--ssim-backend", choices=["reference", "fused"], default="reference")
    parser.add_argument("--ssim-source", type=Path)
    parser.add_argument("--ssim-library", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-file", type=Path)
    parser.add_argument("--simulate-crash-after", type=int)
    parser.add_argument("--rasterizer-library", type=Path)
    parser.add_argument("--optimizer-type", choices=["default", "sparse_adam"])
    parser.add_argument("--dense-adam-backend", choices=["native", "fused"], default="native")
    parser.add_argument("--input-version", default="C212_frozen")
    parser.add_argument("--expected-image-count", type=int, default=212)
    parser.add_argument("--gsplat-build", type=Path)
    parser.add_argument("--pyramid-until", type=int, default=0)
    parser.add_argument("--compact-image-cache", action="store_true")
    parser.add_argument("--target-density-roi", type=Path)
    parser.add_argument("--resident-fused-adam", action="store_true")
    parser.add_argument("--vectorized-ply", action="store_true")
    parser.add_argument("--verify-native-ply", action="store_true")
    parser.add_argument("--compact-host-images", action="store_true")
    parser.add_argument("--trim-host-after-load", action="store_true")
    parser.add_argument("--gpu-resident-inputs", action="store_true")
    args = parser.parse_args()
    if args.gpu_resident_inputs and (not args.trim_host_after_load or not args.gsplat_build
            or args.projection != "full" or args.compact_host_images or args.compact_image_cache
            or args.resident_fused_adam or args.pyramid_until or args.target_density_roi
            or args.dense_adam_backend != "native" or args.ssim_backend != "reference"
            or args.optimizer_type not in (None, "default")):
        raise ValueError("Native CUDA residency requires the isolated full-K dense RGB configuration")
    if args.compact_host_images and (not args.trim_host_after_load or not args.gsplat_build or args.projection != "full"
            or args.compact_image_cache or args.resident_fused_adam or args.pyramid_until
            or args.target_density_roi or args.dense_adam_backend != "native" or args.ssim_backend != "reference"
            or args.optimizer_type not in (None, "default") or args.checkpoint_every or args.resume_file
            or args.simulate_crash_after):
        raise ValueError("Host image storage first comparisons require the isolated frozen gsplat configuration")
    if args.verify_native_ply and not args.vectorized_ply:
        raise ValueError("The native byte oracle requires the vectorized serializer")
    if args.vectorized_ply and (args.compact_image_cache or args.resident_fused_adam
            or args.pyramid_until or args.target_density_roi or args.dense_adam_backend != "native"
            or args.ssim_backend != "reference" or args.optimizer_type not in (None, "default")):
        raise ValueError("The first serialization comparison excludes unaccepted training candidates")
    if args.resident_fused_adam:
        if args.compact_image_cache or args.dense_adam_backend != "native":
            raise ValueError("Use only the registered combined flag, without duplicate component overrides")
        args.compact_image_cache = True
        args.dense_adam_backend = "fused"
    if args.target_density_roi:
        if (not args.gsplat_build or args.projection != "full" or args.pyramid_until or args.compact_image_cache
                or args.checkpoint_every or args.resume_file or args.simulate_crash_after
                or args.ssim_backend != "reference" or args.dense_adam_backend != "native"
                or args.optimizer_type not in (None, "default") or args.input_version != "C212_frozen"
                or args.expected_image_count != 212):
            raise ValueError("Target density first comparison requires the frozen C212 gsplat configuration")
    if args.compact_image_cache:
        if (not args.gsplat_build or args.projection != "full" or args.pyramid_until or args.checkpoint_every
                or args.resume_file or args.simulate_crash_after or args.ssim_backend != "reference"
                or (args.dense_adam_backend != "native" and not args.resident_fused_adam)
                or args.optimizer_type not in (None, "default")):
            raise ValueError("Resident cache first comparison requires the frozen gsplat configuration without unverified combinations")
    if args.pyramid_until:
        if (args.iterations, args.pyramid_until) not in ((1200, 600), (12000, 5000)):
            raise ValueError("Only the registered resolution smoke or fixed curriculum is supported")
        if (not args.gsplat_build or args.projection != "full" or args.checkpoint_every
                or args.resume_file or args.simulate_crash_after or args.ssim_backend != "reference"
                or args.dense_adam_backend != "native" or args.optimizer_type not in (None, "default")):
            raise ValueError("Resolution staging requires the verified gsplat configuration without unverified recovery")
    if args.checkpoint_every < 0 or ((args.resume_file or args.simulate_crash_after) and not args.checkpoint_every):
        raise ValueError("Recovery requires a positive checkpoint interval")
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "STATE.json"
    state = {"status": "initializing", "pid": os.getpid(), "iteration": 0,
             "projection": args.projection, "seed": args.seed, "target_iterations": args.iterations}
    atomic_json(state_path, state)
    try:
        metadata = activate_reference(args.runtime, args.rasterizer_library)
        snapshot = args.output / "source_snapshot"
        snapshot.mkdir()
        implementation = []
        source_names = ["train_case.py", "full_intrinsics.py", "runtime.py"]
        if args.checkpoint_every:
            source_names.append("checkpoint_adapter.py")
        if args.dense_adam_backend == "fused":
            source_names.append("optimizer_adapter.py")
        if args.gsplat_build:
            source_names.append("gsplat_adapter.py")
        if args.pyramid_until:
            source_names.append("resolution_adapter.py")
        if args.compact_image_cache:
            source_names.append("resident_image_cache.py")
        if args.target_density_roi:
            source_names.append("target_density_adapter.py")
        if args.vectorized_ply:
            source_names.append("ply_serialization.py")
        if args.compact_host_images:
            source_names.append("host_image_storage.py")
        if args.trim_host_after_load:
            source_names.append("host_memory.py")
        for name in source_names:
            target = snapshot / name
            shutil.copyfile(Path(__file__).parent / name, target)
            implementation.append(data_identity(target))
        optional_ssim = None
        optional_gsplat = None
        if args.ssim_backend == "fused":
            if args.ssim_source is None or args.ssim_library is None:
                raise ValueError("Explicit isolated source and library paths are required for fused SSIM")
            libraries = list(args.ssim_library.glob("fused_ssim_cuda*.so"))
            if len(libraries) != 1:
                raise ValueError("The isolated fused SSIM extension is missing or ambiguous")
            optional_ssim = {"python": data_identity(args.ssim_source/"fused_ssim/__init__.py"),
                             "extension": data_identity(libraries[0])}
            sys.path[:0] = [str(args.ssim_source), str(args.ssim_library)]
        import torch
        import train as reference
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from plyfile import PlyData

        properties = set(PlyData.read(args.dataset / "sparse/0/points3D.ply")["vertex"].data.dtype.names)
        if not {"nx", "ny", "nz"}.issubset(properties):
            raise ValueError("Prepare compatible PLY in a new input directory; do not mutate the baseline")
        native_parser = argparse.ArgumentParser()
        groups = [ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)]
        native = native_parser.parse_args([])
        frozen = json.loads((args.runtime / "MAIN_CONFIG.json").read_text())["parameters"]
        for key, value in frozen.items():
            if not hasattr(native, key):
                raise ValueError(f"Unsupported frozen parameter: {key}")
            setattr(native, key, value)
        native.source_path = str(args.dataset.resolve())
        native.model_path = str((args.output / "model").resolve())
        native.iterations = args.iterations
        if args.optimizer_type:
            native.optimizer_type = args.optimizer_type
        if args.iterations < 1000:
            native.densify_until_iter = 0
        if args.gpu_resident_inputs:
            if native.data_device != "cpu" or native.train_test_exp or native.depths:
                raise ValueError("Native residency control must use plain CPU RGB inputs")
            native.data_device = "cuda"
        model, optimization, pipeline = [group.extract(native) for group in groups]
        if args.compact_host_images and (native.data_device != "cpu" or native.train_test_exp or native.depths):
            raise ValueError("Host image storage currently supports CPU RGB inputs without depth or exposure masks")
        if args.gsplat_build:
            if (args.projection != "full" or args.rasterizer_library or native.optimizer_type != "default"
                    or native.depth_l1_weight_init != 0 or native.depth_l1_weight_final != 0 or native.train_test_exp):
                raise ValueError("First gsplat comparison requires full K, dense Adam, and disabled depth/exposure supervision")
            from gsplat_adapter import activate_gsplat, render as gsplat_render
            optional_gsplat = activate_gsplat(args.gsplat_build)
            reference.render = gsplat_render
        image_paths = sorted(p for p in (args.dataset/"images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        if args.expected_image_count <= 0 or len(image_paths) != args.expected_image_count:
            raise ValueError("Actual images differ from the explicitly declared input denominator")
        if args.input_version == "C212_frozen" and args.expected_image_count != 212:
            raise ValueError("A new frame set requires its own explicit input version")
        input_identity = [data_identity(args.dataset/"sparse/0"/name)
                          for name in ["cameras.bin", "images.bin", "points3D.bin", "points3D.ply"]]
        input_identity.extend(data_identity(p) for p in image_paths)
        atomic_json(args.output/"INPUT_MANIFEST.json", {"dataset": str(args.dataset), "files": input_identity,
                    "image_count": len(image_paths), "role": "frozen_development_training",
                    "input_version": args.input_version, "expected_image_count": args.expected_image_count})
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        budget = min(12*1024**3, free-2*1024**3)
        if budget <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.ssim_backend == "fused" and not reference.FUSED_SSIM_AVAILABLE:
            raise RuntimeError("Requested fused SSIM did not import; refuse a silent reference fallback")
        if args.optimizer_type == "sparse_adam" and not reference.SPARSE_ADAM_AVAILABLE:
            raise RuntimeError("Sparse Adam requires the explicit compatible accelerated rasterizer")
        if args.dense_adam_backend == "fused":
            from optimizer_adapter import fused_adam_model
            reference.GaussianModel = fused_adam_model(reference.GaussianModel)
        if args.vectorized_ply:
            from ply_serialization import vectorized_ply_model
            reference.GaussianModel = vectorized_ply_model(reference.GaussianModel, args.verify_native_ply)
        optional_density = None
        if args.target_density_roi:
            from target_density_adapter import target_densification_model
            roi = json.loads(args.target_density_roi.read_text())
            reference.GaussianModel = target_densification_model(reference.GaussianModel,
                roi["roi_minimum"], roi["roi_maximum"], 2.)
            optional_density = {"source": data_identity(args.target_density_roi),
                "minimum": roi["roi_minimum"], "maximum": roi["roi_maximum"], "multiplier": 2.}
        reference.FUSED_SSIM_AVAILABLE = args.ssim_backend == "fused"
        atomic_json(args.output/"CONFIG.json", {"runtime": metadata, "parameters": vars(native),
            "implementation": implementation,
            "intervention": {"projection": args.projection, "ssim_backend": args.ssim_backend,
                "custom_rasterizer": args.rasterizer_library is not None, "optimizer_type": native.optimizer_type,
                "dense_adam_backend": args.dense_adam_backend,
                "pyramid_until": args.pyramid_until,
                "compact_image_cache": args.compact_image_cache,
                "resident_fused_adam": args.resident_fused_adam}, "seed": args.seed,
            "ply_serialization": {"vectorized": args.vectorized_ply, "native_byte_oracle": args.verify_native_ply},
            "compact_host_images": args.compact_host_images,
            "trim_host_after_load": args.trim_host_after_load,
            "gpu_resident_inputs": args.gpu_resident_inputs,
            "target_density": optional_density,
            "optional_ssim_identity": optional_ssim,
            "optional_gsplat_identity": optional_gsplat,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "gpu_budget_bytes": budget, "ram_budget_bytes": 32*1024**3,
            "maximum_wall_seconds": args.maximum_seconds,
            "fixed_fused_ssim": args.ssim_backend == "fused", "camera_optimization": False,
            "input_version": args.input_version, "expected_image_count": args.expected_image_count,
            "strict_blind_evaluation": False,
            "initialization_seconds": time.monotonic()-started})
        scene_timing = {}
        original_scene = reference.Scene
        pyramid = None
        image_cache = None
        density_model = None
        serialization_model = None
        staged_render = None

        class TimedScene(original_scene):
            def __init__(self, *scene_args, **kwargs):
                nonlocal pyramid, staged_render, image_cache, density_model, serialization_model
                before = time.monotonic()
                super().__init__(*scene_args, **kwargs)
                if args.trim_host_after_load:
                    from host_memory import trim_unused_host_pages
                    scene_timing["host_memory_trim"] = trim_unused_host_pages()
                if args.gpu_resident_inputs:
                    import psutil
                    payload = {"cpu": 0, "cuda": 0}
                    for camera in self.getTrainCameras():
                        for value in (camera.original_image, camera.alpha_mask):
                            if value is not None:
                                if value.device.type != "cuda":
                                    raise ValueError("Native CUDA residency silently used another input device")
                                payload[value.device.type] += value.numel() * value.element_size()
                    if torch.cuda.max_memory_reserved() > budget:
                        raise MemoryError("Native image residency exceeds the registered CUDA budget")
                    scene_timing["input_residency"] = {"payload_bytes_by_device": payload,
                        "rss_after_load_bytes": psutil.Process().memory_info().rss,
                        "cuda_reserved_after_load_bytes": torch.cuda.memory_reserved(),
                        "camera_count": len(self.getTrainCameras())}
                if args.vectorized_ply:
                    serialization_model = self.gaussians
                if args.target_density_roi:
                    density_model = self.gaussians
                    initial_roi = density_model.target_membership()
                    initial_fraction = float(initial_roi.float().mean().item())
                    if not 0 < initial_fraction < .5:
                        raise ValueError("The target ROI fails the frozen selective-volume admission")
                    scene_timing["initial_target_fraction"] = initial_fraction
                if args.compact_image_cache:
                    from resident_image_cache import ResidentImageCache
                    image_cache = ResidentImageCache(self.getTrainCameras(), 4*1024**3, budget)
                if args.pyramid_until:
                    from resolution_adapter import ProgressiveResolution
                    pyramid = ProgressiveResolution(self.getTrainCameras(), args.pyramid_until)
                    staged_render = pyramid.wrap_renderer(render_original)
                torch.cuda.synchronize()
                scene_timing["load_and_gaussian_initialization_seconds"] = time.monotonic()-before

        reference.Scene = TimedScene
        render_original = reference.render
        host_store = None
        host_context = nullcontext()
        if args.compact_host_images:
            from host_image_storage import HostImageStorage, host_image_loader
            host_store = HostImageStorage(4*1024**3)
            host_context = host_image_loader(host_store)
        sample_counts = Counter()
        sample_sequence = hashlib.sha256()

        def counted_render(camera, *render_args, **kwargs):
            if host_store:
                host_store.activate(camera)
            if image_cache:
                image_cache.activate(camera)
            sample_counts[camera.image_name] += 1
            sample_sequence.update((camera.image_name+"\n").encode("utf-8"))
            return (staged_render or render_original)(camera, *render_args, **kwargs)

        reference.render = counted_render
        original_report = reference.training_report
        iteration_gpu_ms = []
        training_started = time.monotonic()

        def monitor(writer, iteration, l1, loss, loss_function, elapsed,
                    testing_iterations, scene, render_fn, render_args, train_test_exp):
            original_report(writer, iteration, l1, loss, loss_function, elapsed,
                            testing_iterations, scene, render_fn, render_args, train_test_exp)
            if image_cache:
                image_cache.release()
            if host_store:
                host_store.release()
            if pyramid:
                pyramid.after_iteration(iteration)
            iteration_gpu_ms.append(float(elapsed))
            if time.monotonic()-started > args.maximum_seconds:
                raise TimeoutError("The registered wall-time cap was reached")
            if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training loss")
                reserved = torch.cuda.max_memory_reserved()
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                if reserved > budget or rss > 32*1024**3:
                    raise MemoryError("The registered resource budget was exceeded")
                state.update(status="running", iteration=iteration, loss=float(loss.item()),
                    l1=float(l1.item()), gaussian_count=int(scene.gaussians.get_xyz.shape[0]),
                    maximum_cuda_reserved_bytes=int(reserved), maximum_rss_bytes=int(rss),
                    elapsed_seconds=time.monotonic()-started,
                    training_call_seconds=time.monotonic()-training_started)
                atomic_json(state_path, state)
                with (args.output/"progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(state)+"\n")

        reference.training_report = monitor
        saving = sorted(set([i for i in [2000, 7000, args.iterations] if i <= args.iterations]))
        recovery = None
        recovery_context = nullcontext()
        if args.checkpoint_every:
            from checkpoint_adapter import RecoveryState, recoverable_training
            parameters = {key: value for key, value in vars(native).items() if key != "model_path"}
            recovery_identity = {"parameters": parameters, "seed": args.seed,
                "input_version": args.input_version, "expected_image_count": args.expected_image_count,
                "input_sha256": [row["sha256"] for row in input_identity],
                "source_sha256": [row["sha256"] for row in implementation],
                "reference_commit": metadata["reference_commit"], "torch": torch.__version__,
                "rasterizer_python_sha256": metadata["rasterizer_python"]["sha256"],
                "rasterizer_extension_sha256": metadata["rasterizer_extension"]["sha256"],
                "dense_adam_backend": args.dense_adam_backend,
                "gsplat_identity": optional_gsplat,
                "gpu_resident_inputs": args.gpu_resident_inputs,
                "trim_host_after_load": args.trim_host_after_load,
                "ply_serialization": {"vectorized": args.vectorized_ply, "native_byte_oracle": args.verify_native_ply},
                "projection": args.projection, "ssim_backend": args.ssim_backend, "optional_ssim": optional_ssim}
            recovery = RecoveryState(args.output, recovery_identity, args.checkpoint_every,
                                     args.resume_file, args.simulate_crash_after)
            recovery_context = recoverable_training(reference, recovery)
        with camera_loader_adapter(args.dataset, enabled=args.projection == "full") as cameras, host_context, recovery_context:
            reference.training(model, optimization, pipeline, [], saving, [], None, -1)
            atomic_json(args.output/"CAMERA_MANIFEST.json", {"pixel_convention": "colmap_corner_first_center_0.5", "cameras": cameras})
        torch.cuda.synchronize()
        resolution_receipt = pyramid.finish(args.iterations) if pyramid else None
        cache_receipt = image_cache.finish() if image_cache else None
        host_receipt = host_store.receipt() if host_store else None
        if host_receipt:
            if (host_receipt["samples"] != args.iterations or host_receipt["camera_count"] != len(image_paths)
                    or not host_receipt["only_active_camera_has_float_inputs"] or host_receipt["active_camera"] is not None):
                raise ValueError("The host storage lifecycle or sample denominator differs from the frozen training")
            atomic_json(args.output/"HOST_IMAGE_STORAGE.json", host_receipt)
        density_receipt = density_model.target_density_receipt() if density_model else None
        serialization_receipt = getattr(serialization_model, "ply_serialization_events", None)
        if args.vectorized_ply:
            from ply_serialization import validate_save_schedule
            save_schedule_check = validate_save_schedule(serialization_receipt, saving,
                recovery.resumed_iteration if recovery else 0)
            atomic_json(args.output/"PLY_SERIALIZATION.json", {"events": serialization_receipt,
                        "verify_native": args.verify_native_ply, "schedule_validation": save_schedule_check,
                        "optimization_count": 0})
        if density_receipt:
            if density_receipt["extra_candidates_outside_sum"] != 0:
                raise ValueError("Target density changed background eligibility")
            atomic_json(args.output/"TARGET_DENSITY.json", density_receipt)
        if cache_receipt:
            if cache_receipt["samples"] != args.iterations:
                raise ValueError("Cached camera sample denominator differs from the training schedule")
            atomic_json(args.output/"RESIDENT_CACHE.json", cache_receipt)
        if resolution_receipt:
            atomic_json(args.output/"RESOLUTION_SCHEDULE.json", resolution_receipt)
        ply = args.output / f"model/point_cloud/iteration_{args.iterations}/point_cloud.ply"
        data = PlyData.read(ply)["vertex"].data
        if len(data) == 0 or not all(np.isfinite(data[key]).all() for key in data.dtype.names):
            raise FloatingPointError("Invalid final Gaussian parameters")
        resumed_iteration = recovery.resumed_iteration if recovery else 0
        if len(cameras) != len(image_paths) or sum(sample_counts.values()) != args.iterations-resumed_iteration:
            raise ValueError("Camera loading or training-sampling denominator mismatch")
        if recovery:
            atomic_json(args.output/"SAMPLING_TRACE.json", {"names": recovery.sample_names,
                "total_count": len(recovery.sample_names), "resumed_iteration": resumed_iteration})
        state.update(status="completed", elapsed_seconds=time.monotonic()-started,
            training_call_seconds=time.monotonic()-training_started,
            final_model=str(ply), gaussian_count=len(data), image_count=len(image_paths),
            sample_counts=dict(sample_counts), mean_samples=args.iterations/len(image_paths),
            sample_sequence_sha256=sample_sequence.hexdigest(), resolution_schedule=resolution_receipt,
            scene_timing=scene_timing, resident_image_cache=cache_receipt,
            host_image_storage=host_receipt,
            ply_serialization=serialization_receipt,
            target_density_summary=({key: value for key, value in density_receipt.items() if key != "events"}
                                    if density_receipt else None),
            iteration_gpu_ms_p50_p95=np.quantile(iteration_gpu_ms, [.5, .95]).tolist(),
            full_resume_checkpoint=bool(recovery), resumed_iteration=resumed_iteration,
            executed_iterations=args.iterations-resumed_iteration,
            counted_as_usable_optimization=False)
        atomic_json(state_path, state)
        print(json.dumps({key:state[key] for key in ["status", "projection", "iteration", "gaussian_count", "training_call_seconds"]}))
    except Exception:
        state.update(status="failed", elapsed_seconds=time.monotonic()-started, error=traceback.format_exc())
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
