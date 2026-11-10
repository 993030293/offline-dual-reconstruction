"""Measure full native Scene loading and fingerprint every camera and RNG state."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import pickle
import random
import resource
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import camera_loader_adapter
from prefetch_camera_images import prefetch_camera_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "prefetch"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import ModelParams
    from scene import Scene, GaussianModel
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    free, _ = torch.cuda.mem_get_info()
    gpu_budget = min(12*1024**3, free-2*1024**3)
    native_parser = argparse.ArgumentParser()
    group = ModelParams(native_parser)
    native = native_parser.parse_args([])
    for key, value in json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(native, key, value)
    native.source_path = str(args.dataset.resolve())
    native.model_path = str((args.output/"scene").resolve())
    Path(native.model_path).mkdir()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    def tensor(value):
        if value is None:
            return None
        return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                "device": str(value.device), "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    def rng():
        return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                "torch_CPU": tensor(torch.get_rng_state()), "torch_CUDA": tensor(torch.cuda.get_rng_state())}
    before = rng()
    atomic_json(args.output/"PROTOCOL.json", {"mode": args.mode, "runtime": metadata,
        "dataset": str(args.dataset), "expected_images": 1016, "workers": 4, "capacity": 8,
        "maximum_seconds": 300, "maximum_RSS_bytes": 32*1024**3, "GPU_budget_bytes": gpu_budget,
        "adapter_source": data_identity(Path(__file__).with_name("prefetch_camera_images.py")),
        "validator_source": data_identity(Path(__file__)), "before_RNG": before})
    started = time.perf_counter()
    with camera_loader_adapter(args.dataset) as manifest, (prefetch_camera_images() if args.mode == "prefetch" else nullcontext()) as prefetch:
        scene = Scene(group.extract(native), GaussianModel(3), shuffle=True)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-started
        after = rng()
        memory = psutil.Process().memory_info().rss
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        cameras = scene.getTrainCameras()
        if len(cameras) != 1016 or len(manifest) != 1016:
            raise ValueError("The native loader did not preserve every admitted input")
        rows = []
        for camera in cameras:
            rows.append({"name": camera.image_name, "uid": camera.uid, "colmap_id": camera.colmap_id,
                "intrinsics": camera.calibrated_intrinsics,
                "image": tensor(camera.original_image), "alpha": tensor(camera.alpha_mask),
                "world_view": tensor(camera.world_view_transform), "projection": tensor(camera.projection_matrix),
                "full_projection": tensor(camera.full_proj_transform), "center": tensor(camera.camera_center)})
        initial = {name: tensor(getattr(scene.gaussians, name)) for name in
                   ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_exposure")}
    checks = {"all1016_cameras": len(rows) == 1016,
        "CPU_image_storage_preserved": all(r["image"]["device"] == "cpu" and r["alpha"]["device"] == "cpu" for r in rows),
        "RAM_budget": peak <= 32*1024**3, "CUDA_budget": torch.cuda.max_memory_reserved() <= gpu_budget,
        "prefetch_consumed_all_within_bound": prefetch is None or
            (prefetch["prepared"] == prefetch["consumed"] == 1016 and prefetch["maximum_pending"] <= 8)}
    result = {"status": "loaded_full_scene_pending_pair_comparison" if all(checks.values()) else "failed",
        "checks": checks, "mode": args.mode, "Scene_load_seconds": load_seconds,
        "RSS_after_load_bytes": memory, "peak_RSS_after_load_bytes": peak,
        "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "before_RNG": before, "after_RNG": after,
        "rows": rows, "initial_Gaussians": initial, "camera_manifest": manifest,
        "camera_extent": scene.cameras_extent, "prefetch": prefetch, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("status", "Scene_load_seconds", "RSS_after_load_bytes", "peak_CUDA_reserved_bytes", "checks")}))
    if not all(checks.values()):
        raise ValueError("A camera prefetch resource guard failed")


if __name__ == "__main__":
    main()
