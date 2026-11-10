"""Validate exact cache reconstruction and actual fixed training camera bindings."""

import argparse
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import torch

from full_intrinsics import camera_loader_adapter
from resident_image_cache import ResidentImageCache
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    checks = {}
    atomic_json(args.output/"CONTRACT.json", {"maximum_seconds": 300, "ram_budget_bytes": 32*1024**3,
        "gpu_budget_bytes": budget, "cache_budget_bytes": 4*1024**3,
        "dataset": str(args.dataset), "source": data_identity(Path(__file__)), "training": False})

    def camera(name="synthetic", rgb=None, alpha=None):
        if rgb is None:
            rgb = (torch.arange(256).reshape(1, 16, 16).float()/255).repeat(3, 1, 1).transpose(1, 2)
        return SimpleNamespace(image_name=name, original_image=rgb, alpha_mask=alpha)

    def rejects(name, values, capacity=4*1024**3, limit=budget):
        try:
            ResidentImageCache(values, capacity, limit)
        except (ValueError, MemoryError):
            checks[name] = True
        else:
            checks[name] = False

    alpha = (torch.arange(256).reshape(1, 16, 16).float()/255).transpose(1, 2)
    samples = [camera(), camera("alpha", alpha=alpha), camera("constant", alpha=torch.full((1, 16, 16), 0.123))]
    cache = ResidentImageCache(samples, 4*1024**3, budget)
    for item in samples:
        before_rgb, before_alpha = item.original_image, item.alpha_mask
        cache.activate(item)
        checks[item.image_name+"_rgb_exact"] = torch.equal(item.original_image.cpu().view(torch.int32), before_rgb.contiguous().view(torch.int32))
        checks[item.image_name+"_alpha_exact"] = (item.alpha_mask is None if before_alpha is None else
            torch.equal(item.alpha_mask.cpu().contiguous().view(torch.int32), before_alpha.contiguous().view(torch.int32)))
        if item is samples[0]:
            try:
                cache.activate(samples[1])
            except RuntimeError:
                checks["overlap_rejected"] = True
            else:
                checks["overlap_rejected"] = False
        cache.release()
        checks[item.image_name+"_restored"] = item.original_image is before_rgb and item.alpha_mask is before_alpha
    try:
        cache.activate(camera("unknown"))
    except ValueError:
        checks["unknown_camera_rejected"] = True
    synthetic_receipt = cache.finish()
    rejects("empty_rejected", [])
    rejects("duplicate_object_rejected", [samples[0], samples[0]])
    rejects("duplicate_name_rejected", [camera(), camera()])
    rejects("capacity_rejected", samples, 1025)
    rejects("headroom_rejected", samples, limit=4*1024**3)
    rejects("gradient_input_rejected", [camera(rgb=samples[0].original_image.clone().requires_grad_())])
    rejects("wrong_dtype_rejected", [camera(rgb=samples[0].original_image.double())])
    rejects("cuda_input_rejected", [camera(rgb=samples[0].original_image.cuda())])
    rejects("nonfinite_rejected", [camera(rgb=torch.full((3, 16, 16), float("nan")))])
    rejects("range_rejected", [camera(rgb=torch.full((3, 16, 16), 1.1))])
    rejects("lossy_rgb_rejected", [camera(rgb=torch.full((3, 16, 16), 0.12345))])
    rejects("bad_alpha_shape_rejected", [camera(alpha=torch.ones(16, 16))])
    del cache
    metadata = activate_reference(args.runtime)
    from arguments import ModelParams
    from scene import Scene, GaussianModel
    native_parser = argparse.ArgumentParser()
    group = ModelParams(native_parser)
    native = native_parser.parse_args([])
    for key, value in json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(native, key, value)
    native.source_path = str(args.dataset)
    native.model_path = str(args.output/"temporary_scene")
    Path(native.model_path).mkdir()
    with camera_loader_adapter(args.dataset) as manifest:
        scene = Scene(group.extract(native), GaussianModel(3), shuffle=False)
    actual = scene.getTrainCameras()
    reference_geometry = [(item.world_view_transform.clone(), item.full_proj_transform.clone(),
                           dict(item.calibrated_intrinsics)) for item in actual]
    cache = ResidentImageCache(actual, 4*1024**3, budget)
    all_rgb_exact, all_alpha_exact, all_no_H2D, all_geometry = True, True, True, True
    for item, matrices in zip(actual, reference_geometry):
        before_rgb, before_alpha = item.original_image, item.alpha_mask
        cache.activate(item)
        all_rgb_exact &= torch.equal(item.original_image.cpu().view(torch.int32), before_rgb.contiguous().view(torch.int32))
        all_alpha_exact &= torch.equal(item.alpha_mask.cpu().contiguous().view(torch.int32), before_alpha.contiguous().view(torch.int32))
        all_no_H2D &= item.original_image.cuda().data_ptr() == item.original_image.data_ptr()
        all_no_H2D &= item.alpha_mask.cuda().data_ptr() == item.alpha_mask.data_ptr()
        all_geometry &= torch.equal(item.world_view_transform, matrices[0]) and torch.equal(item.full_proj_transform, matrices[1])
        all_geometry &= item.calibrated_intrinsics == matrices[2]
        cache.release()
    receipt = cache.finish()
    checks.update(actual212_loaded=len(actual) == 212, actual_all_RGB_bit_exact=all_rgb_exact,
        actual_all_alpha_bit_exact=all_alpha_exact, actual_cuda_call_no_transfer=all_no_H2D,
        actual_all_geometry_unchanged=all_geometry, actual_references_restored=receipt["original_references_restored"],
        finite_resource_bound=time.monotonic()-started < 300 and torch.cuda.max_memory_reserved() < budget
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 32*1024**3)
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "runtime": metadata, "actual_cache": receipt, "synthetic_cache": synthetic_receipt,
        "actual_camera_manifest": manifest, "elapsed_seconds": time.monotonic()-started,
        "maximum_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "sources": [data_identity(Path(__file__).parent/name) for name in
                    ("resident_image_cache.py", "full_intrinsics.py", "validate_resident_cache.py")],
        "counted": False, "full_training_run": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "actual_cache", "elapsed_seconds")}))
    if result["status"] != "passed":
        raise RuntimeError("Resident cache validation failed")


if __name__ == "__main__":
    main()
