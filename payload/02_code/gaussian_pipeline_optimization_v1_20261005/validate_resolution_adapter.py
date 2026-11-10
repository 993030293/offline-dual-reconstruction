"""Verify real calibrated cameras, tensor restoration, and density pixel units."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np

from full_intrinsics import Pinhole, camera_loader_adapter
from gsplat_adapter import activate_gsplat, render
from resolution_adapter import ProgressiveResolution
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "build", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    contract = {"status": "registered", "maximum_seconds": 300, "ram_budget_bytes": 32*1024**3,
        "dataset": str(args.dataset), "actual_cameras": 212, "renders": 4,
        "projection_tolerance_pixels": .001, "counted": False,
        "sources": [data_identity(Path(__file__).parent/name) for name in
                    ("validate_resolution_adapter.py", "resolution_adapter.py", "full_intrinsics.py", "gsplat_adapter.py")]}
    atomic_json(args.output/"CONTRACT.json", contract)
    try:
        reference = activate_reference(args.runtime)
        backend = activate_gsplat(args.build)
        import torch
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from scene import Scene, GaussianModel

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        native_parser = argparse.ArgumentParser()
        groups = [ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)]
        native = native_parser.parse_args([])
        for key, value in json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"].items():
            if not hasattr(native, key):
                raise ValueError("Unsupported frozen parameter")
            setattr(native, key, value)
        native.source_path = str(args.dataset)
        native.model_path = str(args.output/"scene")
        Path(native.model_path).mkdir()
        model, optimization, pipeline = [group.extract(native) for group in groups]
        gaussian = GaussianModel(model.sh_degree)
        with camera_loader_adapter(args.dataset) as manifest:
            scene = Scene(model, gaussian)
        cameras = scene.getTrainCameras()
        gaussian.training_setup(optimization)
        if len(cameras) != 212 or len(manifest) != 212 or scene.getTestCameras():
            raise ValueError("The actual frozen training denominator changed")
        full = [{"intrinsic": dict(cam.calibrated_intrinsics), "image": cam.original_image,
                 "mask": cam.alpha_mask, "world": cam.world_view_transform.clone()} for cam in cameras]
        original_hash = hashlib.sha256(full[0]["image"].numpy().tobytes()).hexdigest()
        small = torch.arange(3*24*26, dtype=torch.float32).reshape(3, 24, 26)
        expected_small = np.zeros((3, 12, 13), dtype=np.float32)
        for y in range(12):
            for x in range(13):
                expected_small[:, y, x] = small.numpy()[:, 2*y:2*y+2, 2*x:2*x+2].mean(axis=(1, 2))
        checks = {"independent_small_area_oracle": np.array_equal(
            torch.nn.functional.avg_pool2d(small, 2, 2).numpy(), expected_small)}
        staging = ProgressiveResolution(cameras, 2)
        original_image = full[0]["image"].numpy()
        expected_real = (original_image[:, 0::2, 0::2]+original_image[:, 0::2, 1::2]
                         +original_image[:, 1::2, 0::2]+original_image[:, 1::2, 1::2])/4
        checks["actual_image_area_oracle"] = bool(np.max(np.abs(expected_real-cameras[0].original_image.numpy())) <= 1e-7)
        points = np.array([[-.3, -.2, 2., 1.], [.2, .1, 3., 1.], [.7, -.6, 4., 1.]], dtype=np.float64)
        errors = []
        for camera, old in zip(cameras, full):
            intrinsic = Pinhole(**old["intrinsic"])
            scaled = Pinhole(**camera.calibrated_intrinsics)
            analytic_full = (intrinsic.matrix() @ points[:, :3].T).T
            analytic_full = analytic_full[:, :2]/analytic_full[:, 2:3]
            clip = (camera.projection_matrix.detach().cpu().numpy().T.astype(np.float64) @ points.T).T
            actual = (clip[:, :2]/clip[:, 3:4]+1)*np.array([scaled.width, scaled.height])/2
            errors.append(float(np.max(np.abs(actual-analytic_full/2))))
            if (asdict(scaled) != asdict(intrinsic.resized(intrinsic.width//2, intrinsic.height//2))
                    or camera.original_image.shape != (3, scaled.height, scaled.width)
                    or camera.alpha_mask.shape != (1, scaled.height, scaled.width)
                    or not torch.equal(camera.world_view_transform, old["world"])):
                raise ValueError("A real camera, mask, or image did not obey the half-resolution contract")
        checks["all_212_coarse_projection_and_pose"] = max(errors) <= .001
        staged = staging.wrap_renderer(render)
        background = torch.zeros(3, device="cuda")
        render_rows = []
        for iteration in range(1, 5):
            camera = cameras[iteration-1]
            phase = staging.phase
            with torch.no_grad():
                direct = render(camera, gaussian, pipeline, background)
            output = staged(camera, gaussian, pipeline, background)
            factor = 2 if phase == "coarse" else 1
            if not torch.equal(output["radii"], direct["radii"].float()*factor):
                raise ValueError("Canonical density radii differ from independently rendered native pixel radii")
            if not torch.equal(output["visibility_filter"], direct["visibility_filter"]):
                raise ValueError("Radius unit conversion changed actual visibility")
            loss = (output["render"]-camera.original_image.cuda()).square().mean()
            loss.backward()
            gradient = output["viewspace_points"].grad
            if gradient is None or not torch.isfinite(gradient).all() or not (gradient != 0).any():
                raise ValueError("The actual gsplat density gradient is missing, zero, or nonfinite")
            if gaussian.get_xyz.grad is None or not torch.isfinite(gaussian.get_xyz.grad).all():
                raise ValueError("Nonfinite world-space gradient")
            render_rows.append({"iteration": iteration, "phase": phase, "shape": list(output["render"].shape),
                "loss": float(loss), "maximum_canonical_radius": float(output["radii"].max()),
                "visible": int(output["visibility_filter"].numel()), "finite_nonzero_ndc_gradient": True})
            gaussian.optimizer.zero_grad(set_to_none=True)
            staging.after_iteration(iteration)
        receipt = staging.finish(4)
        checks["four_real_forward_backward_radii_visibility"] = len(render_rows) == 4
        checks["all_original_tensors_and_calibration_restored"] = receipt["original_tensor_references_and_calibration_restored"]
        checks["original_image_bytes_unchanged"] = original_hash == hashlib.sha256(cameras[0].original_image.numpy().tobytes()).hexdigest()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        checks["bounded_resources"] = rss <= contract["ram_budget_bytes"] and time.monotonic()-started < 300
        result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "projection_max_error_pixels": max(errors), "renders": render_rows, "schedule": receipt,
            "maximum_rss_bytes": rss, "elapsed_seconds": time.monotonic()-started,
            "reference": reference, "backend": backend, "source_sha256": {row["path"].split("/")[-1]: row["sha256"] for row in contract["sources"]},
            "counted": False, "numerical_equivalence_to_full_resolution_claim": False}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "projection_max_error_pixels", "elapsed_seconds")}), flush=True)
        if result["status"] != "passed":
            raise RuntimeError("Resolution prerequisite failed")
    except Exception as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed", "error": str(error), "elapsed_seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
