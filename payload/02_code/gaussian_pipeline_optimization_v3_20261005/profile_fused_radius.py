"""Check all real radii and gradients before a cold-cost radius kernel comparison."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "build", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--trial", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(args.output/"triton_cache")
    reference = activate_reference(args.runtime)
    import torch
    import triton
    import gsplat_adapter
    from fused_projection_radius import native_radii, fused_radii, fused_radius_adapter, admission
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from arguments import PipelineParams
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    build = gsplat_adapter.activate_gsplat(args.build)
    paths = [Path(__file__), Path(__file__).with_name("fused_projection_radius.py"), Path(gsplat_adapter.__file__), args.model, args.localization]
    identity = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identity, "reference": reference, "build": build,
        "trial": args.trial, "all24_real_views_and_all_native_radii": True, "original_RNG_sampling_and_dense_layout": True,
        "radii_bit_exact": True, "pixel_atol": 1e-5, "loss_atol": 1e-7, "gradient_atol": 2e-6, "gradient_rtol": 1e-4,
        "radius_stage_call_budget": 12000, "first_cold_compile_and_prepare_in_candidate_stage": True,
        "fresh_owned_triton_cache": os.environ["TRITON_CACHE_DIR"], "triton_version": triton.__version__,
        "maximum_cuda_reserved_bytes": 12*1024**3, "minimum_stage_reduction": .1,
        "full_training_and_recovery_required_before_acceptance": True, "counted": False})
    try:
        localization = json.loads(args.localization.read_text())
        if len(localization["rows"]) != 24 or any(r["status"] != "localized" for r in localization["rows"]):
            raise ValueError("The complete original24-view denominator changed")
        meta = localization["PINHOLE_camera"]
        native = Pinhole(meta["width"], meta["height"], *meta["params"])
        scaled = native.resized(round(native.width/2), round(native.height/2))
        cameras = []
        for row in localization["rows"]:
            with Image.open(mounted_path(row["image_reference"])) as photo:
                camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
                    2*math.atan(native.width/(2*native.fx)), 2*math.atan(native.height/(2*native.fy)),
                    None, photo.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
            apply_to_camera(camera, scaled)
            cameras.append(camera)
        model = GaussianModel(3)
        with vectorized_gaussian_read(True):
            model.load_ply(str(args.model))
        native_parser = argparse.ArgumentParser()
        group = PipelineParams(native_parser)
        pipeline = group.extract(native_parser.parse_args([]))
        background = torch.zeros(3, device="cuda")
        parameters = [getattr(model, n) for n in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")]
        inputs, native_formula_checks = [], []
        original_rasterize = gsplat_adapter.rasterize_values

        def capture(*values, **kwargs):
            image, radii, info = original_rasterize(*values, **kwargs)
            conics, original = info["conics"][0].detach().clone(), info["radii"][0].detach().clone()
            native_formula_checks.append(bool(torch.equal(radii, native_radii(conics, original))))
            inputs.append((conics, original, radii.detach().clone()))
            return image, radii, info

        gsplat_adapter.rasterize_values = capture
        try:
            with torch.no_grad():
                for camera in cameras:
                    gsplat_adapter.render(camera, model, pipeline, background)
        finally:
            gsplat_adapter.rasterize_values = original_rasterize
        torch.cuda.synchronize()
        cold = time.perf_counter()
        first = fused_radii(inputs[0][0], inputs[0][1])
        torch.cuda.synchronize()
        cold = time.perf_counter()-cold
        real_checks = [bool(torch.equal(first, inputs[0][2]))]
        for conics, original, expected in inputs[1:]:
            real_checks.append(bool(torch.equal(fused_radii(conics, original), expected)))
        atomic_json(args.output/"REAL_RADII.json", {"all24_native_formula_exact": all(native_formula_checks),
            "all24_fused_radii_exact": all(real_checks), "cold_compile_seconds": cold,
            "counts": [len(a) for a, _, _ in inputs]})
        if not all(native_formula_checks) or not all(real_checks):
            raise ValueError("An actual calibrated-view integer radius changed")
        synthetic = []
        generator = np.random.default_rng(20261005)
        for scale in (.0001, 1., 10000.):
            matrices = generator.normal(size=(4096, 2, 2))
            values = matrices@matrices.transpose(0, 2, 1)+np.eye(2)*.01
            conics = torch.tensor(np.column_stack((values[:, 0, 0], values[:, 0, 1], values[:, 1, 1]))*scale, dtype=torch.float32, device="cuda")
            original = torch.tensor(generator.integers(-1, 10, (len(conics), 2)), dtype=torch.int32, device="cuda")
            synthetic.append(bool(torch.equal(native_radii(conics, original), fused_radii(conics, original))))
        diagonal = []
        for k in range(2, 500):
            value = np.float32(1/((k/3)**2-np.sqrt(.1)))
            diagonal += [np.nextafter(value, np.float32(0)), value, np.nextafter(value, np.float32(np.inf))]
        conics = torch.tensor([[v, 0., v] for v in diagonal]+[[float("nan"), 0., 1.], [float("inf"), 0., 1.],
            [1., 2., 1.], [0., 0., 0.], [-1., 0., -1.], [1e-30, 0., 1e-30]], dtype=torch.float32, device="cuda")
        original = torch.ones((len(conics), 2), dtype=torch.int32, device="cuda")
        boundary_exact = bool(torch.equal(native_radii(conics, original), fused_radii(conics, original)))
        empty = torch.empty((0, 3), device="cuda")
        empty_r = torch.empty((0, 2), dtype=torch.int32, device="cuda")
        route_checks = {"CPU_preselects_original": not admission(conics.cpu(), original.cpu()),
            "float64_preselects_original": not admission(conics.double(), original),
            "empty_native_radius_exact": torch.equal(native_radii(empty, empty_r), fused_radii(empty, empty_r))}
        atomic_json(args.output/"BOUNDARIES.json", {"all_three_scales_exact": all(synthetic), "integer_and_bad_value_boundary_exact": boundary_exact,
            "routes": route_checks})
        if not all(synthetic) or not boundary_exact or not all(route_checks.values()):
            raise ValueError("An arbitrary-scale or integer-boundary radius contract failed")
        gradients = []
        for number, camera in enumerate(cameras):
            outputs = {}
            for mode in ("native", "fused"):
                for parameter in parameters:
                    parameter.grad = None
                with fused_radius_adapter(gsplat_adapter, mode == "fused"):
                    package = gsplat_adapter.render(camera, model, pipeline, background)
                    image = package["render"]
                    target = camera.original_image.cuda()
                    loss = .8*l1_loss(image, target)+.2*(1-ssim(image, target))
                    loss.backward()
                    outputs[mode] = {"pixels": image.detach().cpu(), "radii": package["radii"].detach().cpu(),
                        "loss": float(loss.detach()), "gradients": [p.grad.detach().cpu() for p in parameters],
                        "density_gradient": package["viewspace_points"].grad.detach().cpu()}
            a, b = outputs["native"], outputs["fused"]
            row = {"camera": camera.image_name, "integer_radii_exact": torch.equal(a["radii"], b["radii"]),
                "pixel_max_abs": float((a["pixels"]-b["pixels"]).abs().max()), "loss_max_abs": abs(a["loss"]-b["loss"]),
                "all_six_gradient_guards": all(torch.allclose(x, y, atol=2e-6, rtol=1e-4) for x, y in zip(a["gradients"], b["gradients"])),
                "screen_density_gradient_guard": torch.allclose(a["density_gradient"], b["density_gradient"], atol=2e-6, rtol=1e-4)}
            gradients.append(row)
            atomic_json(args.output/"GRADIENTS.json", gradients)
            if not row["integer_radii_exact"] or row["pixel_max_abs"] > 1e-5 or row["loss_max_abs"] > 1e-7 or not row["all_six_gradient_guards"] or not row["screen_density_gradient_guard"]:
                raise ValueError("An original full pixel, loss, model or density gradient changed")
        times = {}
        for mode in (("native", "fused") if args.trial % 2 == 0 else ("fused", "native")):
            method = native_radii if mode == "native" else fused_radii
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(12000):
                conics, original, expected = inputs[step%24]
                output = method(conics, original)
            torch.cuda.synchronize()
            times[mode] = time.perf_counter()-start+(cold if mode == "fused" else 0)
            if not torch.equal(output, expected):
                raise ValueError("The final complete stage output changed")
        checks = {"all24_real_radii_and_original_gradients": len(gradients) == 24,
            "all_generic_scale_boundary_and_preselection_checks": all(synthetic) and boundary_exact and all(route_checks.values()),
            "maximum_original_CUDA_budget": torch.cuda.max_memory_reserved() <= 12*1024**3,
            "sources_inputs_unchanged": identity == [data_identity(p) for p in paths]}
        result = {"status": "radius_stage_pair_complete_full_training_pending", "trial": args.trial, "checks": checks,
            "complete_stage12000_call_seconds": times, "cold_compile_seconds": cold,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "sources_inputs": identity,
            "whole_training_executed": False, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "complete_stage12000_call_seconds", "cold_compile_seconds")}, flush=True)
        if not all(checks.values()):
            raise ValueError("A complete radius stage identity or memory guard failed")
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "counted": False})
        raise


if __name__ == "__main__":
    main()
