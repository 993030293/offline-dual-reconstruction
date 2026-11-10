"""Audit active SH on complete actual views before any full training campaign."""

import argparse
import gc
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from active_sh_materialization import active_features, active_sh_adapter


def generic_oracle(torch):
    checks = []
    generator = torch.Generator(device="cuda").manual_seed(20261005)
    for size in (0, 1, 2, 7, 31, 37, 129, 257):
        for degree in range(4):
            for strided in (False, True):
                dc = torch.randn((size, 1, 3), device="cuda", generator=generator)
                rest = torch.randn((size, 15, 6 if strided else 3), device="cuda", generator=generator)
                if strided:
                    rest = rest[:, :, ::2]
                dc = dc.detach().requires_grad_(True)
                rest = rest.detach().requires_grad_(True)
                model = SimpleNamespace(_features_dc=dc, _features_rest=rest, active_sh_degree=degree)
                count = (degree + 1) ** 2
                baseline = torch.cat((dc, rest), dim=1)[:, :count, :]
                candidate = active_features(model)
                weight = torch.arange(candidate.numel(), device="cuda", dtype=candidate.dtype).reshape(candidate.shape)
                (baseline * weight).sum().backward()
                original = (dc.grad.clone(), rest.grad.clone())
                dc.grad = rest.grad = None
                (candidate * weight).sum().backward()
                checks.append({"count": size, "degree": degree, "strided": strided,
                    "values_exact": torch.equal(baseline, candidate),
                    "full_parameter_gradients_exact": torch.equal(original[0], dc.grad) and torch.equal(original[1], rest.grad),
                    "all_unused_rest_gradients_present": rest.grad is not None and rest.grad.shape == rest.shape})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "build", "early-run", "final-run", "localization", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = activate_reference(args.runtime)
    import torch
    import gsplat_adapter as adapter
    from arguments import PipelineParams
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    build = adapter.activate_gsplat(args.build)
    paths = [Path(__file__), Path(__file__).with_name("active_sh_materialization.py"),
        Path(adapter.__file__), V2 / "vectorized_gaussian_read.py", args.localization]
    for folder, iteration in ((args.early_run, 1100), (args.final_run, 12000)):
        paths.extend([folder / "STATE.json", folder / "CONFIG.json",
                      folder / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"])
    identities = [data_identity(path) for path in paths]
    protocol = {"candidate": "R106_old_Q66_unaccepted_family", "sources_inputs": identities,
        "reference": reference, "build": build, "three_pairs_each_actual_state": True,
        "early_1100_degree1_and_final12000_degree3": True,
        "generic_coefficients_degrees0to3_empty_single_large_and_strided": True,
        "all24_original_development_views": True, "pixel_atol": 1e-5, "loss_atol": 1e-7,
        "gradient_atol": 2e-6, "gradient_rtol": 1e-4, "minimum_complete_render_backward_reduction": .1,
        "maximum_each_diagnostic_stage_ratio": 1.1, "maximum_cuda_reserved_bytes": 12 * 1024**3,
        "maximum_WSL_RSS_bytes": 32 * 1024**3,
        "payload_bytes_do_not_prove_peak_memory_or_cost_reduction": True,
        "setup_close_synchronization_and_target_copy_inside_stage": True,
        "oracle_CPU_copies_and_checks_reported_separately": True,
        "full_Scene_optimizer_densification_output_recovery_and_three_seed12k_not_this_diagnostic": True,
        "GT_object_shape_ROI_future_poses_not_used": True, "counted": False}
    atomic_json(args.output / "PROTOCOL.json", protocol)
    state = {"status": "running", "candidate": protocol["candidate"], "results": [], "counted": False}
    began = time.perf_counter()
    try:
        generic = generic_oracle(torch)
        exact = all(row["values_exact"] and row["full_parameter_gradients_exact"] and row["all_unused_rest_gradients_present"] for row in generic)
        atomic_json(args.output / "GENERIC.json", {"rows": generic, "all64_cases_exact": exact, "counted": False})
        if not exact:
            raise ValueError("A heterogeneous native SH value or complete gradient changed")
        localization = json.loads(args.localization.read_text())
        if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
            raise ValueError("The complete development-view denominator changed")
        meta = localization["PINHOLE_camera"]
        native = Pinhole(meta["width"], meta["height"], *meta["params"])
        scaled = native.resized(round(native.width / 2), round(native.height / 2))
        cameras = []
        for row in localization["rows"]:
            with Image.open(mounted_path(row["image_reference"])) as photo:
                camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
                    2 * math.atan(native.width / (2 * native.fx)), 2 * math.atan(native.height / (2 * native.fy)),
                    None, photo.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
            cameras.append(apply_to_camera(camera, scaled))
        native_parser = argparse.ArgumentParser()
        group = PipelineParams(native_parser)
        pipeline = group.extract(native_parser.parse_args([]))
        background = torch.zeros(3, device="cuda")
        for folder, iteration, degree in ((args.early_run, 1100, 1), (args.final_run, 12000, 3)):
            saved = json.loads((folder / "STATE.json").read_text())
            config = json.loads((folder / "CONFIG.json").read_text())
            if saved["status"] != "completed" or saved["iteration"] != iteration or saved["image_count"] != 1016 or config["parameters"]["sh_degree"] != 3:
                raise ValueError("The declared actual native training state changed")
            model = GaussianModel(3)
            with vectorized_gaussian_read(True):
                model.load_ply(str(folder / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"))
            model.active_sh_degree = degree
            parameters = [getattr(model, name) for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")]
            rows = []
            for trial in range(3):
                totals = {"native": 0., "active": 0.}
                peaks = {"native": 0, "active": 0}
                guards, traces = [], {"native": [], "active": []}
                verification = 0.
                original_render = adapter.render
                setup_start = time.perf_counter()
                with active_sh_adapter(adapter, True) as active_trace:
                    prepared_render = adapter.render
                totals["active"] += time.perf_counter() - setup_start
                if adapter.render is not original_render:
                    raise ValueError("The renderer was not restored after preparing the owned callable")
                for camera in cameras:
                    outputs = {}
                    for mode in (("native", "active") if trial % 2 == 0 else ("active", "native")):
                        for parameter in parameters:
                            parameter.grad = None
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = time.perf_counter()
                        selected_render = prepared_render if mode == "active" else original_render
                        package = selected_render(camera, model, pipeline, background)
                        target = camera.original_image.cuda()
                        if camera.alpha_mask is not None:
                            package["render"] *= camera.alpha_mask.cuda()
                        loss = .8 * l1_loss(package["render"], target) + .2 * (1 - ssim(package["render"], target))
                        loss.backward()
                        torch.cuda.synchronize()
                        totals[mode] += time.perf_counter() - before
                        peaks[mode] = max(peaks[mode], torch.cuda.max_memory_allocated())
                        if torch.cuda.max_memory_reserved() > 12 * 1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 32 * 1024**3:
                            raise MemoryError("The original diagnostic memory budget was exceeded")
                        check_begin = time.perf_counter()
                        outputs[mode] = {"pixels": package["render"].detach().cpu().clone(), "loss": float(loss.detach()),
                            "radii": package["radii"].detach().cpu().clone(),
                            "density": package["viewspace_points"].grad.detach().cpu().clone(),
                            "gradients": [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]}
                        verification += time.perf_counter() - check_begin
                        traces[mode].append(dict(active_trace) if mode == "active" else {"enabled": False})
                        del package, target, loss
                    check_begin = time.perf_counter()
                    a, b = outputs["native"], outputs["active"]
                    gradient_guards = [x is not None and y is not None and torch.allclose(x, y, atol=2e-6, rtol=1e-4)
                                       for x, y in zip(a["gradients"], b["gradients"])]
                    row = {"camera": camera.image_name, "pixels_max_abs": float((a["pixels"] - b["pixels"]).abs().max()),
                        "loss_max_abs": abs(a["loss"] - b["loss"]), "integer_radii_exact": torch.equal(a["radii"], b["radii"]),
                        "all_six_gradient_groups": all(gradient_guards), "gradient_groups": gradient_guards,
                        "screen_gradient_guard": torch.allclose(a["density"], b["density"], atol=2e-6, rtol=1e-4),
                        "complete_unused_rest_gradient": b["gradients"][-1] is not None}
                    guards.append(row)
                    verification += time.perf_counter() - check_begin
                    del outputs, a, b
                    gc.collect()
                    atomic_json(args.output / "PROGRESS.json", {"iteration": iteration, "degree": degree,
                        "trial": trial, "guards": guards, "partial_diagnostic_seconds": totals, "counted": False})
                    if row["pixels_max_abs"] > 1e-5 or row["loss_max_abs"] > 1e-7 or not row["integer_radii_exact"] or not row["all_six_gradient_groups"] or not row["screen_gradient_guard"] or not row["complete_unused_rest_gradient"]:
                        raise ValueError("An original actual pixel, loss, model or density gradient guard failed")
                pair = {"trial": trial, "complete24_render_loss_backward_seconds": totals,
                    "observed_GPU_total_peak_allocation_bytes": peaks, "all24_guards": guards, "traces": traces,
                    "oracle_copies_and_comparison_seconds_separate": verification}
                rows.append(pair)
                atomic_json(args.output / "STATE.json", {**state, "current_iteration": iteration, "pairs": rows})
                print(json.dumps({key: pair[key] for key in ("trial", "complete24_render_loss_backward_seconds", "observed_GPU_total_peak_allocation_bytes")}), flush=True)
            medians = {mode: statistics.median(row["complete24_render_loss_backward_seconds"][mode] for row in rows) for mode in totals}
            reduction = 1 - medians["active"] / medians["native"]
            guarded = all(row["complete24_render_loss_backward_seconds"]["active"] <= 1.1 * row["complete24_render_loss_backward_seconds"]["native"] for row in rows)
            result = {"iteration": iteration, "degree": degree, "actual_gaussians": len(model.get_xyz), "pairs": rows,
                "median_seconds": medians, "stage_reduction": reduction, "every_pair_cost_guard": guarded,
                "diagnostic_speed_gate": reduction >= .1 and guarded, "full_training_quality_and_recovery_pending": True}
            state["results"].append(result)
            atomic_json(args.output / "STATE.json", state)
            del model, parameters
            gc.collect()
            torch.cuda.empty_cache()
        unchanged = identities == [data_identity(path) for path in paths]
        speed = state["results"][0]["diagnostic_speed_gate"] and all(row["every_pair_cost_guard"] for row in state["results"])
        state.update(status="diagnostic_passed_full_training_pending" if speed and unchanged else "failed_verification",
                     sources_inputs_unchanged=unchanged, diagnostic_speed_passed=speed, elapsed_seconds=time.perf_counter() - began)
        atomic_json(args.output / "RESULT.json", state)
        print(json.dumps({key: state[key] for key in ("status", "sources_inputs_unchanged", "diagnostic_speed_passed", "elapsed_seconds")}), flush=True)
        if not unchanged:
            raise ValueError("An original source or actual input changed")
    except BaseException as error:
        state.update(status="failed_verification", error=repr(error), elapsed_seconds=time.perf_counter() - began)
        atomic_json(args.output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
