"""Profile native raster execution using actual early and final trained model states."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from gsplat_execution_specialization import gsplat_execution_specialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "build", "early-run", "final-run", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mechanism", choices=["both", "active_sh", "late_density"], default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(args.runtime)
    import torch
    import gsplat_adapter as adapter
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    build = adapter.activate_gsplat(args.build)
    rows = json.loads(args.localization.read_text())["rows"]
    intr = json.loads(args.localization.read_text())["PINHOLE_camera"]
    if len(rows) != 24 or any(row["status"] != "localized" for row in rows):
        raise ValueError("Every original actual development view is required")
    sources = [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("gsplat_execution_specialization.py"))]
    protocol = {"trials": 3, "all24_views_each": True, "model_states": "native early1100 SH1 and final12000 SH3",
        "build": build, "sources": sources, "activeSH_minimum_actual_color_payload_reduction": .1,
        "late_density_minimum_full_render_backward_reduction": .1, "maximum_complete_time_ratio": 1.1,
        "forward_max_abs": 1e-5, "gradient_atol": 2e-6, "gradient_rtol": 1e-4,
        "no_raster_kernel_precision_or_training_formula_change": True, "complete_training_quality_pending": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    native = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = native.resized(round(native.width/2), round(native.height/2))
    cameras = []
    for row in rows:
        with Image.open(mounted_path(row["image_reference"])) as image:
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
                2*math.atan(native.width/(2*native.fx)), 2*math.atan(native.height/(2*native.fy)),
                None, image.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
        cameras.append(apply_to_camera(camera, scaled))
    pipeline = SimpleNamespace(antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device="cuda")
    results, checks = {}, []
    state = {"status": "running", "results": results, "checks": checks, "counted": False}
    try:
        for mechanism, folder, iteration, degree in (("active_sh", args.early_run, 1100, 1), ("late_density", args.final_run, 12000, 3)):
            if args.mechanism not in ("both", mechanism):
                continue
            native_state = json.loads((folder/"STATE.json").read_text())
            config = json.loads((folder/"CONFIG.json").read_text())
            if native_state["status"] != "completed" or native_state["iteration"] != iteration or config["parameters"]["sh_degree"] != 3:
                raise ValueError("A native actual model state or degree schedule changed")
            path = folder/f"model/point_cloud/iteration_{iteration}/point_cloud.ply"
            model = GaussianModel(3)
            model.load_ply(str(path))
            model.active_sh_degree = degree
            model._execution_iteration = iteration
            model._execution_density_until = config["parameters"]["densify_until_iter"]
            if mechanism == "late_density" and iteration < model._execution_density_until:
                raise ValueError("An early native iteration cannot skip density feedback")
            parameters = [getattr(model, name) for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")]
            pairs, numerical = [], []
            for trial in range(3):
                values, seconds, peaks, traces = {}, {}, {}, {}
                order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                for mode in order:
                    values[mode] = []
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with gsplat_execution_specialization(active_sh=mode == "candidate" and mechanism == "active_sh",
                            late_density=mode == "candidate" and mechanism == "late_density") as trace:
                        elapsed, peak = time.perf_counter()-begin, 0
                        for camera in cameras:
                            for parameter in parameters:
                                parameter.grad = None
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            before_bytes = torch.cuda.memory_allocated()
                            before = time.perf_counter()
                            output = adapter.render(camera, model, pipeline, background)
                            output["render"].square().mean().backward()
                            torch.cuda.synchronize()
                            elapsed += time.perf_counter()-before
                            peak = max(peak, torch.cuda.max_memory_allocated()-before_bytes)
                            if torch.cuda.max_memory_reserved() > 12*1024**3:
                                raise MemoryError("The exact renderer exceeded its declared CUDA budget")
                            values[mode].append({"image": output["render"].detach().cpu().clone(),
                                "grads": [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters],
                                "radii": None if output["radii"] is None else output["radii"].detach().cpu().clone(),
                                "proxy_grad": None if output["viewspace_points"] is None else output["viewspace_points"].grad.detach().cpu().clone()})
                            del output
                        traces[mode] = trace
                        # Context preparation is included once per complete entry; oracle copies are reported separately.
                        seconds[mode] = elapsed
                        peaks[mode] = peak
                    traces[mode]["complete_profile_seconds_including_oracle_copies"] = time.perf_counter()-begin
                for index, row in enumerate(rows):
                    a, b = values["baseline"][index], values["candidate"][index]
                    gradients = [x is not None and y is not None and torch.allclose(x, y, atol=2e-6, rtol=1e-4)
                        for x, y in zip(a["grads"], b["grads"])]
                    item = {"mechanism": mechanism, "trial": trial, "name": row["image_name"],
                        "forward_max_abs": float((a["image"]-b["image"]).abs().max()), "six_gradient_groups_preserved": all(gradients),
                        "gradient_groups": gradients,
                        "gradient_max_abs": [None if x is None or y is None else float((x-y).abs().max()) for x, y in zip(a["grads"], b["grads"])],
                        "unused_rest_gradient_still_present": b["grads"][-1] is not None,
                        "early_density_feedback_preserved": mechanism != "active_sh" or (torch.equal(a["radii"], b["radii"]) and
                            torch.allclose(a["proxy_grad"], b["proxy_grad"], atol=2e-6, rtol=1e-4))}
                    numerical.append(item)
                    if item["forward_max_abs"] > 1e-5 or not item["six_gradient_groups_preserved"] or not item["early_density_feedback_preserved"]:
                        atomic_json(args.output/"FAILURE.json", {"status": "failed_original_numerical_guard", "row": item, "counted": False})
                        raise AssertionError("An exact renderer changed actual pixels or required gradients")
                pair = {"trial": trial, "full24_render_backward_seconds": seconds, "peak_stage_allocation_bytes": peaks, "traces": traces}
                pairs.append(pair)
                print(json.dumps({"mechanism": mechanism, "trial": trial, "seconds": seconds, "peak_bytes": peaks}), flush=True)
                atomic_json(args.output/"STATE.json", {**state, "measuring": mechanism, "current_pairs": pairs})
            median = {mode: statistics.median(p["full24_render_backward_seconds"][mode] for p in pairs) for mode in ("baseline", "candidate")}
            reduction = 1-median["candidate"]/median["baseline"]
            payload = 1-(degree+1)**2/16 if mechanism == "active_sh" else None
            benefit = payload >= .1 if mechanism == "active_sh" else reduction >= .1
            protected = all(p["full24_render_backward_seconds"]["candidate"] <= 1.1*p["full24_render_backward_seconds"]["baseline"] for p in pairs)
            results[mechanism] = {"status": "profile_benefit_passed_training_quality_pending" if benefit and protected else "rejected_insufficient_benefit",
                "model_identity": data_identity(path), "iteration": iteration, "active_degree": degree,
                "pairs": pairs, "numerical_checks": numerical, "median_seconds": median, "full_render_backward_reduction": reduction,
                "actual_native_color_payload_reduction": payload, "complete_time_guards": protected, "counted": False}
            del model, parameters, values
            torch.cuda.empty_cache()
        state.update(status="completed_training_quality_pending", sources_unchanged=sources == protocol["sources"])
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
