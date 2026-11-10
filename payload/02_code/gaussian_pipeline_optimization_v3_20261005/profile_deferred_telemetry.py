"""Diagnose native synchronization versus bounded telemetry on actual mature steps."""

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

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from deferred_training_telemetry import DeferredTelemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "build", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = activate_reference(args.runtime)
    import torch
    import gsplat_adapter
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams, PipelineParams
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    build = gsplat_adapter.activate_gsplat(args.build)
    localization = json.loads(args.localization.read_text())
    if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
        raise ValueError("Retain the complete development camera denominator")
    paths = [Path(__file__), Path(__file__).with_name("deferred_training_telemetry.py"), args.model, args.localization,
        V1/"gsplat_adapter.py", V1/"runtime.py"]
    identities = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identities, "runtime": runtime, "build": build,
        "three_paired_mature216_step_diagnostics": True, "same_initial_native_parameters_and_camera_order": True,
        "native_render_copy_loss_backward_dense_Adam_formulas_unchanged": True,
        "native_sync_every_step_vs_ten_scalar_queue": True, "event_allocation_and_complete_final_drain_in_cost": True,
        "minimum_complete_mature_step_reduction_to_proceed": .1,
        "original_full_all24_values_gradients_and_three_seed12k_training_required_later": True,
        "no_densification_or_full_Scene_loading_or_model_export_in_stage": True,
        "maximum_RSS_bytes": 32*1024**3, "maximum_CUDA_bytes": 12*1024**3,
        "counted": False, "no_new_object_or_full_training_acceptance_from_this_diagnostic": True})
    intr = localization["PINHOLE_camera"]
    native = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = native.resized(round(native.width/2), round(native.height/2))
    cameras = []
    for row in localization["rows"]:
        with Image.open(mounted_path(row["image_reference"])) as photo:
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
                2*math.atan(native.width/(2*native.fx)), 2*math.atan(native.height/(2*native.fy)), None,
                photo.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
        apply_to_camera(camera, scaled)
        cameras.append(camera)
    native_parser = argparse.ArgumentParser()
    opt_group, pipe_group = OptimizationParams(native_parser), PipelineParams(native_parser)
    native_args = native_parser.parse_args([])
    optimization, pipeline = opt_group.extract(native_args), pipe_group.extract(native_args)
    model = GaussianModel(3)
    with vectorized_gaussian_read(True):
        model.load_ply(str(args.model))
    model.pretrained_exposures = None
    model.exposure_mapping = {camera.image_name: i for i, camera in enumerate(cameras)}
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(len(cameras), 1, 1))
    model.spatial_lr_scale = 1.
    model.training_setup(optimization)
    snapshot = copy.deepcopy(model.capture())
    background = torch.zeros(3, device="cuda")
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient isolated CUDA headroom")

    # The queue preserves detached scalar values, exact chronological EMA and gradients.
    oracle_queue = DeferredTelemetry(10, initial_ema=.125)
    native_ema, expected = .125, []
    leaves = []
    for index in range(257):
        start, end = oracle_queue.event_pair()
        start.record()
        leaf = torch.tensor((index % 19)/19, device="cuda", dtype=torch.float32, requires_grad=True)
        scalar = leaf.square()+.00390625
        scalar.backward()
        end.record()
        value = scalar.item()
        native_ema = .4*value+.6*native_ema
        expected.append(value)
        oracle_queue.push(scalar, start, end, force=index == 128)
        leaves.append((leaf, float(leaf.detach())))
    oracle_trace = oracle_queue.close()
    oracle_checks = {"all257_native_float32_values_and_CPU_EMA_bit_exact": oracle_queue.values == expected and oracle_queue.ema == native_ema,
        "all257_gradients_unchanged_by_detached_logging": all(torch.equal(leaf.grad, 2*leaf.detach()) for leaf, _ in leaves),
        "capacity_ten_and_complete_forced_partial_drain": oracle_trace["maximum_pending"] <= 10 and oracle_trace["samples"] == 257
            and oracle_trace["all_samples_and_events_consumed"],
        "no_autograd_graph_retained_in_queue": not oracle_queue.pending}
    atomic_json(args.output/"SCALAR_ORACLE.json", {"checks": oracle_checks, "trace": oracle_trace, "counted": False})
    if not all(oracle_checks.values()):
        raise ValueError("The detached scalar telemetry oracle failed")
    del oracle_queue, leaves
    rows = []
    for trial in range(3):
        times, telemetry = {}, {}
        for mode in (("native", "deferred") if trial % 2 == 0 else ("deferred", "native")):
            model.restore(copy.deepcopy(snapshot), optimization)
            model.update_learning_rate(12000)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            queue = DeferredTelemetry(10) if mode == "deferred" else None
            events = None if queue else (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ema, values, timings = 0., [], []
            for index in range(216):
                camera = cameras[index % 24]
                start, end = queue.event_pair() if queue else events
                start.record()
                package = gsplat_adapter.render(camera, model, pipeline, background)
                image = package["render"]
                if camera.alpha_mask is not None:
                    image *= camera.alpha_mask.cuda()
                target = camera.original_image.cuda()
                l1 = l1_loss(image, target)
                loss = .8*l1+.2*(1-ssim(image, target))
                loss.backward()
                end.record()
                with torch.no_grad():
                    if queue:
                        ema = queue.push(loss, start, end)
                    else:
                        value = loss.item()
                        ema = .4*value+.6*ema
                        values.append(value)
                        timings.append(start.elapsed_time(end))
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
            if queue:
                queue.close()
                values, timings, ema = queue.values, queue.elapsed_ms, queue.ema
            torch.cuda.synchronize()
            times[mode] = time.perf_counter()-before
            telemetry[mode] = {"samples": len(values), "EMA": ema, "mean_loss": sum(values)/len(values),
                "all216_losses_and_actual_GPU_timings": len(values) == len(timings) == 216 and all(math.isfinite(x) for x in values+timings),
                "trace": queue.trace if queue else None,
                "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()}
            if torch.cuda.max_memory_reserved() > budget or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 32*1024**3:
                raise MemoryError("The mature telemetry diagnostic exceeded its memory budget")
        row = {"trial": trial, "complete_mature_step_seconds": times, "telemetry": telemetry}
        rows.append(row)
        atomic_json(args.output/"STATE.json", {"status": "running", "trials": rows})
        print(json.dumps(row), flush=True)
    median = {mode: statistics.median(row["complete_mature_step_seconds"][mode] for row in rows) for mode in ("native", "deferred")}
    reduction = 1-median["deferred"]/median["native"]
    checks = {"scalar_oracle": all(oracle_checks.values()), "three_complete_pairs": len(rows) == 3,
        "all_logged_scalar_and_GPU_timing_denominators_preserved": all(all(t["all216_losses_and_actual_GPU_timings"] for t in row["telemetry"].values()) for row in rows),
        "sources_and_original_model_localization_unchanged": identities == [data_identity(p) for p in paths]}
    result = {"status": "diagnostic_cost_passed_full_numeric_training_recovery_pending" if reduction >= .1 and all(checks.values()) else "rejected_insufficient_mature_step_effect",
        "checks": checks, "trials": rows, "median_seconds": median, "stage_reduction": reduction, "counted": False,
        "full_actual_training_and_all24_gradients_not_executed": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "median_seconds", "stage_reduction")}), flush=True)
    if not all(checks.values()):
        raise ValueError("A complete mature telemetry diagnostic invariant failed")


if __name__ == "__main__":
    main()
