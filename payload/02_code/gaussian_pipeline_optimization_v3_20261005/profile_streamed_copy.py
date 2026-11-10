"""Compare render-copy-loss-backward makespan on all actual calibrated development inputs."""

import argparse
import json
import math
from pathlib import Path
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
from streamed_camera_copy import CameraCopy


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
    from arguments import PipelineParams
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    build = gsplat_adapter.activate_gsplat(args.build)
    localization = json.loads(args.localization.read_text())
    if len(localization["rows"]) != 24 or any(row["status"] != "localized" for row in localization["rows"]):
        raise ValueError("The complete actual24 denominator changed")
    identities = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("streamed_camera_copy.py"), args.model, args.localization)]
    atomic_json(args.output/"PROTOCOL.json", {"runtime": runtime, "build": build, "sources_inputs": identities,
        "trials": 3, "minimum_complete_makespan_reduction": .1, "same24_real_cameras_and_pixels": True,
        "staging_prepare_allocation_and_waits_included": True, "loss_max_abs": 1e-7,
        "input_tensor_values_dtype_and_stride_exact": True, "render_pixel_max_abs": 1e-5,
        "single_slot_pinned_and_single_copy_stream_only": True, "maximum_cuda_bytes": 12*1024**3,
        "sampling_unchanged_no_future_access": True, "training_and_recovery_pending": True, "counted": False})
    intrinsic = localization["PINHOLE_camera"]
    native = Pinhole(intrinsic["width"], intrinsic["height"], *intrinsic["params"])
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
    parameters = [getattr(model, name) for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")]
    parser_native = argparse.ArgumentParser()
    group = PipelineParams(parser_native)
    pipeline = group.extract(parser_native.parse_args([]))
    background = torch.zeros(3, device="cuda")
    trials = []
    for trial in range(3):
        values, seconds, traces, peaks = {}, {}, {}, {}
        for mode in (["native", "nonblocking", "streamed"] if trial % 2 == 0 else ["streamed", "nonblocking", "native"]):
            outputs, elapsed = [], 0.
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            transfer = CameraCopy(mode)
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-before
            try:
                for camera in cameras:
                    for parameter in parameters:
                        parameter.grad = None
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    transfer.begin(camera.original_image)
                    predicted = gsplat_adapter.render(camera, model, pipeline, background)["render"]
                    copied = transfer.consume()
                    loss = .8*l1_loss(predicted, copied)+.2*(1-ssim(predicted, copied))
                    loss.backward()
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter()-before
                    exact = copied.shape == camera.original_image.shape and copied.dtype == camera.original_image.dtype
                    exact = exact and copied.stride() == camera.original_image.stride() and torch.equal(copied.cpu(), camera.original_image)
                    outputs.append({"input_exact": exact, "loss": float(loss.detach()), "pixels": predicted.detach().cpu().clone()})
                    if torch.cuda.max_memory_reserved() > 12*1024**3:
                        raise MemoryError("A copy-render combination exceeded its CUDA budget")
                before = time.perf_counter()
                transfer.close()
                elapsed += time.perf_counter()-before
            finally:
                transfer.close()
            values[mode], seconds[mode], traces[mode], peaks[mode] = outputs, elapsed, transfer.trace, torch.cuda.max_memory_allocated()
        checks = {}
        for mode in ("nonblocking", "streamed"):
            checks[mode] = all(a["input_exact"] and b["input_exact"] and abs(a["loss"]-b["loss"]) <= 1e-7
                and float((a["pixels"]-b["pixels"]).abs().max()) <= 1e-5 for a, b in zip(values["native"], values[mode]))
        row = {"trial": trial, "render_copy_loss_backward_seconds": seconds, "all24_original_guards": checks,
            "trace": traces, "cuda_allocated_peaks": peaks}
        trials.append(row)
        atomic_json(args.output/"STATE.json", {"status": "running", "trials": trials})
        print(json.dumps(row), flush=True)
        if not all(checks.values()):
            raise ValueError("An original input layout, pixel or loss guard failed")
    median = {mode: statistics.median(row["render_copy_loss_backward_seconds"][mode] for row in trials) for mode in ("native", "nonblocking", "streamed")}
    results = {mode: {"reduction": 1-median[mode]/median["native"],
        "status": "stage_passed_training_recovery_pending" if 1-median[mode]/median["native"] >= .1 else "rejected_insufficient_complete_gain"}
        for mode in ("nonblocking", "streamed")}
    unchanged = identities == [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("streamed_camera_copy.py"), args.model, args.localization)]
    result = {"status": "actual_complete_copy_parallel_campaign_finished", "trials": trials, "median_seconds": median,
        "results": results, "sources_inputs_unchanged": unchanged, "three_seed_training_executed": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}), flush=True)
    if not unchanged:
        raise ValueError("A transfer source or input changed")


if __name__ == "__main__":
    main()
