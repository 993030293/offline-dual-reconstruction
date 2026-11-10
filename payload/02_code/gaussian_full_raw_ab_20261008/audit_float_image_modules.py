"""Compare image mechanisms on complete native float renders of current raw models."""

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from render_evaluation import mounted


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
    import torch
    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import create_window, ssim
    from device_artifact_picture import DeviceArtifactPicture, native_picture
    from reference_ssim import ReferenceSSIM
    from interior_ssim_reuse import InteriorSSIM
    from fused_ssim_formula_v2 import FusedInteriorSSIM
    from view_major_evaluation import finish_reference_view
    torch.set_num_threads(1)
    source_paths = [Path(__file__), V3 / "device_artifact_picture.py", V2 / "reference_ssim.py",
        V3 / "interior_ssim_reuse.py", V3 / "fused_ssim_formula.py", V3 / "fused_ssim_formula_v2.py",
        V3 / "view_major_evaluation.py", V1 / "full_intrinsics.py"]
    source_ids = [data_identity(path) for path in source_paths]
    state = {"status": "running", "sources": source_ids, "denominator": 24,
        "native_float_render_inputs": True, "PNG_predictions_read": False,
        "tolerance": 1e-6, "primary_results_unchanged": True, "families": {}}
    atomic_json(args.output / "CONTRACT.json", state)
    try:
        parsed = argparse.ArgumentParser()
        group = PipelineParams(parsed)
        pipeline = group.extract(parsed.parse_args([]))
        background = torch.zeros(3, device="cuda")
        images = {}
        model_paths, model_ids, failures = [], [], []
        before = time.perf_counter()
        for mode in ("baseline", "optimized"):
            side = args.campaign / args.case / mode
            training = read(side / "training/STATE.json")
            if training["status"] != "completed" or training["iteration"] != training["target_iterations"]:
                raise ValueError("Native float audit requires complete current training")
            path = side / f"training/model/point_cloud/iteration_{training['iteration']}/point_cloud.ply"
            model_paths.append(path)
            model_ids.append(data_identity(path))
            model = GaussianModel(3)
            model.load_ply(str(path))
            location = read(side / "localization/LOCALIZATION.json")
            if len(location["rows"]) != 24:
                raise ValueError("The evaluation denominator changed")
            intr = location["PINHOLE_camera"]
            native = Pinhole(intr["width"], intr["height"], *intr["params"])
            scaled = native.resized(round(native.width/2), round(native.height/2))
            for row in location["rows"]:
                if row["status"] != "localized":
                    failures.append({"mode": mode, "image_name": row["image_name"], "status": row["status"]})
                    continue
                reference_path = mounted(row["image_reference"])
                with Image.open(reference_path) as photo:
                    camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T,
                        np.asarray(row["t_CW"]), 2*math.atan(native.width/(2*native.fx)),
                        2*math.atan(native.height/(2*native.fy)), None, photo.convert("RGB"),
                        None, row["image_name"], 0, data_device="cpu")
                camera = apply_to_camera(camera, scaled)
                with torch.no_grad():
                    prediction = render(camera, model, pipeline, background)["render"].unsqueeze(0)
                reference = camera.original_image.unsqueeze(0).cuda()
                if not torch.isfinite(prediction).all().item():
                    raise FloatingPointError("A complete native float render is nonfinite")
                sx, sy = scaled.width/native.width, scaled.height/native.height
                x1, y1, x2, y2 = [round(v*s) for v, s in zip(row["central_window_PINHOLE"], [sx, sy, sx, sy])]
                entry = images.setdefault(row["image_name"], {"prediction": {}, "reference": reference,
                    "reference_bytes": reference_path.read_bytes(), "crop": (y1, y2, x1, x2)})
                if entry["reference_bytes"] != reference_path.read_bytes() or not torch.equal(entry["reference"], reference):
                    raise ValueError("The complete raw reference pixels differ between sides")
                entry["prediction"][mode] = prediction
            del model
            gc.collect()
        images = {key: row for key, row in images.items() if len(row["prediction"]) == 2}
        state.update(models=model_ids, native_complete_render_setup_seconds=time.perf_counter()-before,
            common_complete_images=len(images), failures=failures, overall_quality_accepted=False)
        if not images:
            raise ValueError("No complete common float render is available")

        def clock():
            torch.cuda.synchronize()
            return time.perf_counter()

        quantized = []
        for trial in range(3):
            outputs, seconds, traces = {}, {}, {}
            for label in (("native", "device") if trial != 1 else ("device", "native")):
                adapter = DeviceArtifactPicture(True)
                before = clock()
                outputs[label] = [native_picture(pred[0]) if label == "native" else adapter(pred[0])
                    for row in images.values() for pred in row["prediction"].values()]
                seconds[label] = clock()-before
                if label == "device":
                    traces[label] = adapter.calls
            quantized.append({"trial": trial, "seconds": seconds,
                "all_complete_float_render_pixels_exact": all(a.tobytes() == b.tobytes() for a, b in zip(outputs["native"], outputs["device"])),
                "trace": traces})
        state["families"]["R100"] = quantized
        metric_rows, original = [], None
        for trial in range(3):
            labels = ("native", "reference", "view_major", "interior", "fused")
            for label in labels if trial != 1 else tuple(reversed(labels)):
                constructor = {"reference": ReferenceSSIM, "view_major": ReferenceSSIM,
                    "interior": InteriorSSIM, "fused": FusedInteriorSSIM}.get(label)
                cache = constructor(create_window, 1024**3) if constructor else None
                operation = cache if cache else ssim
                values, released = {}, []
                sequence = [(mode, name) for mode in ("baseline", "optimized") for name in images] if label != "view_major" else [(mode, name) for name in images for mode in ("baseline", "optimized")]
                before = clock()
                for mode, name in sequence:
                    row = images[name]
                    prediction, reference = row["prediction"][mode], row["reference"]
                    y1, y2, x1, x2 = row["crop"]
                    with torch.no_grad():
                        values[mode+"/"+name] = [float(operation(prediction, reference).item()),
                            float(operation(prediction[:, :, y1:y2, x1:x2], reference[:, :, y1:y2, x1:x2]).item())]
                    if label == "view_major" and mode == "optimized":
                        finish_reference_view(cache, released)
                duration = clock()-before
                if label == "native" and original is None:
                    original = values
                metric_rows.append({"trial": trial, "label": label, "seconds": duration,
                    "values": values, "trace": cache.receipt() if cache else None, "last_use_release": released})
                if cache:
                    cache.clear()
        for row in metric_rows:
            differences = [abs(value-original[key][i]) for key, values in row["values"].items() for i, value in enumerate(values)]
            row["maximum_absolute_delta"] = max(differences)
            row["all_values_within_frozen_tolerance"] = max(differences) <= 1e-6
        state["families"]["Q30_R135_R134_R172"] = metric_rows
        state.update(status="complete_native_float_image_module_outcomes_recorded", elapsed_seconds=time.perf_counter()-started,
            sources_unchanged=source_ids == [data_identity(path) for path in source_paths],
            models_unchanged=model_ids == [data_identity(path) for path in model_paths],
            maximum_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
            all_float_pixels_exact=all(row["all_complete_float_render_pixels_exact"] for row in quantized),
            all_float_SSIM_values_within_frozen_tolerance=all(row["all_values_within_frozen_tolerance"] for row in metric_rows))
        if state["maximum_CUDA_reserved_bytes"] > 12*1024**3:
            raise MemoryError("Native float audit exceeded the declared CUDA budget")
        atomic_json(args.output / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_native_float_image_module_audit", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
