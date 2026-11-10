"""Render a complete frozen denominator with the native calibrated evaluator."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from full_intrinsics import Pinhole, apply_to_camera
from runtime import activate_reference, atomic_json, data_identity


def mounted(value):
    text = str(value).replace("\\", "/")
    return Path("/mnt/" + text[0].lower() + text[2:]) if len(text) > 2 and text[1] == ":" else Path(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    state = json.loads((args.side / "training/STATE.json").read_text())
    if state["status"] != "completed" or state["iteration"] != state["target_iterations"]:
        raise ValueError("Native evaluation requires complete declared training iterations")
    localization_path = args.side / "localization/LOCALIZATION.json"
    localization = json.loads(localization_path.read_text())
    if len(localization["rows"]) != 24:
        raise ValueError("The raw evaluation denominator changed")
    intr = localization["PINHOLE_camera"]
    native_K = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = native_K.resized(round(native_K.width/2), round(native_K.height/2))
    model_path = args.side / f"training/model/point_cloud/iteration_{state['iteration']}/point_cloud.ply"
    atomic_json(args.output / "CONTRACT.json", {"runtime": metadata, "model": data_identity(model_path),
        "localization": data_identity(localization_path), "source": data_identity(Path(__file__)),
        "evaluation_intrinsics": asdict(scaled), "complete_denominator": 24,
        "render_backend": "native_INRIA_for_both_sides", "own_map_frozen_localized_poses": True,
        "map_update": False, "central_region_is_not_object_ground_truth": True,
        "strict_blind_test": False, "independent_geometry_accuracy_claim": False})
    argument_parser = argparse.ArgumentParser()
    pipeline_group = PipelineParams(argument_parser)
    pipeline = pipeline_group.extract(argument_parser.parse_args([]))
    gaussian = GaussianModel(3)
    gaussian.load_ply(str(model_path))
    background = torch.zeros(3, device="cuda")
    rows = []

    def metric(predicted, reference):
        mse = float((predicted-reference).square().mean().item())
        return {"PSNR_dB": -10*math.log10(max(mse, 1e-12)), "SSIM": float(ssim(predicted[None], reference[None]).item())}

    for source in localization["rows"]:
        row = {"image_name": source["image_name"], "quality_eligible": source["quality_eligible"],
               "localized": source["status"] == "localized", "localization_status": source["status"]}
        if row["localized"]:
            photo = Image.open(mounted(source["image_reference"])).convert("RGB")
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(source["R_CW"]).T, np.asarray(source["t_CW"]),
                2*math.atan(native_K.width/(2*native_K.fx)), 2*math.atan(native_K.height/(2*native_K.fy)),
                None, photo, None, source["image_name"], 0, data_device="cpu")
            camera = apply_to_camera(camera, scaled)
            reference = camera.original_image.to("cuda")
            with torch.no_grad():
                predicted = render(camera, gaussian, pipeline, background)["render"]
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("The complete native Gaussian render contains invalid values")
            row["full_image"] = metric(predicted, reference)
            sx, sy = scaled.width/native_K.width, scaled.height/native_K.height
            x1, y1, x2, y2 = [round(value*factor) for value, factor in zip(source["central_window_PINHOLE"], [sx, sy, sx, sy])]
            row["central_diagnostic"] = metric(predicted[:, y1:y2, x1:x2], reference[:, y1:y2, x1:x2])
            pixels = (predicted.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(args.output / (Path(source["image_name"]).stem + ".png"))
        rows.append(row)
    summaries = {}
    for region in ("full_image", "central_diagnostic"):
        values = [row[region] for row in rows if region in row]
        summaries[region] = {"rendered_count": len(values), "complete_denominator": 24,
            "mean_PSNR_dB": float(np.mean([row["PSNR_dB"] for row in values])) if values else None,
            "mean_SSIM": float(np.mean([row["SSIM"] for row in values])) if values else None}
    result = {"status": "all24_rendered_quality_comparison_pending" if len([r for r in rows if r["localized"]]) == 24 else "partial_diagnostic_render_no_quality_acceptance",
        "rows": rows, "summary": summaries, "gaussian_count": len(gaussian.get_xyz),
        "finite_gaussian_positions": bool(torch.isfinite(gaussian.get_xyz).all()),
        "elapsed_seconds": time.monotonic()-started, "training_call_seconds": state["training_call_seconds"],
        "scale_unit": "coarse_motion_aligned_meters_not_independent_GT", "strict_blind_test": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "summary", "gaussian_count")}), flush=True)


if __name__ == "__main__":
    main()
