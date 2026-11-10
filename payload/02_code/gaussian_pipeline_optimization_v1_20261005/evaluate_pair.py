"""Evaluate both trained variants with identical calibrated development cameras."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from full_intrinsics import Pinhole, apply_to_camera
from runtime import activate_reference, atomic_json, data_identity


def mounted_path(value):
    value = value.replace("\\", "/")
    if len(value) > 2 and value[1] == ":":
        return Path("/mnt/" + value[0].lower() + value[2:])
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--control-label", default="legacy")
    parser.add_argument("--candidate-label", default="full")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--control-iterations", type=int)
    parser.add_argument("--candidate-iterations", type=int)
    parser.add_argument("--evaluation-folder", default="evaluation")
    parser.add_argument("--localization", type=Path)
    parser.add_argument("--input-version", default="C212_frozen")
    args = parser.parse_args()
    if args.control_label == args.candidate_label or "ground_truth" in (args.control_label, args.candidate_label):
        raise ValueError("Model labels must be distinct from each other and the reference image label")
    if any(Path(label).name != label for label in (args.control_label, args.candidate_label)):
        raise ValueError("Model labels must be direct child directory names")
    iterations_by_label = {args.control_label: args.iterations if args.control_iterations is None else args.control_iterations,
                           args.candidate_label: args.iterations if args.candidate_iterations is None else args.candidate_iterations}
    if min(iterations_by_label.values()) < 1:
        raise ValueError("Both declared training budgets must be positive")
    if (Path(args.evaluation_folder).name != args.evaluation_folder
            or args.evaluation_folder in ("", ".", "..")
            or not args.evaluation_folder.startswith("evaluation")):
        raise ValueError("Evaluation output must be a new direct evaluation child directory")
    output = args.pair/args.evaluation_folder
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim

    localization_path = args.localization or args.reference_run/"05_evaluation/C_augmented_212/H_LOCALIZATION.json"
    localization = json.loads(localization_path.read_text())
    if args.localization and (args.input_version == "C212_frozen"
            or localization.get("input_version") != args.input_version):
        raise ValueError("An explicit new localization requires its own matching input version")
    if len(localization["rows"]) != 24 or len({row["image_name"] for row in localization["rows"]}) != 24:
        raise ValueError("The complete frozen 24-view evaluation denominator is required")
    intr = localization["PINHOLE_camera"]
    original = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = original.resized(round(original.width/2), round(original.height/2))
    sx, sy = scaled.width/original.width, scaled.height/original.height
    calibration = {"localization_source": data_identity(localization_path), "K": asdict(scaled),
        "input_version": args.input_version, "evaluator_source": data_identity(Path(__file__)),
        "evaluation_camera_refitted": False, "split_role": "previously_viewed_development",
        "training_iterations_by_label": iterations_by_label,
        "frame_count": len(localization["rows"]), "images": [data_identity(mounted_path(row["image_reference"])) for row in localization["rows"]]}
    atomic_json(output/"EVALUATION_CONTRACT.json", calibration)
    native = argparse.ArgumentParser()
    group = PipelineParams(native)
    pipeline = group.extract(native.parse_args([]))
    background = torch.zeros(3, device="cuda")

    def camera_for(rotation, translation, photo, name):
        camera = Camera((scaled.width, scaled.height), 0, np.asarray(rotation).T, np.asarray(translation),
            2*math.atan(original.width/(2*original.fx)), 2*math.atan(original.height/(2*original.fy)),
            None, photo, None, name, 0, data_device="cpu")
        return apply_to_camera(camera, scaled)

    def metrics(predicted, reference):
        mse = float((predicted-reference).square().mean().item())
        return {"PSNR_dB": -10*math.log10(max(mse, 1e-12)),
                "SSIM": float(ssim(predicted[None], reference[None]).item()), "MSE": mse}

    def picture(tensor):
        pixels = (tensor.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(pixels)

    cameras, references = {}, {}
    for row in localization["rows"]:
        if row["status"] != "localized":
            continue
        photo = Image.open(mounted_path(row["image_reference"])).convert("RGB")
        camera = camera_for(row["R_CW"], row["t_CW"], photo, row["image_name"])
        cameras[row["image_name"]] = camera
        references[row["image_name"]] = camera.original_image.to("cuda")
    selected = [localization["rows"][i]["image_name"] for i in [0, 4, 8, 12, 16, 20]]
    tiles = {(name, "ground_truth"): picture(references[name]) for name in selected if name in references}
    results = {}
    for label in [args.control_label, args.candidate_label]:
        expected_iterations = iterations_by_label[label]
        state = json.loads((args.pair/label/"STATE.json").read_text())
        if state["status"] != "completed" or state["iteration"] != expected_iterations:
            raise RuntimeError("Both runs must complete the declared iteration count before comparison")
        if args.localization:
            manifest = json.loads((args.pair/label/"INPUT_MANIFEST.json").read_text())
            if manifest["input_version"] != args.input_version:
                raise ValueError("The trained model and explicit evaluation use different input versions")
        gaussian = GaussianModel(3)
        model_path = args.pair/label/f"model/point_cloud/iteration_{expected_iterations}/point_cloud.ply"
        gaussian.load_ply(str(model_path))
        rows = []
        for source in localization["rows"]:
            name = source["image_name"]
            row = {"image_name": name, "localized": name in cameras, "quality_eligible": source["quality_eligible"]}
            if name not in cameras:
                rows.append(row)
                continue
            with torch.no_grad():
                predicted = render(cameras[name], gaussian, pipeline, background)["render"]
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Non-finite evaluation render")
            row["full_image"] = metrics(predicted, references[name])
            window = source.get("target_window_PINHOLE")
            if window:
                x1, y1, x2, y2 = [round(x*scale) for x, scale in zip(window, [sx, sy, sx, sy])]
                if min(x2-x1, y2-y1) >= 11:
                    row["target"] = metrics(predicted[:, y1:y2, x1:x2], references[name][:, y1:y2, x1:x2])
                    row["target_window"] = [x1, y1, x2, y2]
            rows.append(row)
            if name in selected:
                tiles[(name, label)] = picture(predicted)
        summary = {}
        for region in ["full_image", "target"]:
            values = [row[region] for row in rows if region in row]
            summary[region] = {"count": len(values),
                "mean_PSNR_dB": float(np.mean([row["PSNR_dB"] for row in values])),
                "mean_SSIM": float(np.mean([row["SSIM"] for row in values]))}
        results[label] = {"model": data_identity(model_path), "summary": summary, "rows": rows,
                          "training_call_seconds": state["training_call_seconds"]}
        del gaussian
        torch.cuda.empty_cache()
    deltas = {region: {metric: results[args.candidate_label]["summary"][region][metric]-results[args.control_label]["summary"][region][metric]
                       for metric in ["mean_PSNR_dB", "mean_SSIM"]} for region in ["full_image", "target"]}
    guard = (all(row["mean_PSNR_dB"] >= -.1 and row["mean_SSIM"] >= -.002 for row in deltas.values())
        and all(results[label]["summary"][region]["count"] == 24
                for label in results for region in ("full_image", "target")))
    cell = (406, 310)
    sheet = Image.new("RGB", (cell[0]*3, 30+(cell[1]+22)*len(selected)), "#202020")
    draw = ImageDraw.Draw(sheet)
    labels = ["ground_truth", args.control_label, args.candidate_label]
    for col, label in enumerate(labels):
        draw.text((col*cell[0]+8, 8), label+" / fixed full K evaluator", fill="white")
    for row_index, name in enumerate(selected):
        top = 30+row_index*(cell[1]+22)
        for col, label in enumerate(labels):
            if (name, label) in tiles:
                image = tiles[(name, label)].copy()
                image.thumbnail(cell)
                sheet.paste(image, (col*cell[0], top))
            draw.text((col*cell[0]+8, top+cell[1]+4), name, fill="white")
    sheet.save(output/"SIX_VIEW_COMPARISON.jpg", quality=95)
    receipt = {"status": "metrics_completed_visual_review_pending", "runtime": metadata,
        "evaluation_protocol": "same_frozen_"+args.input_version+"_H_poses_full_K_for_both_models",
        "all_frames_included_in_denominator": len(localization["rows"]),
        "strict_blind_test": False, "results": results, "candidate_minus_control": deltas,
        "control_label": args.control_label, "candidate_label": args.candidate_label,
        "iterations": next(iter(iterations_by_label.values())) if len(set(iterations_by_label.values())) == 1 else None,
        "training_iterations_by_label": iterations_by_label,
        "development_metric_guard_passed": guard, "independent_pose_accuracy_claim": False,
        "elapsed_seconds": time.monotonic()-started, "counted_as_usable_optimization": False}
    if args.control_label == "legacy" and args.candidate_label == "full":
        receipt["full_minus_legacy"] = deltas
    atomic_json(output/"EVALUATION.json", receipt)
    print(json.dumps({"status": receipt["status"], "candidate_minus_control": deltas, "metric_guard": guard}))


if __name__ == "__main__":
    main()
