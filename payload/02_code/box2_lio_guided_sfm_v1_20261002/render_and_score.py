"""Render frozen development leave-outs and common qualitative views with INRIA code."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--attempt", type=int, default=1)
cli = parser.parse_args()
repo = cli.runtime / "source/gaussian-splatting"
library = sorted((cli.runtime / "runtime/rasterizer-build").glob("lib.*"))
if len(library) != 1:
    raise RuntimeError("Rasterizer runtime selection is ambiguous")
sys.path[:0] = [str(repo), str(library[0])]

import torch
from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.loss_utils import ssim


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def local_path(value):
    value = value.replace("\\", "/")
    if value.startswith("C:/"):
        return Path("/mnt/c" + value[2:])
    return Path(value)


def build_camera(intrinsics, rotation_CW, translation_CW, photo, name):
    width, height = intrinsics["width"], intrinsics["height"]
    fx, fy = intrinsics["params"][:2]
    return Camera((round(width / 2), round(height / 2)), 0,
                  np.asarray(rotation_CW).T, np.asarray(translation_CW),
                  2 * math.atan(width / (2 * fx)), 2 * math.atan(height / (2 * fy)),
                  None, photo, None, name, 0, data_device="cpu")


def render_image(camera, gaussians, pipeline):
    with torch.no_grad():
        predicted = render(camera, gaussians, pipeline, torch.zeros(3, device="cuda"))["render"]
    if not torch.isfinite(predicted).all():
        raise RuntimeError("Evaluation render contains nonfinite pixels")
    return predicted


def tensor_to_image(tensor):
    pixels = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def image_metrics(predicted, target):
    mse = ((predicted - target) ** 2).mean().item()
    return {"PSNR_dB": float(-10 * math.log10(max(mse, 1e-12))),
            "SSIM_INRIA": float(ssim(predicted[None], target[None]).item()), "MSE": float(mse)}


def mean_metrics(rows, key):
    values = [r[key] for r in rows if key in r]
    return {"frame_count": len(values), "mean_PSNR_dB": float(np.mean([r["PSNR_dB"] for r in values])),
            "mean_SSIM_INRIA": float(np.mean([r["SSIM_INRIA"] for r in values]))} if values else {"frame_count": 0}


def look_at(center, focus):
    center, focus = np.asarray(center), np.asarray(focus)
    z = focus - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z]).T
    return rotation, -rotation @ center


def contact_sheet(path, columns, rows, images, cell=(406, 310)):
    sheet = Image.new("RGB", (len(columns) * cell[0], 26 + len(rows) * (cell[1] + 22)), "#141514")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        draw.text((col * cell[0] + 8, 8), title, fill="white")
    for row, label in enumerate(rows):
        top = 26 + row * (cell[1] + 22)
        for col in range(len(columns)):
            picture = images[row][col].copy()
            picture.thumbnail(cell)
            sheet.paste(picture, (col * cell[0], top))
            draw.text((col * cell[0] + 8, top + cell[1] + 4), label, fill="white")
    sheet.save(path)


def main():
    run = cli.run.resolve()
    root = run / "05_evaluation"
    output = root / ("native_renders" if cli.attempt == 1 else f"native_renders_attempt{cli.attempt}")
    output.mkdir(exist_ok=False)
    cases = ["A1_visual", "B_w_sigma015", "C_augmented_212"]
    for case in cases:
        if read_json(run / "04_downstream" / case / "MAIN_STATE.json")["status"] != "completed":
            raise RuntimeError("All registered training runs must finish before evaluation rendering")
    parser = argparse.ArgumentParser()
    group = PipelineParams(parser)
    pipeline = group.extract(parser.parse_args([]))
    baseline = read_json(root / cases[0] / "H_LOCALIZATION.json")
    for row in baseline["rows"]:
        if not local_path(row["image_reference"]).is_file():
            raise FileNotFoundError("Evaluation reference image is unavailable in WSL")
    baseline_by_name = {r["image_name"]: r for r in baseline["rows"]}
    uniform_names = [baseline["rows"][i]["image_name"] for i in [0, 4, 8, 12, 16, 20]]
    intrinsics = baseline["PINHOLE_camera"]
    source_sparse = run / "04_downstream/A1_visual/dataset/sparse/0"
    core = sorted(read_extrinsics_binary(source_sparse / "images.bin").values(), key=lambda x: x.name)
    roi = read_json(run / "03_coverage_augmented/SELECTION_MANIFEST.json")["ROI"]
    focus = np.asarray(roi["center"])
    base_pose = core[90]
    base_R = qvec2rotmat(base_pose.qvec)
    base_center = -base_R.T @ base_pose.tvec
    free_centers = [base_center + base_R.T[:, 0] * .75,
                    base_center + np.array([0., 0., .75]), focus + np.array([0., -3., 2.])]
    free_poses = [look_at(center, focus) for center in free_centers]
    summaries, all_results, tiled_H, tiled_free = [], {}, {}, {}
    start = time.monotonic()
    for case in cases:
        case_output = output / case
        case_output.mkdir()
        localization = read_json(root / case / "H_LOCALIZATION.json")
        gaussians = GaussianModel(3)
        gaussians.load_ply(str(run / "04_downstream" / case / "model/point_cloud/iteration_10000/point_cloud.ply"))
        results = []
        for row in localization["rows"]:
            record = {"image_name": row["image_name"], "quality_eligible": row["quality_eligible"],
                      "localized": row["status"] == "localized"}
            if not record["localized"]:
                results.append(record)
                continue
            photo = Image.open(local_path(row["image_reference"])).convert("RGB")
            camera = build_camera(localization["PINHOLE_camera"], row["R_CW"], row["t_CW"], photo, row["image_name"])
            predicted = render_image(camera, gaussians, pipeline)
            target = camera.original_image.to("cuda")
            record["full_image"] = image_metrics(predicted, target)
            window = baseline_by_name[row["image_name"]].get("target_window_PINHOLE")
            if window:
                x1, y1, x2, y2 = [round(x / 2) for x in window]
                if x2 - x1 >= 11 and y2 - y1 >= 11:
                    record["fixed_target_rectangle"] = image_metrics(predicted[:, y1:y2, x1:x2], target[:, y1:y2, x1:x2])
                    record["target_window_812x620"] = [x1, y1, x2, y2]
            picture = tensor_to_image(predicted)
            picture.save(case_output / (Path(row["image_name"]).stem + ".png"))
            if row["image_name"] in uniform_names:
                tiled_H[(case, row["image_name"])] = picture
                tiled_H[("reference", row["image_name"])] = tensor_to_image(target)
            results.append(record)
        blank = Image.new("RGB", (intrinsics["width"], intrinsics["height"]))
        for index, (rotation, translation) in enumerate(free_poses):
            camera = build_camera(intrinsics, rotation, translation, blank, f"free_{index}")
            picture = tensor_to_image(render_image(camera, gaussians, pipeline))
            picture.save(case_output / f"free_{index}.png")
            tiled_free[(case, index)] = picture
        case_core = sorted(read_extrinsics_binary(run / "04_downstream" / case / "dataset/sparse/0/images.bin").values(), key=lambda x: x.name)
        by_name = {im.name: im for im in case_core}
        for index in [0, 45, 90, 135, 179]:
            image = by_name[core[index].name]
            camera = build_camera(intrinsics, qvec2rotmat(image.qvec), image.tvec, blank, image.name)
            tensor_to_image(render_image(camera, gaussians, pipeline)).save(case_output / f"capture_{index}.png")
        summary = {"case": case, "all_H_count": len(results), "rendered_H_count": sum(r["localized"] for r in results),
            "full_image": mean_metrics(results, "full_image"),
            "fixed_target_rectangle": mean_metrics(results, "fixed_target_rectangle"),
            "quality_eligible_full_image": mean_metrics([r for r in results if r["quality_eligible"]], "full_image"),
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "pose_source": "own_map_PnP_H_loc_only_not_independent_camera_GT"}
        summaries.append(summary)
        all_results[case] = results
        save_json(case_output / "RENDER_METRICS.json", {"status": "completed", "summary": summary, "rows": results})
        del gaussians
        torch.cuda.empty_cache()
    H_grid = [[tiled_H.get((label, name), Image.new("RGB", (812, 620), "#444444"))
               for label in ["reference", *cases]] for name in uniform_names]
    contact_sheet(output / "H_comparison.png", ["Reference", "Visual BA / 180", "Weak prior / 180", "Augmented / 212"],
                  [Path(name).stem for name in uniform_names], H_grid)
    free_grid = [[tiled_free[(case, index)] for case in cases] for index in range(3)]
    contact_sheet(output / "free_view_comparison.png", ["Visual BA / 180", "Weak prior / 180", "Augmented / 212"],
                  ["Lateral offset 0.75m", "Height offset 0.75m", "Unobserved elevated view"], free_grid)
    paired = []
    a_by_name = {r["image_name"]: r for r in all_results[cases[0]]}
    for case in cases[1:]:
        for row in all_results[case]:
            a = a_by_name[row["image_name"]]
            if "full_image" in row and "full_image" in a:
                paired.append({"case": case, "image_name": row["image_name"],
                               "full_PSNR_delta_dB": row["full_image"]["PSNR_dB"] - a["full_image"]["PSNR_dB"],
                               "full_SSIM_delta": row["full_image"]["SSIM_INRIA"] - a["full_image"]["SSIM_INRIA"]})
    save_json(root / "RENDER_COMPARISON.json", {"status": "completed", "cases": summaries,
        "native_render_directory": str(output), "attempt": cli.attempt,
        "paired_deltas_vs_A1": paired, "elapsed_seconds": time.monotonic() - start,
        "LPIPS": "not_evaluated_optional_metric", "selection_from_H": False,
        "interpretation": "development_map_localization_and_rendering_system_effect_not_absolute_pose_or_geometry_accuracy",
        "native_INRIA_projection": "FoV_centered_projection_inherited_from_frozen_reference_implementation",
        "PINHOLE_principal_point_offset_from_image_center_px": [intrinsics["params"][2] - intrinsics["width"] / 2,
                                                                intrinsics["params"][3] - intrinsics["height"] / 2],
        "principal_point_boundary": "same_centered_projection_for_all_cases_preserved_as_controlled_baseline_limitation",
        "free_views": [{"center_m": center.tolist(), "has_photographic_reference": False} for center in free_centers]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
