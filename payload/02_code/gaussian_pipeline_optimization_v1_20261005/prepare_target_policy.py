"""Inspect every real training-camera volume hint before authorizing weighted training."""

import argparse
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
from PIL import Image, ImageDraw

from full_intrinsics import camera_loader_adapter
from runtime import activate_reference, atomic_json, data_identity
from target_weight_adapter import TargetVolumeLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "baseline-run", "roi-source", "numeric", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    numeric = json.loads(args.numeric.read_text())
    if numeric["status"] != "passed" or not all(numeric["checks"].values()):
        raise ValueError("Independent geometry and loss oracles did not pass")
    validated = {Path(row["path"]).name: row["sha256"] for row in numeric["sources"]}
    if any(data_identity(Path(__file__).parent/name)["sha256"] != validated[name]
           for name in ("target_weight_adapter.py", "full_intrinsics.py")):
        raise ValueError("The target adapter changed after its numeric validation")
    roi = json.loads(args.roi_source.read_text())
    baseline = json.loads((args.baseline_run/"INPUT_MANIFEST.json").read_text())
    if roi["image_count"] != 212 or baseline["image_count"] != 212 or baseline["input_version"] != "C212_frozen":
        raise ValueError("The historical ROI cannot silently transfer to a new route or frame set")
    images = sorted(p for p in (args.dataset/"images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = [args.dataset/"sparse/0"/name for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")]+images
    identity = [data_identity(path) for path in paths]
    if [row["sha256"] for row in identity] != [row["sha256"] for row in baseline["files"]]:
        raise ValueError("The current inputs differ from the frozen C212 training input")
    atomic_json(args.output/"CONTRACT.json", {"maximum_seconds": 300, "RAM_budget_bytes": 32*1024**3,
        "input_identity": identity, "roi_source": data_identity(args.roi_source), "numeric": data_identity(args.numeric),
        "purpose": "inspect conservative object-volume hints; no training or pixel-label accuracy claim", "counted": False})
    runtime = activate_reference(args.runtime)
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel

    native_parser = argparse.ArgumentParser()
    groups = [ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)]
    native = native_parser.parse_args([])
    for key, value in json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"].items():
        if not hasattr(native, key):
            raise ValueError("Unknown baseline parameter")
        setattr(native, key, value)
    native.source_path = str(args.dataset)
    native.model_path = str(args.output/"scene")
    Path(native.model_path).mkdir()
    model, _, _ = [group.extract(native) for group in groups]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    gaussian = GaussianModel(model.sh_degree)
    with camera_loader_adapter(args.dataset) as manifest:
        scene = Scene(model, gaussian, shuffle=False)
    cameras = scene.getTrainCameras()
    reference_manifest = json.loads((args.baseline_run/"CAMERA_MANIFEST.json").read_text())["cameras"]
    if (len(cameras) != 212 or {row["name"]: row["intrinsics"] for row in manifest}
            != {row["name"]: row["intrinsics"] for row in reference_manifest}):
        raise ValueError("Loaded cameras differ from the actually trained full-K baseline")
    target = TargetVolumeLoss(cameras, roi["roi_minimum"], roi["roi_maximum"])
    coverage = target.finish(0)
    selected = np.linspace(0, 211, 6, dtype=int)
    sheet = Image.new("RGB", (812, 6*338), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row_index, index in enumerate(selected):
        camera = cameras[index]
        record = target.records[id(camera)]
        pixels = camera.original_image.permute(1, 2, 0).numpy()
        mask = record["mask"].numpy().astype(bool)
        overlay = pixels.copy()
        overlay[mask] = .65*overlay[mask]+.35*np.array([0., 1., 1.])
        for column, data in enumerate((pixels, overlay)):
            picture = Image.fromarray(np.clip(np.rint(data*255), 0, 255).astype(np.uint8)).resize((406, 310))
            sheet.paste(picture, (column*406, row_index*338))
        draw.text((8, row_index*338+313), camera.image_name, fill="white")
        draw.text((414, row_index*338+313), f"volume hint {record['pixels']/record['total_pixels']:.3%}; weight2 vs1", fill="white")
    sheet.save(args.output/"SIX_TARGET_HINTS.jpg", quality=95)
    fractions = [row["pixels"]/row["total_pixels"] for row in coverage["rows"]]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if time.monotonic()-started > 300 or rss > 32*1024**3:
        raise RuntimeError("The registered actual-camera diagnostic exceeded its budget")
    atomic_json(args.output/"CAMERA_HINTS.json", coverage)
    policy = {"status": "prepared_actual_hint_visual_review_pending", "input_version": "C212_frozen",
        "input_sha256": [row["sha256"] for row in identity], "image_count": 212,
        "roi_minimum": roi["roi_minimum"], "roi_maximum": roi["roi_maximum"],
        "roi_source": data_identity(args.roi_source), "foreground_weight": 2, "background_weight": 1,
        "role": "conservative_world_volume_ray_intersection_weak_hint_not_pixel_GT_or_instance_segmentation",
        "near_camera_depth_m": .2, "ssim": "unchanged_full_image_reference", "geometry_numeric": data_identity(args.numeric),
        "adapter": data_identity(Path(__file__).with_name("target_weight_adapter.py")), "camera_hints": data_identity(args.output/"CAMERA_HINTS.json"),
        "old_ROI_allowed_for_raw_initial_selection": False, "transfer_to_new_route_requires_new_ROI": True}
    atomic_json(args.output/"TARGET_POLICY.json", policy)
    receipt = {"status": "actual_camera_geometry_prepared_visual_review_pending", "runtime": runtime,
        "cameras": len(cameras), "coverage_min_median_max": [min(fractions), float(np.median(fractions)), max(fractions)],
        "zero_hint_frames": sum(value == 0 for value in fractions), "selected_visual_indices": selected.tolist(),
        "all_background_weights_positive": True, "source": data_identity(Path(__file__)),
        "preparation_seconds": coverage["preparation_seconds"], "elapsed_seconds": time.monotonic()-started,
        "maximum_rss_bytes": rss, "counted": False, "training_launched": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
