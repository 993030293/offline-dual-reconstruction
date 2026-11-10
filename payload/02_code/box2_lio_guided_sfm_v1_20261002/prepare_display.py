"""Build isolated three-case viewers from the preserved local Spark viewer template."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--repair-vendor", action="store_true")
parser.add_argument("--align-default-view", action="store_true")
cli = parser.parse_args()
sys.path.insert(0, str(cli.runtime / "source/gaussian-splatting"))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


def save_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    root = cli.run / "viewer"
    root.mkdir(exist_ok=False)
    roi = json.loads((cli.run / "03_coverage_augmented/SELECTION_MANIFEST.json").read_text())["ROI"]
    lo, hi = np.asarray(roi["minimum"]), np.asarray(roi["maximum"])
    labels = {"A1_visual": "纯视觉 BA · 180 帧", "B_w_sigma015": "弱位置先验 · 180 帧", "C_augmented_212": "覆盖增补 · 212 帧"}
    baseline_images = sorted(read_extrinsics_binary(cli.run / "04_downstream/A1_visual/dataset/sparse/0/images.bin").values(), key=lambda im: im.name)
    common_default = baseline_images[90].name
    template = cli.runtime / "viewer"
    text = (template / "index.html").read_text(encoding="utf-8")
    records = []
    for case, label in labels.items():
        destination = root / "cases" / case
        destination.mkdir(parents=True)
        shutil.copytree(template / "vendor", destination / "vendor")
        shutil.copy2(template / "viewer.js", destination / "viewer.js")
        choices = "".join(f'<option value="{key}"'+ (' selected' if key == case else '') + f'>{name}</option>' for key, name in labels.items())
        page = text.replace("<h1>绕箱子 · 三维展示</h1>", f"<h1>{label}</h1>")
        page = page.replace('<nav aria-label="展示设置">', '<nav aria-label="展示设置">\n<select id="case-select" aria-label="对照分支">' + choices + '</select>')
        page = page.replace('</body>', '<script>document.querySelector("#case-select").addEventListener("change", event => { location.href = "../" + event.target.value + "/index.html"; });</script>\n</body>')
        (destination / "index.html").write_text(page, encoding="utf-8")
        source = cli.run / "04_downstream" / case / "model/point_cloud/iteration_10000/point_cloud.ply"
        ply = PlyData.read(source)
        vertices = ply["vertex"].data
        xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
        if not np.isfinite(xyz).all():
            raise RuntimeError("Display Gaussian centers are not finite")
        selected = np.all(xyz >= lo, axis=1) & np.all(xyz <= hi, axis=1)
        shutil.copy2(source, destination / "scene.ply")
        PlyData([PlyElement.describe(vertices[selected].copy(), "vertex")], text=False).write(destination / "box.ply")
        sparse = cli.run / "04_downstream" / case / "dataset/sparse/0"
        cameras = read_intrinsics_binary(sparse / "cameras.bin")
        images = sorted(read_extrinsics_binary(sparse / "images.bin").values(), key=lambda im: im.name)
        path = []
        for image in images:
            camera = cameras[image.camera_id]
            rotation = qvec2rotmat(image.qvec).T
            path.append({"image_id": image.id, "image_name": image.name,
                "position": (-rotation @ image.tvec).tolist(), "forward": rotation[:, 2].tolist(),
                "up": (-rotation[:, 1]).tolist(), "rotation_c2w": rotation.tolist(),
                "width": camera.width, "height": camera.height,
                "fx": camera.params[0], "fy": camera.params[1], "cx": camera.params[2], "cy": camera.params[3]})
        save_json(destination / "camera_path.json", path)
        metadata = {"case": case, "label": label, "iteration": 10000, "image_count": len(images),
                    "default_view": next(i for i, image in enumerate(images) if image.name == common_default),
                    "default_image_name": common_default, "focus": roi["center"],
                    "full_gaussian_count": len(vertices), "box_gaussian_count": int(selected.sum()),
                    "roi_minimum": roi["minimum"], "roi_maximum": roi["maximum"],
                    "crop_policy": "same_fixed_development_ROI_Gaussian_center_display_crop_only",
                    "source_model": str(source), "training_cameras_fixed": True,
                    "claim_boundary": "development_display_not_independent_geometry_accuracy"}
        save_json(destination / "display_manifest.json", metadata)
        records.append(metadata)
    render = json.loads((cli.run / "05_evaluation/RENDER_COMPARISON.json").read_text())
    render_directory = Path(render["native_render_directory"])
    for name in ["H_comparison.png", "free_view_comparison.png"]:
        shutil.copy2(render_directory / name, root / name)
    shutil.copy2(Path(__file__).with_name("display_index.html"), root / "index.html")
    save_json(root / "DISPLAY_EXPORT_RECEIPT.json", {"status": "exported_pending_browser_validation", "cases": records,
        "source_template": str(template), "new_training_count": 3, "old_viewer_modified": False})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if cli.align_default_view:
        root = cli.run / "viewer/cases"
        baseline_path = json.loads((root / "A1_visual/camera_path.json").read_text())
        common = baseline_path[90]["image_name"]
        records = []
        for case in ["A1_visual", "B_w_sigma015", "C_augmented_212"]:
            folder = root / case
            path = json.loads((folder / "camera_path.json").read_text())
            metadata_path = folder / "display_manifest.json"
            metadata = json.loads(metadata_path.read_text())
            backup = folder / "display_manifest_initial_default.json"
            if backup.exists():
                raise FileExistsError(backup)
            shutil.copy2(metadata_path, backup)
            previous = metadata["default_view"]
            metadata.update(default_view=next(i for i, image in enumerate(path) if image["image_name"] == common), default_image_name=common)
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(metadata_path)
            records.append({"case": case, "old_default_view": previous, "common_default_view": metadata["default_view"]})
        save_json(cli.run / "viewer/DEFAULT_VIEW_ALIGNMENT_RECEIPT.json", {"status": "aligned_pending_C_visual_recheck",
            "common_capture_image": common, "cases": records,
            "model_weights_or_camera_paths_changed": False})
    elif cli.repair_vendor:
        repaired = []
        for case in ["A1_visual", "B_w_sigma015", "C_augmented_212"]:
            destination = cli.run / "viewer/cases" / case / "vendor"
            for source in (cli.runtime / "viewer/vendor").iterdir():
                if source.is_dir() and not (destination / source.name).exists():
                    shutil.copytree(source, destination / source.name)
                    repaired.append(str(destination / source.name))
        save_json(cli.run / "viewer/DISPLAY_ASSET_REPAIR_RECEIPT.json", {
            "status": "copied_pending_browser_revalidation", "root_cause": "nested_vendor_modules_omitted_from_flat_copy",
            "observed_404": "vendor/postprocessing/Pass.js", "copied_directories": repaired,
            "original_template_unchanged": True})
    else:
        main()
