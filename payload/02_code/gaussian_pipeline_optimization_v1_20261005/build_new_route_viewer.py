"""Stage both admitted-map models with the accepted calibrated lazy viewer."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData, PlyElement

from build_packed_viewer import replace_once
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair", "source-viewer", "camera-path", "roi", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pair = read(args.pair / "STATE.json")
    evaluation = read(args.pair / "evaluation/EVALUATION.json")
    if pair["status"] != "training_and_restore_passed_quality_pending" or not evaluation["development_metric_guard_passed"]:
        raise ValueError("The new-route pair has not passed its training, restore and all-view quality checks")
    roi = read(args.roi)
    path_receipt = read(args.camera_path / "RECEIPT.json")
    views = read(args.camera_path / "camera_path.json")
    if roi["status"] != "complete" or roi["roi_derivation"]["camera_count"] != 1016:
        raise ValueError("The display crop must be derived from the new admitted map")
    if path_receipt["status"] != "all_admitted_cameras_verified" or len(views) != 1016:
        raise ValueError("The new calibrated camera path was not verified")
    minimum = np.asarray(roi["roi_world_lio_m"]["minimum"])
    maximum = np.asarray(roi["roi_world_lio_m"]["maximum"])
    source = args.source_viewer
    code = (source / "viewer.js").read_text(encoding="utf-8")
    if not all(token in code for token in ("resolveRenderFilter", "createLazyModels", 'state.renderPolicy = "demand"', "createCameraProjection")):
        raise ValueError("The source viewer lacks an accepted combination mechanism")
    code = replace_once(code, 'location.href = "../" + event.target.value + "/index.html";',
        'location.href = "../../../" + event.target.value + "/cases/packed/index.html";')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    titles = {"reference": "新地图 · 基准模型", "combined": "新地图 · 优化组合"}
    for label, iterations in (("reference", 10000), ("combined", 12000)):
        run = args.pair / label
        config = read(run / "CONFIG.json")
        state = read(run / "STATE.json")
        manifest = read(run / "INPUT_MANIFEST.json")
        if (state["status"] != "completed" or state["iteration"] != iterations
                or manifest["input_version"] != "new_map_admission_v1" or manifest["image_count"] != 1016
                or config["parameters"]["antialiasing"] or config["intervention"]["projection"] != "full"):
            raise ValueError("Training and display input/filter contracts differ")
        model = run / f"model/point_cloud/iteration_{iterations}/point_cloud.ply"
        model_identity = data_identity(model)
        if model_identity["sha256"] != evaluation["results"][label]["model"]["sha256"]:
            raise ValueError("The displayed model differs from the evaluated model")
        vertices = PlyData.read(model)["vertex"].data
        if len(vertices) != state["gaussian_count"] or len([key for key in vertices.dtype.names if key.startswith("f_rest_")]) != 45:
            raise ValueError("The final model lost its population or full SH3 coefficients")
        if not all(np.isfinite(vertices[key]).all() for key in vertices.dtype.names):
            raise ValueError("Nonfinite Gaussian attributes")
        xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
        mask = ((xyz >= minimum) & (xyz <= maximum)).all(axis=1)
        if not mask.any() or mask.all():
            raise ValueError("The new display crop is empty or covers the entire scene")
        root = args.output / label
        target = root / "cases/packed"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("scene.spk", "box.spk", "scene.ply", "box.ply"))
        shutil.copyfile(model, target / "scene.ply")
        PlyData([PlyElement.describe(vertices[mask].copy(), "vertex")], text=False).write(target / "box.ply")
        shutil.copyfile(args.camera_path / "camera_path.json", target / "camera_path.json")
        (target / "viewer.js").write_text(code, encoding="utf-8")
        metadata = {"case": label, "label": titles[label], "iteration": iterations, "image_count": 1016,
            "default_view": path_receipt["default_view"], "default_image_name": views[path_receipt["default_view"]]["image_name"],
            "focus": roi["roi_derivation"]["focus_world_lio_m"], "roi_minimum": minimum.tolist(),
            "roi_maximum": maximum.tolist(), "full_gaussian_count": len(vertices), "box_gaussian_count": int(mask.sum()),
            "crop_policy": "new_admitted_map_geometric_ROI_Gaussian_center_display_crop_only",
            "ROI_source": data_identity(args.roi), "camera_path_source": data_identity(args.camera_path / "RECEIPT.json"),
            "source_model": str(model), "training_cameras_fixed": True, "input_version": "new_map_admission_v1",
            "claim_boundary": "development_display_not_independent_geometry_accuracy",
            "display_projection": "full_calibrated_capture_and_independent_free_camera"}
        atomic_json(target / "display_manifest.json", metadata)
        assets = {}
        vendor = data_identity(target / "vendor/spark.module.js")
        for model_label, count in (("scene", len(vertices)), ("box", int(mask.sum()))):
            identity = data_identity(target / (model_label + ".ply"))
            assets[model_label] = {"spark_sha256": vendor["sha256"], "source_ply_sha256": identity["sha256"],
                                   "source_bytes": identity["bytes"], "numSplats": count}
        atomic_json(root / "ASSET_SOURCES.json", assets)
        atomic_json(target / "render_filter.json", {"schema": 1, "rasterizeMode": "classic", "eps2d": .3,
            "opacityCompensation": False, "antialiasing": False,
            "trainingConfigSha256": data_identity(run / "CONFIG.json")["sha256"],
            "models": {key: value["source_ply_sha256"] for key, value in assets.items()}})
        page = (target / "index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index("</select>", start) + len("</select>")
        selector = '<select id="case-select">' + "".join('<option value="' + key + '"'
            + (' selected' if key == label else '') + '>' + title + '</option>' for key, title in titles.items()) + '</select>'
        page = page[:start] + selector + page[end:]
        page = page.replace("完整内参修正 · 212 帧", titles[label] + " · 1,016 帧")
        page = page.replace('max="179"', 'max="1015"')
        (target / "index.html").write_text(page, encoding="utf-8")
        (target / "export.html").write_text('''<!doctype html><html><head><meta charset="utf-8">
<link rel="icon" href="data:,">
<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/","@sparkjsdev/spark":"./vendor/spark.module.js"}}</script>
</head><body>Local packed asset export</body></html>''', encoding="utf-8")
        records.append({"label": label, "training_model": model_identity, "assets": assets,
                        "full_count": len(vertices), "box_count": int(mask.sum())})
    atomic_json(args.output / "CONTRACT.json", {"status": "staged_native_packing_and_browser_validation_pending",
        "mechanism": "FINAL_NEW_ROUTE_COMBINATION", "source": data_identity(Path(__file__)),
        "assets": records, "camera_count": 1016, "old_map_ROI_or_camera_path_used": False,
        "new_optimization_count": 0, "independent_geometry_accuracy_claim": False})
    print(json.dumps({"status": "staged_native_packing_and_browser_validation_pending", "models": [
        {key: row[key] for key in ("label", "full_count", "box_count")} for row in records]}))


if __name__ == "__main__":
    main()
