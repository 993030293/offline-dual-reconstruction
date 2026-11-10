"""Package isolated control/corrected viewers without modifying an existing service."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData, PlyElement

from runtime import atomic_json, data_identity


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("The preserved viewer source no longer matches the integration contract")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    original = project/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/viewer/cases/C_augmented_212"
    output = args.pair/"viewer"
    output.mkdir(exist_ok=False)
    text = (original/"viewer.js").read_text(encoding="utf-8")
    text = 'import { createCameraProjection } from "./calibrated_camera.js";\n'+text
    text = replace_once(text, 'ui("viewport").appendChild(canvas);',
                        'ui("viewport").appendChild(canvas);\n  const projection = createCameraProjection(camera, renderer, state);')
    text = replace_once(text, 'function freeView() {\n    stopPlayback();',
                        'function freeView() {\n    stopPlayback();\n    projection.free();')
    text = replace_once(text, 'function setMode(mode) {\n    stopPlayback();',
                        'function setMode(mode) {\n    stopPlayback();\n    projection.free();')
    text = replace_once(text, 'camera.fov = a.fov;\n    camera.updateProjectionMatrix();',
                        'camera.fov = a.fov;\n    projection.capture(views[index]);')
    text = replace_once(text, 'camera.fov = 50;\n    camera.updateProjectionMatrix();',
                        'camera.fov = 50;\n    projection.free();')
    text = replace_once(text, 'addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });',
                        'addEventListener("resize", projection.resize);')
    assets = []
    for label in ["legacy", "full"]:
        folder = output/"cases"/label
        folder.mkdir(parents=True)
        shutil.copytree(original/"vendor", folder/"vendor")
        metadata = json.loads((original/"display_manifest.json").read_text(encoding="utf-8"))
        state = json.loads((args.pair/label/"STATE.json").read_text())
        if state["status"] != "completed":
            raise ValueError("Do not publish an incomplete model")
        model = args.pair/label/"model/point_cloud/iteration_10000/point_cloud.ply"
        shutil.copyfile(model, folder/"scene.ply")
        vertices = PlyData.read(model)["vertex"].data
        xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
        mask = ((xyz >= metadata["roi_minimum"]) & (xyz <= metadata["roi_maximum"])).all(axis=1)
        PlyData([PlyElement.describe(vertices[mask].copy(), "vertex")], text=False).write(folder/"box.ply")
        metadata.update(case=label, label="完整内参修正" if label == "full" else "旧投影对照",
                        source_model=str(model), full_gaussian_count=len(vertices), box_gaussian_count=int(mask.sum()),
                        display_projection="full_calibrated_capture_and_independent_free_camera")
        atomic_json(folder/"display_manifest.json", metadata)
        shutil.copyfile(original/"camera_path.json", folder/"camera_path.json")
        shutil.copyfile(Path(__file__).with_name("calibrated_camera.js"), folder/"calibrated_camera.js")
        page = (original/"index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index('</select>', start)+len('</select>')
        options = '<select id="case-select">'+''.join('<option value="'+name+'"'+
            (' selected' if name == label else '')+'>'+title+'</option>'
            for name, title in [("legacy", "旧投影训练 · 同一完整内参预览"), ("full", "完整内参训练 · 同一完整内参预览")])+'</select>'
        page = page[:start]+options+page[end:]
        page = page.replace("覆盖增补 · 212 帧", metadata["label"]+" · 212 帧")
        (folder/"index.html").write_text(page, encoding="utf-8")
        (folder/"viewer.js").write_text(text, encoding="utf-8")
        assets.append({"case": label, "full": data_identity(folder/"scene.ply"),
                       "box": data_identity(folder/"box.ply"), "full_count": len(vertices), "box_count": int(mask.sum())})
    atomic_json(output/"EXPORT_RECEIPT.json", {"status": "packaged_browser_verification_pending",
        "assets": assets, "raw_world_coordinates_preserved": True, "capture_intrinsics_preserved": True,
        "existing_8195_service_modified": False, "ROI_role": "same_historical_display_crop_only",
        "geometry_accuracy_claim": False})
    print(json.dumps({"output": str(output), "status": "browser_verification_pending"}))


if __name__ == "__main__":
    main()
