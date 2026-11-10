"""Export a completed backend pair through the accepted calibrated demand viewer."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData, PlyElement

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--source-viewer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = json.loads((args.pair / "PAIR_STATE.json").read_text())
    evaluation = json.loads((args.pair / "evaluation/EVALUATION.json").read_text())
    if pair["status"] != "training_completed_pending_evaluation" or not evaluation["development_metric_guard_passed"]:
        raise ValueError("Require both completed training runs and the frozen quality guard")
    if pair["candidate"] not in ("gsplat", "P13_GSPLAT_STAGE_FUSION", "P17_TARGET_DENSITY"):
        raise ValueError("This bounded export applies only to the declared full-K gsplat comparisons")
    source_js = (args.source_viewer / "viewer.js").read_text(encoding="utf-8")
    if 'state.renderPolicy = "demand"' not in source_js or 'createCameraProjection' not in source_js:
        raise ValueError("Require the previously accepted full-K demand viewer")
    metadata = json.loads((args.source_viewer / "display_manifest.json").read_text())
    if pair["candidate"] == "gsplat":
        budgets = {"reference": pair["control_iterations"], "gsplat": pair["candidate_iterations"]}
        titles = {"reference": f"原实现 {budgets['reference']} 步", "gsplat": f"gsplat {budgets['gsplat']} 步"}
    elif pair["candidate"] == "P13_GSPLAT_STAGE_FUSION":
        if pair["iterations_both"] != 12000:
            raise ValueError("The fusion display requires the declared equal12k budget")
        budgets = {"reference": 12000, "fused_step": 12000}
        titles = {"reference": "gsplat 基准 12000 步", "fused_step": "融合计算 12000 步"}
    else:
        if (pair["iterations"] != 12000 or not pair["resource_ratio_guard_passed"]
                or evaluation["candidate_minus_control"]["target"]["mean_PSNR_dB"] < .15):
            raise ValueError("Target-density display requires the registered first-pair quality and resource gates")
        budgets = {"reference": 12000, "target_density": 12000}
        titles = {"reference": "gsplat 基准 12000 步", "target_density": "目标增密候选 12000 步"}
    source_inputs = None
    prepared = []
    for label in titles:
        run = args.pair / label
        state = json.loads((run / "STATE.json").read_text())
        iterations = budgets[label]
        if state["status"] != "completed" or state["iteration"] != iterations or state["projection"] != "full":
            raise ValueError("A model is incomplete or differs from the declared camera/training contract")
        inputs = json.loads((run / "INPUT_MANIFEST.json").read_text())
        identities = [(row["path"], row["bytes"], row["sha256"]) for row in inputs["files"]]
        if source_inputs is None:
            source_inputs = identities
        elif source_inputs != identities:
            raise ValueError("The exported pair used different training inputs")
        if inputs["image_count"] != 212 or inputs["input_version"] != "C212_frozen":
            raise ValueError("This camera path belongs only to the frozen 212-frame input")
        model = run / f"model/point_cloud/iteration_{iterations}/point_cloud.ply"
        if data_identity(model)["sha256"] != evaluation["results"][label]["model"]["sha256"]:
            raise ValueError("The exported model differs from the actually evaluated model")
        vertices = PlyData.read(model)["vertex"].data
        names = vertices.dtype.names
        if len([name for name in names if name.startswith("f_rest_")]) != 45:
            raise ValueError("The viewer comparison requires all SH3 coefficients")
        if not all(np.isfinite(vertices[name]).all() for name in names):
            raise ValueError("The training PLY contains nonfinite values")
        xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
        mask = ((xyz >= metadata["roi_minimum"]) & (xyz <= metadata["roi_maximum"])).all(axis=1)
        if not mask.any() or len(vertices) != state["gaussian_count"]:
            raise ValueError("The model population or unchanged target crop is invalid")
        prepared.append((label, model, vertices, mask, iterations))
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for label, model, vertices, mask, iterations in prepared:
        folder = args.output / "cases" / label
        shutil.copytree(args.source_viewer, folder, ignore=shutil.ignore_patterns("scene.ply", "box.ply"))
        shutil.copyfile(model, folder / "scene.ply")
        PlyData([PlyElement.describe(vertices[mask].copy(), "vertex")], text=False).write(folder / "box.ply")
        derived = dict(metadata, case=label, label=titles[label], source_model=str(model),
                       full_gaussian_count=len(vertices), box_gaussian_count=int(mask.sum()))
        atomic_json(folder / "display_manifest.json", derived)
        page = (folder / "index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index('</select>', start) + len('</select>')
        options = '<select id="case-select">' + ''.join(
            '<option value="' + key + '"' + (' selected' if key == label else '') + '>' + title + '</option>'
            for key, title in titles.items()) + '</select>'
        page = page[:start] + options + page[end:]
        page = page.replace("完整内参修正 · 212 帧", titles[label] + " · 212 帧")
        (folder / "index.html").write_text(page, encoding="utf-8")
        receipts.append({"case": label, "iterations": iterations, "full": data_identity(folder / "scene.ply"),
                         "source": data_identity(model), "box": data_identity(folder / "box.ply"),
                         "full_count": len(vertices), "box_count": int(mask.sum())})
    atomic_json(args.output / "EXPORT_RECEIPT.json", {
        "status": "exported_browser_verification_pending", "assets": receipts,
        "source_viewer": str(args.source_viewer), "seed": pair["seed"], "projection": "full",
        "world_transform_applied": False, "all_SH_preserved": True,
        "ROI_role": ("historical_display_crop_also_used_by_target_density_candidate"
                     if pair["candidate"] == "P17_TARGET_DENSITY" else "unchanged_historical_display_crop_only"),
        "counted": False})
    print(json.dumps({"status": "exported_browser_verification_pending", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
