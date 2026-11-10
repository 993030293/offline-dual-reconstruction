"""Build a separate navigation preview while preserving trained model assets."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def data_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    receipt = json.loads((source / "DISPLAY_EXPORT_RECEIPT.json").read_text(encoding="utf-8"))
    names = [item["case"] for item in receipt["cases"]]
    assets = []
    for name in names:
        original = source / "cases" / name
        destination = output / "cases" / name
        destination.mkdir(parents=True)
        shutil.copytree(original / "vendor", destination / "vendor")
        metadata = json.loads((original / "display_manifest.json").read_text(encoding="utf-8"))
        next(item for item in receipt["cases"] if item["case"] == name).update(metadata)
        for filename in ("scene.ply", "box.ply", "camera_path.json", "display_manifest.json"):
            shutil.copy2(original / filename, destination / filename)
            if filename.endswith(".ply"):
                old_hash = data_digest(original / filename)
                new_hash = data_digest(destination / filename)
                if new_hash != old_hash:
                    raise ValueError("The copied model differs from its source")
                assets.append({"case": name, "file": filename, "source": str(original / filename),
                               "sha256": new_hash, "bytes": (destination / filename).stat().st_size})
        options = "".join('<option value="' + item["case"] + '"' +
                          (" selected" if item["case"] == name else "") + ">" +
                          html.escape(item["label"]) + "</option>" for item in receipt["cases"])
        template = (code / "viewer.html").read_text(encoding="utf-8")
        template = template.replace("__CASE_LABEL__", html.escape(metadata["label"]))
        template = template.replace("__CASE_OPTIONS__", options)
        (destination / "index.html").write_text(template, encoding="utf-8")
        shutil.copy2(code / "viewer.js", destination / "viewer.js")
    # Keep the prior result comparison available without changing its evidence.
    landing = (source / "index.html").read_text(encoding="utf-8")
    landing = landing.replace("可旋转、平移、缩放、切换箱体范围和播放采集路径", "可自由旋转、平移、缩放，使用 WASD / Q / E 飞行，切换无人机俯视、箱体范围和采集路径")
    landing = landing.replace("<h1>Box2 · 位姿先验与覆盖增补对照</h1>", "<h1>Box2 · 自由飞行三维预览</h1>")
    landing = landing.replace("<nav aria-label=", "<p>当前显示上一轮已训练的模型。新一轮初始选帧正在独立生成，尚未用于这些模型的训练。</p>\n  <nav aria-label=")
    (output / "index.html").write_text(landing, encoding="utf-8")
    for filename in ("H_comparison.png", "free_view_comparison.png"):
        shutil.copy2(source / filename, output / filename)
    receipt.update(status="assets_verified_browser_validation_pending", source_viewer=str(source),
                   navigation_version="free_flight_v2", model_retrained=False,
                   initial_selection_v2_used_for_displayed_models=False, assets=assets)
    (output / "DISPLAY_EXPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": names, "copied_model_count": len(assets)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source.resolve(strict=True), args.output.resolve())
