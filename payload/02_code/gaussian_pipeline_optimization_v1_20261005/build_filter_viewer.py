"""Stage a model-bound classic filter comparison without modifying an accepted viewer."""

import argparse
import json
from pathlib import Path
import shutil

from build_packed_viewer import replace_once
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "training-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    acceptance = json.loads((args.source/"ACCEPTANCE.json").read_text())
    if acceptance["status"] != "accepted_stage_optimization" or acceptance["mechanism"] != "P14_LAZY_MODEL_LOADING":
        raise ValueError("Use the accepted lazy viewer as the preserved reference")
    manifest = json.loads((args.source/"ASSET_MANIFEST.json").read_text())
    config_path = args.training_run/"CONFIG.json"
    config = json.loads(config_path.read_text())
    state = json.loads((args.training_run/"STATE.json").read_text())
    if (state["status"] != "completed" or config["parameters"].get("antialiasing") is not False
            or config["intervention"]["projection"] != "full"):
        raise ValueError("Training does not provide an explicit completed classic full-K model")
    if config.get("optional_gsplat_identity") and config["optional_gsplat_identity"]["antialiasing"] is not False:
        raise ValueError("Training renderer and pipeline filter contracts disagree")
    ply = args.training_run/f'model/point_cloud/iteration_{state["iteration"]}/point_cloud.ply'
    if data_identity(ply)["sha256"] != manifest["scene"]["source_ply_sha256"]:
        raise ValueError("Training configuration belongs to a different full-scene model")
    source = args.source/"cases/lazy"
    base = (source/"viewer.js").read_text(encoding="utf-8")
    old = "  scene.add(new SparkRenderer({ renderer, onDirty: requestRender }));"
    replacement = '''  const filterContract = await loadJSON("./render_filter.json");
  const filterAssets = await loadJSON("../../ASSET_MANIFEST.json");
  const filterOptions = resolveRenderFilter(filterContract, filterAssets);
  state.renderFilter = { ...filterOptions, trainingConfigSha256: filterContract.trainingConfigSha256 };
  scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));'''
    corrected = 'import { resolveRenderFilter } from "./render_filter_contract.js";\n' + replace_once(base, old, replacement)
    args.output.mkdir(parents=True, exist_ok=False)
    filter_contract = {"schema": 1, "rasterizeMode": "classic", "eps2d": .3,
        "opacityCompensation": False, "antialiasing": False,
        "trainingConfigSha256": data_identity(config_path)["sha256"],
        "models": {label: manifest[label]["source_ply_sha256"] for label in ("scene", "box")}}
    identities = []
    for label, code in (("default", base), ("classic", corrected)):
        target = args.output/"cases"/label
        shutil.copytree(source, target)
        (target/"viewer.js").write_text(code, encoding="utf-8")
        for filename in ("render_filter_contract.js", "render_filter_probe.js"):
            shutil.copyfile(Path(__file__).with_name(filename), target/filename)
        atomic_json(target/"render_filter.json", filter_contract)
        page = (target/"index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index("</select>", start) + len("</select>")
        options = '<select id="case-select">' + ''.join(
            '<option value="'+name+'"'+(' selected' if name == label else '')+'>'+title+'</option>'
            for name, title in (("default", "现有显示对照"), ("classic", "模型滤波一致性"))) + '</select>'
        (target/"index.html").write_text(page[:start]+options+page[end:], encoding="utf-8")
        (target/"probe.html").write_text('''<!doctype html><html><head><meta charset="utf-8">
<link rel="icon" href="data:,">
<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/","@sparkjsdev/spark":"./vendor/spark.module.js"}}</script>
</head><body><script type="module" src="./render_filter_probe.js"></script></body></html>''', encoding="utf-8")
        for filename in ("scene.spk", "box.spk", "camera_path.json", "display_manifest.json", "calibrated_camera.js", "render_schedule.js"):
            original = data_identity(source/filename)
            if data_identity(target/filename)["sha256"] != original["sha256"]:
                raise ValueError("Staging changed an asset or protected navigation source")
            if label == "classic":
                identities.append(original)
    shutil.copyfile(args.source/"ASSET_MANIFEST.json", args.output/"ASSET_MANIFEST.json")
    atomic_json(args.output/"CONTRACT.json", {"status": "staged_numeric_and_browser_validation_pending",
        "mechanism": "P25_RENDER_FILTER_CONTRACT", "source": str(args.source),
        "training_config": data_identity(config_path), "training_model": data_identity(ply),
        "source_identity": data_identity(Path(__file__)), "filter": filter_contract,
        "identical_assets_and_navigation_sources": identities, "counted": False})
    print(json.dumps({"status": "staged_numeric_and_browser_validation_pending", "output": str(args.output)}))


if __name__ == "__main__":
    main()
