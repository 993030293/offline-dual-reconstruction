"""Stage identical packed assets for eager versus selected-model loading."""

import argparse
import json
from pathlib import Path
import shutil

from build_packed_viewer import replace_once
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acceptance = json.loads((args.source/"ACCEPTANCE.json").read_text())
    if acceptance["status"] != "accepted_stage_optimization" or acceptance["mechanism"] != "P10_PACKED_VIEWER_ASSET":
        raise ValueError("Start from the actually validated packed viewer")
    source = args.source/"cases/packed"
    base = (source/"viewer.js").read_text(encoding="utf-8")
    base = replace_once(base, 'const modelLoadStarted = performance.now();',
        'const modelLoadStarted = performance.now();\n'
        '  const initialMode = new URLSearchParams(location.search).get("mode") === "box" ? "box" : "scene";\n'
        '  state.firstRequestedMode = initialMode;\n'
        '  ui("scene-mode").value = initialMode;')
    base = replace_once(base, 'renderer.render(scene, camera); state.frames++;',
        'renderer.render(scene, camera); state.frames++;\n'
        '    if (state.ready && state.firstReadyRenderMs == null) {\n'
        '      state.firstReadyRenderMs = performance.now() - modelLoadStarted;\n'
        '      state.navigationToFirstReadyRenderMs = performance.now();\n'
        '    }')
    eager = replace_once(base, 'boxModel.visible = false;',
        'fullModel.visible = initialMode === "scene";\n'
        '  boxModel.visible = initialMode === "box";\n'
        '  state.sceneMode = initialMode;\n'
        '  state.loadedModels = ["box", "scene"];')
    lazy = 'import { createLazyModels } from "./lazy_models.js";\n' + base
    old = '''  const [fullPacked, boxPacked] = await Promise.all([loadPackedAsset("./scene.spk", assets.scene), loadPackedAsset("./box.spk", assets.box)]);
  const fullModel = new SplatMesh({ splats: fullPacked, lod: false });
  const boxModel = new SplatMesh({ splats: boxPacked, lod: false });
  scene.add(fullModel, boxModel);
  boxModel.visible = false;
  ui("scene-mode").addEventListener("change", event => {
    state.sceneMode = event.target.value;
    fullModel.visible = event.target.value === "scene"; boxModel.visible = event.target.value === "box";
  });'''
    new = '''  state.loadFailures = [];
  state.modelLoadingPolicy = "selected_only";
  const modelRegistry = createLazyModels({ scene, state, requestRender,
    load: async mode => new SplatMesh({ splats: await loadPackedAsset("./" + mode + ".spk", assets[mode]), lod: false }) });
  const retry = document.createElement("button");
  retry.textContent = "重试加载";
  retry.hidden = true;
  ui("scene-mode").insertAdjacentElement("afterend", retry);
  async function selectModel(mode) {
    ui("status").textContent = (mode === "box" ? "箱体范围" : "完整场景") + "加载中";
    retry.hidden = true;
    try {
      const result = await modelRegistry.select(mode);
      if (!result.applied) return;
      ui("error").style.display = "none";
      ui("error").textContent = "";
      ui("status").textContent = "模型已加载 · 可自由浏览或切换飞行";
      if (state.modelInitializationMs == null) state.modelInitializationMs = performance.now() - modelLoadStarted;
    } catch (error) {
      state.loadFailures.push({ mode, message: String(error), at: performance.now() });
      if (!error.isCurrentRequest) return;
      ui("status").textContent = "请求的模型加载失败，可重试";
      ui("error").style.display = "block";
      ui("error").textContent = String(error);
      retry.hidden = false;
    }
  }
  retry.addEventListener("click", () => selectModel(ui("scene-mode").value));
  ui("scene-mode").addEventListener("change", event => { selectModel(event.target.value); });'''
    lazy = replace_once(lazy, old, new)
    end = '''  await Promise.all([fullModel.initialized, boxModel.initialized]);
  state.modelInitializationMs = performance.now() - modelLoadStarted;
  requestRender();
  state.ready = true; state.fullGaussianCount = fullModel.numSplats; state.boxGaussianCount = boxModel.numSplats;
  ui("status").textContent = "模型已加载 · 可自由浏览或切换飞行";'''
    lazy = replace_once(lazy, end, '  await selectModel(initialMode);')
    args.output.mkdir(parents=True, exist_ok=False)
    for label, code in [("eager", eager), ("lazy", lazy)]:
        target = args.output/"cases"/label
        shutil.copytree(source, target)
        (target/"viewer.js").write_text(code, encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("lazy_models.js"), target/"lazy_models.js")
        page = (target/"index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        stop = page.index('</select>', start)+len('</select>')
        select = '<select id="case-select">'+''.join('<option value="'+name+'"'+(' selected' if name == label else '')+'>'+title+'</option>'
            for name, title in [("eager", "同时加载对照"), ("lazy", "按所选范围加载")])+'</select>'
        (target/"index.html").write_text(page[:start]+select+page[stop:], encoding="utf-8")
    shutil.copyfile(args.source/"ASSET_MANIFEST.json", args.output/"ASSET_MANIFEST.json")
    identities = []
    for name in ("scene.spk", "box.spk", "display_manifest.json", "camera_path.json", "calibrated_camera.js", "render_schedule.js"):
        source_id = data_identity(source/name)
        if any(data_identity(args.output/"cases"/label/name)["sha256"] != source_id["sha256"] for label in ("eager", "lazy")):
            raise ValueError("The lazy comparison changed its asset, camera, or renderer")
        identities.append(source_id)
    atomic_json(args.output/"CONTRACT.json", {"status": "staged_browser_validation_pending", "source": str(args.source),
        "identical_assets_and_navigation_sources": identities, "initial_scene_default_unchanged": True,
        "box_first_query": "?mode=box", "first_view_performance_target": "box mode >=10% median reduction over3empty-cachecontexts",
        "required": ["both_modes_no_geometry_change", "six_exact_rendered_views", "latest_switch_wins", "no_duplicate_requests",
                     "failed_request_retains_visible_model_and_retries", "flight_and_path_regression"],
        "counted": False, "source_identity": data_identity(Path(__file__)),
        "lazy_loader_identity": data_identity(Path(__file__).with_name("lazy_models.js"))})
    print(json.dumps({"status": "staged_browser_validation_pending", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
