"""Package an isolated continuous/on-demand comparison using identical splats."""

import argparse
import json
from pathlib import Path
import shutil

from runtime import atomic_json, data_identity


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Viewer source anchor changed: " + old[:80])
    return text.replace(old, new)


def demand_source(text):
    text = 'import { createFrameScheduler } from "./render_schedule.js";\n' + text
    text = replace_once(text, 'window.box2Viewer = state;',
        'window.box2Viewer = state;\nconst renderSchedule = createFrameScheduler();\n'
        'const requestRender = renderSchedule.request;\nstate.renderPolicy = "demand";\n'
        'state.isRenderPending = renderSchedule.isPending;')
    text = replace_once(text, 'scene.add(new SparkRenderer({ renderer }));',
        'scene.add(new SparkRenderer({ renderer, onDirty: requestRender }));')
    text = replace_once(text, 'const controls = new OrbitControls(camera, canvas);',
        'const controls = new OrbitControls(camera, canvas);\n'
        '  controls.addEventListener("change", requestRender);')
    text = replace_once(text, 'function applyView(index, fraction = 0) {',
        'function applyView(index, fraction = 0) {\n    requestRender();')
    text = replace_once(text, 'addEventListener("resize", projection.resize);',
        'addEventListener("resize", () => { projection.resize(); requestRender(); });')
    text = replace_once(text, 'renderer.setAnimationLoop(time => {',
        'for (const type of ["click", "change", "input"]) document.addEventListener(type, requestRender);\n'
        '  for (const type of ["keydown", "keyup", "blur"]) addEventListener(type, requestRender);\n'
        '  for (const type of ["pointerdown", "pointerup", "pointercancel", "wheel"]) canvas.addEventListener(type, requestRender);\n'
        '  canvas.addEventListener("pointermove", event => { if (event.buttons) requestRender(); });\n'
        '  document.addEventListener("visibilitychange", requestRender);\n'
        '  renderSchedule.start(time => {')
    text = replace_once(text, 'renderer.render(scene, camera); state.frames++;',
        'renderer.render(scene, camera); state.frames++;\n'
        '    if (state.playing || (state.navigationMode === "fly" && keys.size)) requestRender();')
    text = replace_once(text, 'await Promise.all([fullModel.initialized, boxModel.initialized]);',
        'await Promise.all([fullModel.initialized, boxModel.initialized]);\n  requestRender();')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    text = (source / "viewer.js").read_text(encoding="utf-8")
    vendor = (source / "vendor/spark.module.js").read_text(encoding="utf-8")
    if "this.onDirty = options.onDirty" not in vendor or "this.onDirty?.()" not in vendor:
        raise ValueError("The installed Spark build has no verified asynchronous dirty callback")
    candidate = demand_source(text)
    args.output.mkdir(parents=True, exist_ok=False)
    for label in ["continuous", "demand"]:
        case = args.output / "cases" / label
        shutil.copytree(source, case)
        original = text.replace('window.box2Viewer = state;',
            'window.box2Viewer = state;\nstate.renderPolicy = "continuous";\nstate.isRenderPending = () => false;')
        (case / "viewer.js").write_text(candidate if label == "demand" else original, encoding="utf-8")
        if label == "demand":
            shutil.copyfile(Path(__file__).with_name("render_schedule.js"), case / "render_schedule.js")
        page = (case / "index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index('</select>', start)+len('</select>')
        options = '<select id="case-select">' + ''.join('<option value="'+name+'"'+
            (' selected' if name == label else '')+'>'+title+'</option>'
            for name, title in [("continuous", "连续渲染对照"), ("demand", "按需渲染候选")]) + '</select>'
        (case / "index.html").write_text(page[:start]+options+page[end:], encoding="utf-8")
    identities = {}
    for name in ["scene.ply", "box.ply", "camera_path.json", "display_manifest.json", "calibrated_camera.js",
                 "vendor/spark.module.js"]:
        hashes = [data_identity(args.output / "cases" / label / name)["sha256"] for label in ["continuous", "demand"]]
        if hashes[0] != hashes[1] or hashes[0] != data_identity(source / name)["sha256"]:
            raise AssertionError("The paired scene/calibration asset changed")
        identities[name] = hashes[0]
    atomic_json(args.output / "CONTRACT.json", {"status": "packaged_not_browser_verified", "source": str(source),
        "scene_asset_identity_equal": True, "assets": identities,
        "intervention": "on-demand renderer with Spark asynchronous dirty notifications and unchanged navigation",
        "acceptance": "3x5s stable idle windows >=95 percent fewer renderer calls; identical protected pixels and real controls; no stale frames",
        "source_identity": data_identity(Path(__file__)), "scheduler": data_identity(Path(__file__).with_name("render_schedule.js")),
        "counted": False})
    print(json.dumps({"status": "packaged_not_browser_verified", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
