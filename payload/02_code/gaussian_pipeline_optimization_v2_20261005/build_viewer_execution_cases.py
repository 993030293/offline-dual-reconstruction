"""Package independent exact-integrity, navigation and build-storage viewer comparisons."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from build_packed_viewer import replace_once


def integrity_source(source, parallel=False):
    source = replace_once(source, "  const arrays = {};\n  let offset = 0;",
        "  const arrays = {};\n  const integrityStarted = performance.now();\n  const integrityParts = [];\n  let offset = 0;")
    if parallel:
        source = replace_once(source,
            '    if (await digestBytes(part) !== row.sha256) throw new Error("Packed buffer checksum mismatch");',
            "    integrityParts.push(part);")
    anchor = "  const packed = new PackedSplats({ packedArray: arrays.packedArray, numSplats: metadata.numSplats,"
    verification = '''  const hashes = await Promise.all(integrityParts.map(digestBytes));
  for (let i = 0; i < hashes.length; i++) {
    if (hashes[i] !== metadata.buffers[i].sha256) throw new Error("Packed buffer checksum mismatch");
  }
''' if parallel else ""
    trace = '''  (globalThis.assetIntegrityTrace ??= []).push({
    payloadBytes: metadata.payloadBytes, buffers: metadata.buffers.length,
    integrity_ms: performance.now() - integrityStarted, all_native_checks_retained: true });
'''
    return replace_once(source, anchor, verification+trace+anchor)


def cache_source(source):
    source = 'import { readAssetArtifact, assetCacheTrace } from "./persistent_asset_cache.js";\n'+source
    source = replace_once(source, '''  const response = await fetch(url);
  if (!response.ok) throw new Error("Packed asset request failed: " + response.status);
  const buffer = await response.arrayBuffer();''',
        '''  const artifact = await readAssetArtifact(url, expected.asset_sha256);
  const buffer = artifact.buffer;''')
    source = replace_once(source, "  return (await decodePackedAsset(buffer, expected)).packed;",
        "  const result = await decodePackedAsset(buffer, expected);\n  await artifact.commitVerified();\n  globalThis.assetCacheTrace = assetCacheTrace;\n  return result.packed;")
    source = replace_once(source, '''  const response = await fetch(url);
  if (!response.ok || await digestBytes(await response.arrayBuffer()) !== expectedHash) {''',
        '''  const artifact = await readAssetArtifact(url, expectedHash);
  if (await digestBytes(artifact.buffer) !== expectedHash) {''')
    source = replace_once(source, '''    throw new Error("The served Spark build differs from the packed asset contract");
  }
}''', '''    throw new Error("The served Spark build differs from the packed asset contract");
  }
  await artifact.commitVerified();
}
''')
    return source


def motion_source(source):
    source = 'import { createMotionResolution } from "./motion_resolution.js";\n'+source
    source = replace_once(source, "  const projection = createCameraProjection(camera, renderer, state);",
        "  const projection = createCameraProjection(camera, renderer, state);\n  const motionResolution = createMotionResolution(renderer, projection, state);")
    source = replace_once(source, "    if (state.playing) {\n      const phase =",
        "    let orbitChanged = false;\n    if (state.playing) {\n      const phase =")
    source = replace_once(source,
        '    } else if (state.navigationMode === "orbit") {\n      controls.update();',
        '    } else if (state.navigationMode === "orbit") {\n      orbitChanged = controls.update();')
    source = replace_once(source, "    renderer.render(scene, camera); state.frames++;",
        "    motionResolution.beforeFrame(state.playing || keys.size > 0 || drag !== null || orbitChanged);\n    renderer.render(scene, camera); state.frames++;")
    source = replace_once(source, " : \"左键旋转 · 右键平移 · 滚轮缩放 · 可切换自由飞行\";",
        " : \"左键旋转 · 右键平移 · 滚轮缩放 · 可切换自由飞行\";\n    ui(\"help\").textContent += \" · 移动时优先流畅，停下恢复清晰\";")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source/"combined/cases/packed"
    original = (source/"viewer.js").read_text(encoding="utf-8")
    packed = (source/"packed_splat_asset.js").read_text(encoding="utf-8")
    manifest = json.loads((args.source/"combined/ASSET_MANIFEST.json").read_text())
    labels = {"baseline": "原生预览对照", "reference": "基准模型预览", "checksum": "并行完整性校验", "motion": "移动时优先流畅", "cache": "重复加载产物缓存",
        "shared": "共享渲染组件", "shared-reference": "共享渲染组件基准模型"}
    shared = args.output/"shared-build/vendor"
    shutil.copytree(source/"vendor", shared)
    asset_identities = {}
    for label, title in labels.items():
        original_case = args.source/("reference" if label in ("reference", "shared-reference") else "combined")
        current_source = original_case/"cases/packed"
        root, target = args.output/label, args.output/label/"cases/packed"
        ignore = ["scene.ply", "box.ply"]+( ["vendor"] if label.startswith("shared") else [] )
        shutil.copytree(current_source, target, ignore=shutil.ignore_patterns(*ignore))
        shutil.copyfile(original_case/"ASSET_MANIFEST.json", root/"ASSET_MANIFEST.json")
        code = (current_source/"viewer.js").read_text(encoding="utf-8")
        integrity = integrity_source(packed, label == "checksum")
        if label == "cache":
            integrity = cache_source(integrity)
            shutil.copyfile(Path(__file__).with_name("persistent_asset_cache.js"), target/"persistent_asset_cache.js")
        if label == "motion":
            code = motion_source(code)
            shutil.copyfile(Path(__file__).with_name("motion_resolution.js"), target/"motion_resolution.js")
        if label.startswith("shared"):
            code = code.replace('"./vendor/OrbitControls.js"', '"../../../shared-build/vendor/OrbitControls.js"')
            code = code.replace('"./vendor/spark.module.js"', '"../../../shared-build/vendor/spark.module.js"')
        (target/"viewer.js").write_text(code, encoding="utf-8")
        (target/"packed_splat_asset.js").write_text(integrity, encoding="utf-8")
        page = (target/"index.html").read_text(encoding="utf-8")
        start, end = page.index('<select id="case-select">'), page.index("</select>", page.index('<select id="case-select">'))+len("</select>")
        selector = '<select id="case-select">'+"".join('<option value="'+name+'"'+(' selected' if name == label else '')+'>'+text+'</option>' for name, text in labels.items())+'</select>'
        page = page[:start]+selector+page[end:]
        if label.startswith("shared"):
            page = page.replace('"./vendor/', '"../../../shared-build/vendor/')
        (target/"index.html").write_text(page, encoding="utf-8")
        for name in ("scene.spk", "box.spk", "camera_path.json", "display_manifest.json", "render_filter.json"):
            identity = data_identity(target/name)
            if identity["sha256"] != data_identity(current_source/name)["sha256"]:
                raise AssertionError("An isolated viewer case changed its original model or calibration")
            asset_identities[label+"/"+name] = identity
    build_files = [path for path in shared.rglob("*") if path.is_file()]
    build_bytes = sum(path.stat().st_size for path in build_files)
    if any(data_identity(path)["sha256"] != data_identity(source/"vendor"/path.relative_to(shared))["sha256"] for path in build_files):
        raise AssertionError("The shared native renderer build changed a dependency")
    protocol = {"status": "packaged_actual_browser_validation_pending", "source": str(args.source), "cases": labels,
        "assets": asset_identities, "native_asset_and_calibration_bytes_unchanged": True,
        "Q68_parallel_integrity": "three actual full scene and box decodes; median integrity phase at least10pct lower; total load at most1.1; all tamper and pixel guards",
        "Q69_motion_resolution": "optional motion-only75pct linear resolution; framebuffer pixel workload at least10pct lower; all24 static and free-flight pixels exact after restoration; three real flight intervals median frame rate at least80pct of native baseline; real controls protected",
        "Q71_cache": "same native HTTP behavior; cold and three warm navigations including reads/checks/commits; at least10pct full-load gain; corruption/version/eviction fail-closed",
        "Q77_shared_build": {"original_two_case_build_bytes": 2*build_bytes, "actual_shared_build_bytes": build_bytes,
            "saved_bytes": build_bytes, "build_reduction": .5, "full_bundle_reduction_not_claimed": True,
            "two_native_case_builds_need_actual_pixel_and_navigation_validation": True},
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("motion_resolution.js")),
            data_identity(Path(__file__).with_name("persistent_asset_cache.js"))], "counted": False}
    atomic_json(args.output/"CONTRACT.json", protocol)
    print({"status": protocol["status"], "shared_build_saved_bytes": build_bytes, "cases": len(labels)}, flush=True)


if __name__ == "__main__":
    main()
