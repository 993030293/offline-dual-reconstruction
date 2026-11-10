"""Stage a native-packed display comparison without re-encoding the training PLY."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

from runtime import atomic_json, data_identity


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("The accepted demand-viewer integration anchor changed: " + old[:80])
    return text.replace(old, new)


def finalize(output):
    manifest = json.loads((output / "ASSET_SOURCES.json").read_text())
    rows = []
    for label in ["scene", "box"]:
        path = output / "cases/packed" / (label + ".spk")
        payload = path.read_bytes()
        if payload[:8] != b"SPKPK001":
            raise ValueError("Unexpected packed binary version")
        size, start = struct.unpack_from("<II", payload, 8)
        if size > 16384 or start != 16 + (size + 3) // 4 * 4:
            raise ValueError("Invalid serialized header bounds")
        metadata = json.loads(payload[16:16+size])
        expected = manifest[label]
        if metadata["provenance"] != {key: expected[key] for key in ["spark_sha256", "source_ply_sha256"]}:
            raise ValueError("The packed model refers to a different PLY or Spark build")
        if metadata["numSplats"] != expected["numSplats"] or metadata["maxSh"] != 3:
            raise ValueError("The packed model dropped splats or spherical harmonics")
        offset = 0
        for name, buffer in zip(["packedArray", "sh1", "sh2", "sh3"], metadata["buffers"], strict=True):
            if buffer["name"] != name or buffer["offset"] != offset or buffer["byteLength"] != buffer["length"]*4:
                raise ValueError("Invalid serialized buffer layout")
            body = payload[start+offset:start+offset+buffer["byteLength"]]
            if hashlib.sha256(body).hexdigest() != buffer["sha256"]:
                raise ValueError("Packed buffer checksum failed after browser download")
            offset += buffer["byteLength"]
        if start+offset != len(payload) or metadata["payloadBytes"] != offset:
            raise ValueError("Packed artifact is truncated or has trailing data")
        expected.update(asset_sha256=hashlib.sha256(payload).hexdigest(), asset_bytes=len(payload))
        rows.append({"name": label, "splats": metadata["numSplats"], "source_bytes": expected["source_bytes"],
            "asset_bytes": len(payload), "size_reduction": 1-len(payload)/expected["source_bytes"]})
    atomic_json(output / "ASSET_MANIFEST.json", manifest)
    atomic_json(output / "PACKAGING_RESULT.json", {"status": "finalized_browser_validation_pending", "rows": rows,
        "all_SH_orders_preserved": True, "stage_performance_or_pixel_checks_completed": False, "counted": False})
    print(json.dumps({"status": "finalized_browser_validation_pending", "rows": rows}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.output)
        return
    if args.source is None:
        raise ValueError("Initial staging requires the accepted source viewer")
    source = args.source.resolve()
    text = (source / "viewer.js").read_text(encoding="utf-8")
    if 'state.renderPolicy = "demand"' not in text:
        raise ValueError("Both comparison cases must retain the accepted on-demand renderer")
    text = replace_once(text, 'const fullModel = new SplatMesh({ url: "./scene.ply", lod: false });',
        'const modelLoadStarted = performance.now();\n  const fullModel = new SplatMesh({ url: "./scene.ply", lod: false });')
    text = replace_once(text, 'await Promise.all([fullModel.initialized, boxModel.initialized]);',
        'await Promise.all([fullModel.initialized, boxModel.initialized]);\n'
        '  state.modelInitializationMs = performance.now() - modelLoadStarted;')
    packed = 'import { loadPackedAsset, verifySparkBuild } from "./packed_splat_asset.js";\n' + text
    packed = replace_once(packed,
        'const fullModel = new SplatMesh({ url: "./scene.ply", lod: false });\n  const boxModel = new SplatMesh({ url: "./box.ply", lod: false });',
        'const assets = await loadJSON("../../ASSET_MANIFEST.json");\n'
        '  await verifySparkBuild("./vendor/spark.module.js", assets.scene.spark_sha256);\n'
        '  const [fullPacked, boxPacked] = await Promise.all([loadPackedAsset("./scene.spk", assets.scene), loadPackedAsset("./box.spk", assets.box)]);\n'
        '  const fullModel = new SplatMesh({ splats: fullPacked, lod: false });\n'
        '  const boxModel = new SplatMesh({ splats: boxPacked, lod: false });')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source / "display_manifest.json").read_text())
    vendor = data_identity(source / "vendor/spark.module.js")
    manifest = {}
    for label, count in [("scene", metadata["full_gaussian_count"]), ("box", metadata["box_gaussian_count"])]:
        identity = data_identity(source / (label + ".ply"))
        manifest[label] = {"spark_sha256": vendor["sha256"], "source_ply_sha256": identity["sha256"],
            "source_bytes": identity["bytes"], "numSplats": count}
    for label in ["ply", "packed"]:
        folder = args.output / "cases" / label
        shutil.copytree(source, folder, ignore=shutil.ignore_patterns("scene.ply", "box.ply") if label == "packed" else None)
        (folder / "viewer.js").write_text(text if label == "ply" else packed, encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("packed_splat_asset.js"), folder / "packed_splat_asset.js")
        page = (folder / "index.html").read_text(encoding="utf-8")
        start = page.index('<select id="case-select">')
        end = page.index('</select>', start)+len('</select>')
        options = '<select id="case-select">'+''.join('<option value="'+name+'"'+(' selected' if name == label else '')+'>'+title+'</option>'
            for name, title in [("ply", "原PLY加载对照"), ("packed", "紧凑显示资产候选")])+'</select>'
        (folder / "index.html").write_text(page[:start]+options+page[end:], encoding="utf-8")
    atomic_json(args.output / "ASSET_SOURCES.json", manifest)
    atomic_json(args.output / "CONTRACT.json", {"status": "staged_browser_export_pending", "source": str(source),
        "source_scene_identity_bound": True, "spark": vendor, "conversion": "same native PackedSplats with allSH; original float32 PLY unchanged",
        "actual_packed_assets_exist": False, "counted": False, "source_identity": data_identity(Path(__file__))})
    print(json.dumps({"status": "staged_browser_export_pending", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
