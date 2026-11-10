"""Stage freshly validated models and compose optional motion, verified cache and shared build."""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
import build_new_route_viewer as native
from build_packed_viewer import replace_once, finalize
from runtime import atomic_json, data_identity
from build_viewer_execution_cases import integrity_source, cache_source, motion_source


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stage(args):
    acceptance = read(args.acceptance)
    campaign = read(args.pair.parent/"STATE.json")
    if acceptance["stage_accepted_count"] != 3 or not all(acceptance["checks"].values()) or campaign["status"] != "three_fresh_seed_quality_and_actual_recovery_passed_visual_pending":
        raise ValueError("The fresh native camera campaign and its actual visual/recovery acceptance are required")
    source = textwrap.dedent(inspect.getsource(native.main))
    replacements = [
        ('    pair = read(args.pair / "STATE.json")', '    pair = {"status": "verified_current_campaign"}'),
        ('pair["status"] != "training_and_restore_passed_quality_pending"', 'pair["status"] != "verified_current_campaign"'),
        ('(("reference", 10000), ("combined", 12000))', '(("baseline", 12000), ("combined", 12000))'),
        ('titles = {"reference": "新地图 · 基准模型", "combined": "新地图 · 优化组合"}',
         'titles = {"baseline": "基准模型", "combined": "优化模型"}'),
        ('''    code = replace_once(code, 'location.href = "../" + event.target.value + "/index.html";',
        'location.href = "../../../" + event.target.value + "/cases/packed/index.html";')''',
         '''    if 'location.href = "../../../" + event.target.value + "/cases/packed/index.html";' not in code:
        raise ValueError("The existing calibrated viewer model-navigation entry changed")'''),
    ]
    for before, after in replacements:
        source = replace_once(source, before, after)
    namespace = dict(native.main.__globals__)
    namespace["__file__"] = str(Path(__file__))
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--pair", str(args.pair), "--source-viewer", str(args.source_viewer),
            "--camera-path", str(args.camera_path), "--roi", str(args.roi), "--output", str(args.output)]
        exec(compile(source, "<current_verified_fresh12k_pair_display_staging>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous
    atomic_json(args.output/"CURRENT_TRAINING_BINDING.json", {"pair": str(args.pair), "preselected_seed": 0,
        "chosen_before_viewer_results_not_best_seed": True, "stage_acceptance": data_identity(args.acceptance),
        "campaign_source": data_identity(args.pair.parent/"STATE.json"), "new_optimization_count": 0})


def compose(args):
    packing = read(args.output/"PACKING_VALIDATION.json")
    if packing["status"] != "four_native_assets_exported_and_roundtrip_verified":
        raise ValueError("Both current models must complete native exact packing before route composition")
    shared = args.output/"shared-build/vendor"
    shutil.copytree(args.output/"combined/cases/packed/vendor", shared)
    records = []
    labels = {"baseline": "基准模型", "combined": "优化模型", "control-baseline": "基准模型 · 原分辨率",
        "control-combined": "优化模型 · 原分辨率"}
    for label in ("baseline", "combined"):
        root = args.output/label
        target = root/"cases/packed"
        control_root = args.output/("control-"+label)
        shutil.copytree(root, control_root)
        for current_label, case in (("control-"+label, control_root/"cases/packed"), (label, target)):
            page = (case/"index.html").read_text(encoding="utf-8")
            start = page.index('<select id="case-select">')
            end = page.index("</select>", start)+len("</select>")
            selector = '<select id="case-select">'+"".join('<option value="'+name+'"'+(' selected' if name == current_label else '')+'>'+title+'</option>' for name, title in labels.items())+'</select>'
            page = page[:start]+selector+page[end:]
            (case/"index.html").write_text(page, encoding="utf-8")
        code = motion_source((target/"viewer.js").read_text(encoding="utf-8"))
        code = replace_once(code, 'state.renderPolicy = "demand";', 'state.renderPolicy = "demand";\nstate.motionResolutionEnabled = true;')
        code = replace_once(code,
            'motionResolution.beforeFrame(state.playing || keys.size > 0 || drag !== null || orbitChanged);',
            'motionResolution.beforeFrame(state.motionResolutionEnabled && (state.playing || keys.size > 0 || drag !== null || orbitChanged));')
        code = replace_once(code, '  const motionResolution = createMotionResolution(renderer, projection, state);',
            '''  const motionResolution = createMotionResolution(renderer, projection, state);
  ui("motion-priority").addEventListener("change", event => {
    state.motionResolutionEnabled = event.target.checked;
    requestRender();
  });''')
        code = code.replace('"./vendor/OrbitControls.js"', '"../../../shared-build/vendor/OrbitControls.js"')
        code = replace_once(code, 'await verifySparkBuild("./vendor/spark.module.js", assets.scene.spark_sha256);',
            'await verifySparkBuild("../../../shared-build/vendor/spark.module.js", assets.scene.spark_sha256);')
        (target/"viewer.js").write_text(code, encoding="utf-8")
        packed = cache_source(integrity_source((target/"packed_splat_asset.js").read_text(encoding="utf-8"), False))
        (target/"packed_splat_asset.js").write_text(packed, encoding="utf-8")
        for name in ("motion_resolution.js", "persistent_asset_cache.js"):
            shutil.copyfile(Path(__file__).with_name(name), target/name)
        page = (target/"index.html").read_text(encoding="utf-8")
        page = page.replace('"./vendor/', '"../../../shared-build/vendor/')
        page = replace_once(page, '<select id="case-select">',
            '<label style="display:block;margin:8px 0"><input id="motion-priority" type="checkbox" checked> 移动时优先流畅</label><select id="case-select">')
        (target/"index.html").write_text(page, encoding="utf-8")
        vendor = target/"vendor"
        files = [p for p in vendor.rglob("*") if p.is_file()]
        if any(data_identity(p)["sha256"] != data_identity(shared/p.relative_to(vendor))["sha256"] for p in files):
            raise ValueError("The current shared renderer dependency differs from the native case build")
        records.append({"label": label, "removed_duplicate_build_bytes": sum(p.stat().st_size for p in files),
            "assets_exact_to_native_control": all((target/name).read_bytes() == (control_root/"cases/packed"/name).read_bytes()
                for name in ("scene.spk", "box.spk", "camera_path.json", "display_manifest.json", "render_filter.json"))})
        if vendor.is_symlink() or not vendor.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("The duplicate derived build path escaped this owned release")
        shutil.rmtree(vendor)
    if not all(row["assets_exact_to_native_control"] for row in records):
        raise ValueError("An accepted display composition changed its model or camera inputs")
    atomic_json(args.output/"COMPOSITION.json", {"status": "staged_actual_browser_combination_pending",
        "mechanisms": ["Q69", "Q71", "Q77"], "records": records, "source": data_identity(Path(__file__)),
        "motion_optional_by_checkbox": True, "two_model_shared_build_bytes": sum(p.stat().st_size for p in shared.rglob("*") if p.is_file()),
        "native_controls_separately_retained_for_validation_not_part_of_shared_release_storage_claim": True,
        "whole_bundle_disk_savings_not_claimed": True, "new_optimization_count": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stage", "finalize", "compose"], required=True)
    for name in ("pair", "source-viewer", "camera-path", "roi", "output", "acceptance"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "finalize":
        for label in ("baseline", "combined"):
            finalize(args.output/label)
    else:
        (stage if args.mode == "stage" else compose)(args)


if __name__ == "__main__":
    main()
