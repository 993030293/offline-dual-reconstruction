"""Retain exact native radial-sort dependencies without running a browser."""

import json
from pathlib import Path
import shutil
import time

from native_sfm_match_shards import data_identity, read, save

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release"
VENDOR = RELEASE / "shared-build/vendor"
PAGE = RELEASE / "baseline/cases/packed"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012"


def between(text, beginning, ending):
    if text.count(beginning) != 1:
        raise ValueError("An exact native radial-sort source boundary changed")
    start = text.index(beginning)
    stop = text.find(ending, start)
    if stop <= start:
        raise ValueError("The next class-level native source endpoint is missing")
    return text[start:stop], text[:start].count("\n") + 1


def main():
    started = time.perf_counter()
    output = OWN / "f1"
    if output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh bounded source audit with its disk reserve is required")
    output.mkdir(parents=True)
    sources = [data_identity(Path(__file__)), data_identity(VENDOR / "spark.module.js")]
    inputs = [data_identity(VENDOR / "SPARK-LICENSE.txt"),
        data_identity(PAGE / "viewer.js"), data_identity(PAGE / "render_filter.json"),
        data_identity(PAGE / "display_manifest.json"), data_identity(PAGE / "camera_path.json")]
    text = (VENDOR / "spark.module.js").read_text(encoding="utf-8")
    license_text = (VENDOR / "SPARK-LICENSE.txt").read_text(encoding="utf-8")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    sections = {}
    for name, beginning, ending in (
        ("DEPTH_GAUSSIAN", "var OutputSplatDepth = class", "var OutputCovSplatDepth = class"),
        ("DEPTH_COVARIANCE", "var OutputCovSplatDepth = class", "var OutputRgba8 = class"),
        ("UPDATE_INTERNAL", "\tasync updateInternal({ scene, camera, autoUpdate }) {", "\tasync driveSort() {"),
        ("DRIVE_SORT", "\tasync driveSort() {", "\tensureLodWorker() {"),
        ("PREPARE_GENERATE", "\tprepareGenerate({ renderer, scene, timer, camera, sortRadial, renderSize, previous, lodInstances }) {", "\tcheckVersions(otherMapping) {"),
        ("READBACK", "\tasync readbackDepth({ current, renderer, numSplats, readback }) {", "\tsaveRenderState(renderer) {")):
        value, line = between(text, beginning, ending)
        sections[name] = {"source_line": line, "line_count": len(value.splitlines()), "text": value}
    literals = [line for line in text.splitlines() if line.startswith("var splatVertex_default = ") and line.endswith(";")]
    if len(literals) != 1:
        raise ValueError("The native vertex shader source is not uniquely available")
    vertex = json.loads(literals[0][len("var splatVertex_default = "):-1].replace("\t", "\\t"))
    depth = sections["DEPTH_GAUSSIAN"]["text"]
    cov_depth = sections["DEPTH_COVARIANCE"]["text"]
    update = sections["UPDATE_INTERNAL"]["text"]
    generate = sections["PREPARE_GENERATE"]["text"]
    display = read(PAGE / "display_manifest.json")
    paths = read(PAGE / "camera_path.json")
    checks = {
        "current_parent36of50_and36_acceptances_exact": state["additional_accepted_count"] == 36 and len(state["accepted_mechanisms"]) == 36 and len(state["acceptance_receipts"]) == 36,
        "native_radial_default_true_not_changed_by_audit": "this.sortRadial = options.sortRadial ?? true;" in text,
        "native_gaussian_and_covariance_radial_metric_length_center": "metric = length(center);" in depth and "metric = length(center);" in cov_depth,
        "native_direction_metric_and_bias_separate_not_replaced": "metric = dot(center, ${viewDir}) + bias;" in depth and "float bias = 100.0;" in depth,
        "native_viewChanged_has_position_and_direction_dependency": "center.distanceTo(this.sortedCenter) > .001 || dir.dot(this.sortedDir) < .999" in update,
        "native_autoUpdate_false_forces_update_not_ignored": "if (autoUpdate && !needsUpdate) doUpdate = false;" in update,
        "native_producer_version_and_mapping_version_still_used": "version !== this.current.version" in update and "mappingVersion !== this.display.mappingVersion" in update,
        "frameUpdate_camera_view_and_LOD_dependencies_retained": "object.frameUpdate?.({" in generate and "viewToWorld: this.viewToWorld" in generate and "lodIndices:" in generate,
        "SH_camera_position_dependency_present_no_unproven_direction_independence": "this.hasRgbDir() && viewOrigin" in text,
        "native_shader_already_rejects_behind_near_far_and_clipXY": "viewCenter.z >= 0.0" in vertex and "abs(clipCenter.z) >= clipCenter.w" in vertex and "abs(clipCenter.x) > clip || abs(clipCenter.y) > clip" in vertex,
        "actual_depth_and_sort_readback_bounds_not_expanded": "while (baseIndex < numSplats)" in sections["READBACK"]["text"] and 'this.sortWorker.call("sortSplats32"' in sections["DRIVE_SORT"]["text"],
        "full_existing_model_and_path_denominators_not_old180_212": display["image_count"] == 1016 and len(paths) == 1016 and display["full_gaussian_count"] == 526886 and display["box_gaussian_count"] == 60451,
        "original_world_labs_MIT_license_retained": "WORLD LABS TECHNOLOGIES" in license_text and "Permission is hereby granted" in license_text,
        "all_original_source_and_input_identities_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs),
        "registered_source_output_and_disk_budget_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3,
    }
    shutil.copyfile(VENDOR / "SPARK-LICENSE.txt", output / "SPARK-LICENSE.txt")
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    save(output / "NATIVE_SOURCE_SECTIONS.json", sections)
    (output / "NATIVE_VERTEX_SHADER.glsl").write_text(vertex, encoding="utf-8")
    save(output / "RESULT.json", {"status": "radial_sort_dependency_source_audit0", "checks": checks,
        "sources": sources, "inputs": inputs, "source_lines": {key: {field: row[field] for field in ("source_line", "line_count")} for key, row in sections.items()},
        "no_browser_GPU_algorithm_or_dynamic_cost_executed": True,
        "direction_invalidation_is_potential_only_producer_and_force_semantics_unverified": True,
        "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    print(json.dumps({"checks": checks, "new_count": 0}))
    if not all(checks.values()):
        raise AssertionError("A native radial-sort source contract check failed")


if __name__ == "__main__":
    main()
