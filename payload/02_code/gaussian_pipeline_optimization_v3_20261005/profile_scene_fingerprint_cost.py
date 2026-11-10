"""Attribute the unchanged complete camera fingerprint path by field and operation."""

import difflib
import json
import math
from pathlib import Path
import shutil
import sys

from profile_scene_transform_cost import BoundedSceneGuard
from runtime import atomic_json, data_identity

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
FIELDS = ("original_image", "alpha_mask", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")
STAGES = ("metadata_detach", "cpu", "contiguous", "numpy", "tobytes", "sha256")


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or "--cache" in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "baseline":
        raise ValueError("The full field diagnostic requires a fresh baseline-only Scene")
    original = CODE / "profile_mapped_scene.py"
    before = original.read_text(encoding="utf-8")
    old_scope = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    old_tensor = '''        def tensor(value):
            if value is None:
                return None
            return {"shape": list(value.shape), "stride": list(value.stride()),
                "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                "dtype": str(value.dtype), "device": str(value.device),
                "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
'''
    new_tensor = '''        def tensor(value, _field=None):
            if value is None:
                if _field is not None:
                    _fp_none[_field] += 1
                return None
            if _field is None:
                return {"shape": list(value.shape), "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                    "dtype": str(value.dtype), "device": str(value.device),
                    "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            _begin = time.perf_counter()
            _meta = {"shape": list(value.shape), "stride": list(value.stride()),
                "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                "dtype": str(value.dtype), "device": str(value.device)}
            _detached = value.detach()
            _a = time.perf_counter()
            _cpu = _detached.cpu()
            _b = time.perf_counter()
            _contiguous = _cpu.contiguous()
            _c = time.perf_counter()
            _array = _contiguous.numpy()
            _d = time.perf_counter()
            _bytes = _array.tobytes()
            _e = time.perf_counter()
            _meta["sha256"] = hashlib.sha256(_bytes).hexdigest()
            _f = time.perf_counter()
            _fp_record(_field, _meta, len(_bytes), (_a-_begin, _b-_a, _c-_b, _d-_c, _e-_d, _f-_e))
            return _meta
'''
    old_call = "**{name: tensor(getattr(camera, name)) for name in field_names}"
    for anchor in (old_scope, old_tensor, old_call):
        if before.count(anchor) != 1:
            raise ValueError("An exact original complete field consumer anchor changed")
    after = before.replace(old_scope, old_scope.replace("mi001", "scf001"))
    after = after.replace(old_tensor, new_tensor).replace(old_call, old_call.replace("getattr(camera, name))", "getattr(camera, name), name)"))
    trace = {field: {"count": 0, "bytes": 0, "layouts": {}, "seconds": {stage: 0. for stage in STAGES}} for field in FIELDS}
    none = {field: 0 for field in FIELDS}

    def record(field, meta, size, times):
        if field not in trace or len(times) != len(STAGES) or any(not math.isfinite(v) or v < 0 for v in times):
            raise ValueError("An original field timing is not finite and complete")
        row = trace[field]
        row["count"] += 1
        row["bytes"] += size
        key = json.dumps({name: meta[name] for name in ("shape", "stride", "dtype", "device")}, sort_keys=True)
        row["layouts"][key] = row["layouts"].get(key, 0) + 1
        for stage, value in zip(STAGES, times):
            row["seconds"][stage] += value

    sources = [Path(__file__), original, CODE / "fingerprint_scene_node.py", CODE / "profile_scene_transform_cost.py",
               CODE / "scene_transform_cost_node.py", CODE / "mapped_image_linux_guard.py"]
    ids = [data_identity(path) for path in sources]
    namespace = {"__file__": str(original), "__name__": "original_complete_field_diagnostic",
                 "_fp_record": record, "_fp_none": none}
    exec(compile(after, str(original), "exec"), namespace)
    namespace["LinuxRSSGuard"] = BoundedSceneGuard
    namespace["main"]()
    current = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    reference_path = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001/c1/t0/native/RESULT.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    differences = [key for key in set(current["scientific"]) | set(reference["scientific"])
                   if current["scientific"].get(key) != reference["scientific"].get(key)]
    files = {key: Path(current["generated_assets"][key]["path"]).read_bytes()
                    == Path(reference["generated_assets"][key]["path"]).read_bytes() for key in current["generated_assets"]}
    seconds = sum(sum(row["seconds"].values()) for row in trace.values())
    checks = {
        "all11_original_complete_Scene_checks": len(current["checks"]) == 11 and all(current["checks"].values()),
        "all1016_by_field_original_counts_and_None_alpha": all(trace[field]["count"] == 1016 for field in FIELDS if field != "alpha_mask")
            and trace["alpha_mask"]["count"] == 0 and none["alpha_mask"] == 1016 and sum(none.values()) == 1016,
        "all15_scientific_fields_and_full1016_RGB_matrix_alias_RNG_exact": len(current["scientific"]) == 15 and not differences,
        "same17_physical_sources_inputs_runtime_and_config": current["sources"] == reference["sources"] and len(current["sources"]) == 17
            and current["inputs"] == reference["inputs"] and current["runtime"] == reference["runtime"] and current["runtime_config"] == reference["runtime_config"],
        "both_generated_assets_full_bytes_exact": len(files) == 2 and all(files.values()),
        "all_field_nonoverlap_timings_covered_by_full_consumer": 0 < seconds <= current["all_pixel_and_state_check_seconds"],
        "own_adapters_and_original_physical_sources_unchanged": ids == [data_identity(path) for path in sources],
    }
    snapshots = output / "diagnostic_source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshots / (f"{index:03d}_" + path.name))
    (output / "EXECUTED_FIELD_SCENE_PROFILE.py").write_text(after, encoding="utf-8")
    (output / "FIELD_SCENE_PROFILE.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="original", tofile="field_cost_diagnostic")), encoding="utf-8")
    result = {"status": "actual_complete1016_field_cost_diagnostic0" if all(checks.values()) else "failed_verification",
        "checks": checks, "by_field": trace, "None_field_counts": none, "nonoverlapping_original_operations_seconds": seconds,
        "instrumented_full_field_consumer_seconds": current["all_pixel_and_state_check_seconds"], "different_scientific_keys": differences,
        "generated_asset_actual_full_bytes_equal": files, "source_identities": ids, "reference_result": str(reference_path),
        "synchronization_and_timing_overhead_not_formal_speed_or_strict_upper_bound": True,
        "full_training_GT_new_object_errors_and_recovery": "not_run", "new_optimization_count": 0}
    atomic_json(output / "FINGERPRINT_COST.json", result)
    if not all(checks.values()):
        raise AssertionError("The actual complete field diagnostic changed an original protected contract")
    print(json.dumps({"status": result["status"], "checks": checks, "all_field_seconds": result["instrumented_full_field_consumer_seconds"],
                      "operations_seconds": {field: row["seconds"] for field, row in trace.items()}}))


if __name__ == "__main__":
    main()
