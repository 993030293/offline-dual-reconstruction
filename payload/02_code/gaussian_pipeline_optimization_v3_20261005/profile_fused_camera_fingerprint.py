"""Measure the original complete Scene with optional exact layout digest fusion."""

import difflib
import json
from pathlib import Path
import shutil
import sys

from fused_camera_fingerprint import ImageFingerprintCache
from profile_scene_fingerprint_cost import ROOT, CODE, OUT
from profile_scene_transform_cost import BoundedSceneGuard
from runtime import atomic_json, data_identity


def main():
    index = sys.argv.index("--fused-enabled")
    enabled = sys.argv[index+1]
    if enabled not in ("0", "1"):
        raise ValueError("The complete consumer requires an explicit original or fused route")
    sys.argv = sys.argv[:index] + sys.argv[index+2:]
    index = sys.argv.index("--library")
    library = Path(sys.argv[index+1]).resolve()
    library.relative_to((OUT / "g0/build").resolve())
    sys.argv = sys.argv[:index] + sys.argv[index+2:]
    output = Path(sys.argv[sys.argv.index("--output")+1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or "--cache" in sys.argv or sys.argv[sys.argv.index("--mode")+1] != "baseline":
        raise ValueError("The full fused fingerprint consumer requires a fresh baseline-only Scene")
    enabled = enabled == "1"
    original = CODE / "profile_mapped_scene.py"
    before = original.read_text(encoding="utf-8")
    scope = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    if before.count(scope) != 1:
        raise ValueError("The original complete Scene output boundary changed")
    after = before.replace(scope, scope.replace("mi001", "scf001"))
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
                return None
            _metadata = {"shape": list(value.shape), "stride": list(value.stride()),
                "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                "dtype": str(value.dtype), "device": str(value.device)}
            _metadata["sha256"] = _fused_take(value) if _field == "original_image" else hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            return _metadata
'''
    if enabled:
        call = "**{name: tensor(getattr(camera, name)) for name in field_names}"
        start = "            pixel_started = time.perf_counter()"
        finish = "            pixel_seconds = time.perf_counter()-pixel_started"
        for anchor in (old_tensor, call, start, finish):
            if after.count(anchor) != 1:
                raise ValueError("An original full camera fingerprint consumer changed")
        after = after.replace(old_tensor, new_tensor).replace(call, call.replace("getattr(camera, name))", "getattr(camera, name), name)"))
        after = after.replace(start, start + "\n            _fused_prepare(cameras)")
        after = after.replace(finish, "            _fused_close()\n" + finish)
    sources = [Path(__file__), original, CODE / "fused_camera_fingerprint.py", CODE / "fused_camera_digest.py",
               CODE / "fused_camera_digest.c", CODE / "fused_fingerprint_scene_node.py", CODE / "profile_scene_fingerprint_cost.py",
               CODE / "profile_scene_transform_cost.py", CODE / "scene_transform_cost_node.py", library]
    ids = [data_identity(path) for path in sources]
    generic = json.loads((OUT / "g0/RESULT.json").read_text(encoding="utf-8"))
    if generic["library"] != data_identity(library) or not all(generic["checks"].values()):
        raise ValueError("The full consumer did not use the exact checked native library")
    snapshot = output.with_name(output.name + "_fingerprint_sources")
    snapshot.mkdir(exist_ok=False)
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshot / (f"{index:03d}_" + path.name))
    (snapshot / "EXECUTED_FULL_SCENE_PROFILE.py").write_text(after, encoding="utf-8")
    (snapshot / "FULL_SCENE_PROFILE.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="original", tofile="private_complete_digest_consumer")), encoding="utf-8")
    trace = {"enabled": enabled, "new_optimization_count": 0}
    cache = ImageFingerprintCache(library, trace) if enabled else None
    namespace = {"__file__": str(original), "__name__": "original_complete_exact_fused_digest_Scene"}
    if cache is not None:
        namespace.update(_fused_prepare=cache.prepare, _fused_take=cache.take, _fused_close=cache.close)
    exec(compile(after, str(original), "exec"), namespace)
    namespace["LinuxRSSGuard"] = BoundedSceneGuard
    try:
        namespace["main"]()
    except BaseException as error:
        if cache is not None:
            cache.close(aborted=True)
        if output.is_dir():
            atomic_json(output / "FAILED_FINGERPRINT_TRACE.json", {"trace": trace, "error": repr(error), "sources": ids, "new_optimization_count": 0})
        raise
    current = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    checks = {"all_original11_complete_checks": len(current["checks"]) == 11 and all(current["checks"].values()),
        "all1016_original_RGB_prepared_consumed_no_native_fallback": not enabled or trace["prepared"] == trace["consumed"] == 1016 and trace["native_fallback"] == 0,
        "owned_weak_bindings_closed_no_retained_images": not enabled or trace["cache_closed"] and trace["active_entries"] == 0,
        "no_full_planar_or_Python_bytes_materializations_in_fused_image_digest": not enabled or trace["full_planar_or_Python_bytes_materializations"] == 0,
        "all_own_physical_sources_and_checked_library_unchanged": ids == [data_identity(path) for path in sources]}
    atomic_json(output / "FINGERPRINT_TRACE.json", {"checks": checks, "trace": trace, "sources": ids, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The current complete fused image consumer changed an admitted binding")


if __name__ == "__main__":
    main()
