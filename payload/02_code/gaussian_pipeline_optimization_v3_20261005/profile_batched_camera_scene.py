"""Run original complete Scene semantics with optional private batched transforms."""

from contextlib import ExitStack, contextmanager
import difflib
import json
from pathlib import Path
import shutil
import sys

import profile_scene_transform_cost as common
from batched_camera_execution import batched_camera_execution

CODE, ROOT, BASE = common.CODE, common.ROOT, common.BASE


def main():
    if "--batch-enabled" not in sys.argv:
        raise ValueError("The complete camera profile requires an explicit mode flag")
    index = sys.argv.index("--batch-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The explicit complete camera mode is invalid")
    enabled = value == "1"
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(BASE.resolve())
    if output.exists() or "--cache" in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "baseline":
        raise ValueError("The full profile requires a fresh original baseline dataset")
    original = CODE / "profile_mapped_scene.py"
    before = original.read_text(encoding="utf-8")
    anchor = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    if before.count(anchor) != 1:
        raise ValueError("The preserved whole Scene output scope changed")
    after = before.replace(anchor, 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001"')
    namespace = {"__file__": str(original), "__name__": "original_complete_private_batch_Scene"}
    exec(compile(after, str(original), "exec"), namespace)
    sources = [Path(__file__), original, CODE / "batched_camera_execution.py", CODE / "batched_camera_transforms.py",
               CODE / "profile_scene_transform_cost.py", CODE / "batched_camera_scene_node.py", CODE / "scene_transform_cost_node.py"]
    source_ids = [common.data_identity(p) for p in sources]
    original_activate = namespace["activate_reference"]
    original_prefetch = namespace["prefetch_camera_images"]
    binder = None
    trace = {"enabled": enabled, "new_optimization_count": 0}
    namespace["LinuxRSSGuard"] = common.BoundedSceneGuard
    with ExitStack() as stack:
        def activate(runtime):
            nonlocal binder
            metadata = original_activate(runtime)
            if enabled:
                binder = stack.enter_context(batched_camera_execution(output, trace))
            return metadata
        @contextmanager
        def prefetch(*args, **kwargs):
            with original_prefetch(*args, **kwargs) as receipt:
                if enabled:
                    if binder is None:
                        raise RuntimeError("The private camera binding was not initialized")
                    with binder():
                        yield receipt
                else:
                    yield receipt
        namespace["activate_reference"] = activate
        namespace["prefetch_camera_images"] = prefetch
        namespace["main"]()
    snapshots = output / "batch_source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshots / (f"{index:03d}_" + path.name))
    (output / "EXECUTED_BATCH_SCENE_PROFILE.py").write_text(after, encoding="utf-8")
    (output / "BATCH_SCENE_PROFILE.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="private_scope")), encoding="utf-8")
    checks = {"all_profile_sources_preserved": source_ids == [common.data_identity(p) for p in sources],
              "full1016_current_batch_bindings_consumed_once": not enabled or trace["batch_prepared"] == trace["batch_consumed"] == 1016,
              "exact_one_train_batch_no_fallback_and_empty_test": not enabled or trace["batch_groups"] == 1 and not trace["nonadmitted_groups"] and trace["native_calls"] == 0,
              "private_batch_functions_and_entries_closed": not enabled or trace["original_functions_and_linecache_restored"] and trace["active_entries_after_close"] == 0}
    common.atomic_json(output / "BATCH_TRACE.json", {"checks": checks, "trace": trace, "sources": source_ids, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("A complete current Scene batch admission or closure check failed")


if __name__ == "__main__":
    main()
