"""Measure unchanged camera transforms during the complete native Scene load."""

from contextlib import contextmanager, ExitStack
import difflib
import inspect
import json
import linecache
from pathlib import Path
import shutil
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001"


class BoundedSceneGuard(LinuxRSSGuard):
    def __init__(self, output, publish):
        super().__init__(output, publish, maximum_seconds=300, maximum_bytes=16 * 1024**3)


@contextmanager
def timed_camera_initialization(output, trace):
    import scene.cameras as cameras
    import torch
    original = cameras.Camera.__init__
    before = textwrap.dedent(inspect.getsource(original))
    anchors = (
        ("    self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()",
         "world_view_prepare_transfer_seconds"),
        ("    self.camera_center = self.world_view_transform.inverse()[3, :3]",
         "camera_center_inverse_seconds"),
    )
    after = before
    for expression, key in anchors:
        if after.count(expression) != 1:
            raise ValueError("The exact original camera transform expression changed")
        measured = "    torch.cuda.synchronize()\n    _cost_begin = _cost_clock()\n" + expression
        measured += "\n    torch.cuda.synchronize()\n    _cost_trace[" + repr(key) + "].append(_cost_clock() - _cost_begin)"
        after = after.replace(expression, measured)
    native_world = original.__globals__["getWorld2View2"]

    def world(*args, **kwargs):
        begin = time.perf_counter()
        result = native_world(*args, **kwargs)
        trace["world2view_CPU_seconds_included_in_prepare"].append(time.perf_counter() - begin)
        return result

    namespace = dict(original.__globals__)
    namespace.update(_cost_clock=time.perf_counter, _cost_trace=trace, getWorld2View2=world)
    filename = str(output / "EXECUTED_TIMED_CAMERA_INIT.py")
    prior_linecache = linecache.cache.get(filename)
    linecache.cache[filename] = (len(after), None, after.splitlines(keepends=True), filename)
    Path(filename).write_text(after, encoding="utf-8")
    (output / "TIMED_CAMERA_INIT.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="diagnostic")), encoding="utf-8")
    exec(compile(after, filename, "exec"), namespace)
    cameras.Camera.__init__ = namespace["__init__"]
    try:
        yield original
    finally:
        cameras.Camera.__init__ = original
        if prior_linecache is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = prior_linecache
        trace["native_camera_function_and_linecache_restored"] = cameras.Camera.__init__ is original


def main():
    if "--output" not in sys.argv:
        raise ValueError("An explicit private diagnostic output is required")
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(BASE.resolve())
    if output.exists() or "--cache" in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "baseline":
        raise ValueError("The first diagnostic requires a new baseline-only complete Scene")
    original_profile = CODE / "profile_mapped_scene.py"
    before = original_profile.read_text(encoding="utf-8")
    anchor = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    if before.count(anchor) != 1:
        raise ValueError("The original complete Scene private output scope changed")
    after = before.replace(anchor, 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001"')
    namespace = {"__file__": str(original_profile), "__name__": "original_complete_Scene_cost_diagnostic"}
    exec(compile(after, str(original_profile), "exec"), namespace)
    original_activate = namespace["activate_reference"]
    trace = {key: [] for key in ("world_view_prepare_transfer_seconds", "camera_center_inverse_seconds",
                               "world2view_CPU_seconds_included_in_prepare")}
    sources = [Path(__file__), original_profile, CODE / "scene_transform_cost_node.py", CODE / "mapped_image_linux_guard.py"]
    source_ids = [data_identity(path) for path in sources]
    namespace["LinuxRSSGuard"] = BoundedSceneGuard
    with ExitStack() as stack:
        def activate(runtime):
            metadata = original_activate(runtime)
            stack.enter_context(timed_camera_initialization(output, trace))
            return metadata
        namespace["activate_reference"] = activate
        namespace["main"]()
    original_result = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    snapshots = output / "diagnostic_source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshots / (f"{index:03d}_" + path.name))
    (output / "EXECUTED_COMPLETE_SCENE_PROFILE.py").write_text(after, encoding="utf-8")
    (output / "COMPLETE_SCENE_PROFILE.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="private_scope")), encoding="utf-8")
    fields = ("world_view_prepare_transfer_seconds", "camera_center_inverse_seconds", "world2view_CPU_seconds_included_in_prepare")
    totals = {key: sum(trace[key]) for key in fields}
    transform = totals[fields[0]] + totals[fields[1]]
    checks = {
        "all11_original_complete_Scene_checks_passed": len(original_result["checks"]) == 11 and all(original_result["checks"].values()),
        "all1016_each_native_transform_call_count": all(len(trace[key]) == 1016 for key in fields),
        "all_measured_intervals_finite_nonnegative": all(0 <= value < 300 for key in fields for value in trace[key]),
        "CPU_world_time_is_included_not_double_counted": totals[fields[2]] <= totals[fields[0]],
        "original_native_camera_and_linecache_restored": trace["native_camera_function_and_linecache_restored"],
        "own_and_original_physical_sources_unchanged": source_ids == [data_identity(path) for path in sources],
        "complete_Scene_load_covers_transform_measurements": transform <= original_result["Scene_load_seconds"],
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_COST.json", {"checks": checks, "trace": trace, "new_optimization_count": 0})
        raise AssertionError("A complete Scene cost diagnostic contract failed")
    atomic_json(output / "TRANSFORM_COST.json", {"status": "actual_complete1016_cost_diagnostic_not_optimization0",
        "checks": checks, "trace": trace, "totals": totals, "nonoverlapping_transform_total_seconds": transform,
        "measured_transform_fraction_of_instrumented_Scene": transform / original_result["Scene_load_seconds"],
        "instrumented_Scene_load_seconds": original_result["Scene_load_seconds"],
        "synchronization_overhead_diagnostic_not_formal_speed_or_strict_upper_bound": True,
        "source_identities": source_ids, "formal_quality_recovery_external_GT_training_production": "not_run",
        "new_optimization_count": 0})
    print(json.dumps({"status": "complete_cost_diagnostic0", "Scene_load_seconds": original_result["Scene_load_seconds"],
        "transform_seconds": transform, "fraction": transform / original_result["Scene_load_seconds"], "checks": checks}))


if __name__ == "__main__":
    main()
