"""Compare real path aliases and native exceptions before a full Scene trial."""

import json
import os
from pathlib import Path
import sys
import time
import threading
import types

from prefetch_path_operand import same_image_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/spr003/g0"


def check_original_factory_binding(directory):
    """Exercise original factory globals with small CPU-only interface stand-ins."""
    from PIL import Image
    from prefetch_path_operand import prefetch_path_operand
    sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"))
    scene = types.ModuleType("scene")
    cameras = types.ModuleType("scene.cameras")
    utils = types.ModuleType("utils")
    camera_utils = types.ModuleType("utils.camera_utils")
    scene.cameras, utils.camera_utils = cameras, camera_utils
    original_list = lambda *args: None
    def prepare(image, resolution):
        return types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"), value=image.resize(resolution).tobytes())
    def load(args, index, info, scale, synthetic, test):
        with Image.open(info.image_path) as image:
            return cameras.PILtoTorch(image, (info.width, info.height))
    scene.cameraList_from_camInfos, cameras.PILtoTorch, camera_utils.loadCam = original_list, prepare, load
    modules = {"scene": scene, "scene.cameras": cameras, "utils": utils, "utils.camera_utils": camera_utils}
    old_modules = {key: sys.modules.get(key) for key in modules}
    sys.modules.update(modules)
    try:
        infos = []
        for index in range(2):
            path = directory / ("binding" + str(index) + ".png")
            Image.new("RGB", (3, 2), (index, 37, 219)).save(path)
            infos.append(types.SimpleNamespace(image_path=str(path), width=3, height=2, depth_path=""))
        args = types.SimpleNamespace(data_device="cpu", resolution=1)
        values = []
        for enabled in (False, True):
            output = OUT / ("binding_candidate" if enabled else "binding_control")
            output.mkdir()
            with prefetch_path_operand(output, enabled, 2, 2) as receipt:
                actual = scene.cameraList_from_camInfos(infos, 1., args, False, False)
                values.append([item.value for item in actual])
            assert receipt["prepared"] == receipt["consumed"] == 2
            assert scene.cameraList_from_camInfos is original_list and cameras.PILtoTorch is prepare
        assert values[0] == values[1]
        assert not any(t.name.startswith("camera-decode") for t in threading.enumerate())
        return True
    finally:
        for key, old in old_modules.items():
            if old is None:
                del sys.modules[key]
            else:
                sys.modules[key] = old


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("prefetch_path_operand.py")]
    ids = [data_identity(path) for path in paths]
    for path in paths:
        (OUT / path.name).write_bytes(path.read_bytes())
    directory = OUT / "inputs"
    directory.mkdir()
    a, b = directory / "first.txt", directory / "other.txt"
    a.write_text("owned path alias fixture\n", encoding="utf-8")
    b.write_text("different native path\n", encoding="utf-8")
    alias, loop = directory / "alias.txt", directory / "loop.txt"
    alias.symlink_to(a)
    loop.symlink_to(loop)
    hard = directory / "hard.txt"
    os.link(a, hard)
    cases = [
        ("same_absolute", str(a), str(a), True),
        ("different_absolute", str(a), str(b), False),
        ("symbolic_alias", str(alias), str(a), True),
        ("hard_alias_native_path_inequality", str(hard), str(a), False),
        ("relative_same", os.path.relpath(a), os.path.relpath(a), True),
        ("pathlike_same", a, a, True),
        ("same_nonexistent_native_non_strict", str(directory / "missing"), str(directory / "missing"), True),
        ("dot_alias", str(directory / "." / "first.txt"), str(a), True),
        ("loop_first", str(loop), str(a), None),
        ("loop_second", str(a), str(loop), None),
        ("same_loop_preserve_first_exception", str(loop), str(loop), None),
    ]
    rows = []
    def result(call):
        try:
            return {"value": call(), "error": None}
        except (OSError, RuntimeError) as error:
            return {"value": None, "error": {"type": type(error).__name__, "message": str(error), "errno": getattr(error, "errno", None)}}
    for name, left, right, expected in cases:
        trace = {"native_resolve_calls": 0, "same_absolute_operand": 0, "native_fallback_comparisons": 0}
        native = result(lambda: Path(left).resolve() == Path(right).resolve())
        candidate = result(lambda: same_image_path(left, right, trace))
        if native != candidate or expected is not None and native["value"] != expected:
            raise AssertionError("The path comparison changed its native alias or exception: " + name)
        rows.append({"name": name, "native": native, "candidate": candidate, "trace": trace})
    binding = check_original_factory_binding(directory)
    checks = {"all11_real_native_path_alias_and_exception_cases_exact": len(rows) == 11,
        "actual_original_factory_globals_binding_and_restoration_regression": binding,
        "all_static_inputs_and_actual_sources_unchanged": ids == [data_identity(path) for path in paths],
        "no_cross_comparison_cache": not hasattr(same_image_path, "cache"),
        "native_first_resolve_remains_for_equal_absolute": rows[0]["trace"]["native_resolve_calls"] == 1,
        "native_two_resolves_for_aliases_relative_and_PathLike": all(row["trace"]["native_resolve_calls"] == 2 for row in rows[1:6])}
    if not all(checks.values()):
        raise AssertionError("The native path consumer contract did not pass")
    atomic_json(OUT / "RESULT.json", {"status": "native_path_component_contract_passed0", "checks": checks,
        "rows": rows, "sources": ids, "elapsed_seconds": time.perf_counter() - started,
        "new_optimization_count": 0, "binding_fixture_uses_CPU_interface_standins_not_real_3DGS_Camera": True,
        "dynamic_namespace_ABA_and_whole_Scene_not_verified": True})
    print(json.dumps({"checks": checks, "cases": len(rows), "new": 0}))


if __name__ == "__main__":
    from mapped_image_linux_guard import LinuxRSSGuard
    resources = OUT.with_name("g0_resource")
    resources.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(resources, atomic_json, maximum_seconds=30, maximum_bytes=1024**3):
        main()
