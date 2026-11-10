"""Bind operand reuse to the unchanged complete native Scene diagnostic."""

from contextlib import ExitStack
from pathlib import Path
import shutil
import sys

import profile_mapped_scene as parent
from mapped_image_linux_guard import LinuxRSSGuard
from native_coordinate_reuse import native_coordinate_reuse
from runtime import atomic_json, data_identity

ROOT, CODE = parent.ROOT, Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup017/R169"
BUILD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build/native_transfer_counter.so"


class CoordinateSceneGuard(LinuxRSSGuard):
    def __init__(self, output, publish):
        super().__init__(output, publish, maximum_seconds=300, maximum_bytes=16 * 1024**3)


def main():
    index = sys.argv.index("--zero-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The private operand route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--mode") + 1] != "baseline":
        raise ValueError("Both complete sides need a fresh original native Scene")
    original = CODE / "profile_mapped_scene.py"
    source = original.read_text(encoding="utf-8")
    anchor = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    if source.count(anchor) != 1:
        raise ValueError("The native complete Scene boundary changed")
    changed = source.replace(anchor, 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup017/R169"')
    paths = [Path(__file__), original, *[CODE / name for name in (
        "native_coordinate_reuse.py", "native_transfer_meter.py", "native_cuda_transfer_counter.c",
        "native_coordinate_node.py")], BUILD]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_coordinate_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for number, path in enumerate(paths):
        shutil.copyfile(path, archive / (f"{number:03d}_" + path.name))
    (archive / "EXECUTED_NATIVE_SCENE.py").write_text(changed, encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "owned_complete_coordinate_Scene"}
    exec(compile(changed, str(original), "exec"), namespace)
    native_activate = namespace["activate_reference"]
    namespace["LinuxRSSGuard"] = CoordinateSceneGuard
    with ExitStack() as stack:
        def activate(runtime):
            metadata = native_activate(runtime)
            namespace["_coordinate_trace"] = stack.enter_context(native_coordinate_reuse(output, value == "1"))
            return metadata
        namespace["activate_reference"] = activate
        namespace["main"]()
    trace = namespace["_coordinate_trace"]
    checks = {"actual_complete_original_initializer_once": len(trace["calls"]) == 1 and trace["calls"][0]["status"] == "complete",
              "actual_native_all_upload_payloads_measured": trace["actual_native_HtoD_bytes"] > 0
                and trace["calls"][0]["native_API_matches_all_tensor_uploads"],
              "all_native_functions_restored": trace["native_functions_restored"],
              "selected_registered_route": trace["calls"][0]["route"] == ("coordinate_reuse" if value == "1" else "direct_zero"),
              "all_own_sources_and_binary_unchanged": identities == [data_identity(path) for path in paths]}
    atomic_json(output / "COORDINATE_ADAPTER_CHECKS.json", {"checks": checks, "sources": identities,
        "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The complete native coordinate contract failed")


if __name__ == "__main__":
    main()
