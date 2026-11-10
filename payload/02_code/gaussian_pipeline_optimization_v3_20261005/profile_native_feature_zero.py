"""Use the complete native 1016 Scene with only the scoped zero route changed."""

from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import shutil
import sys

import profile_mapped_scene as parent
from native_feature_zero_v2 import native_feature_zero
from runtime import atomic_json, data_identity

ROOT, CODE = parent.ROOT, Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"


def main():
    index = sys.argv.index("--zero-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The original zero-feature route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--mode") + 1] != "baseline":
        raise ValueError("Both complete zero-feature sides require a fresh native baseline Scene")
    original = CODE / "profile_mapped_scene.py"
    source = original.read_text(encoding="utf-8")
    anchor = 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"'
    if source.count(anchor) != 1:
        raise ValueError("The pinned complete Scene output boundary changed")
    changed = source.replace(anchor, 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"')
    paths = [Path(__file__), original, *[CODE / name for name in ("native_feature_zero.py",
        "native_feature_zero_v2.py", "native_transfer_meter.py", "native_cuda_transfer_counter.c",
        "native_zero_feature_node.py")], OUT / "build/native_transfer_counter.so"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_zero_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(paths):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    (archive / "EXECUTED_NATIVE_SCENE.py").write_text(changed, encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "owned_complete_zero_feature_Scene"}
    exec(compile(changed, str(original), "exec"), namespace)
    native_activate = namespace["activate_reference"]
    with ExitStack() as stack:
        def activate(runtime):
            metadata = native_activate(runtime)
            namespace["_zero_trace"] = stack.enter_context(native_feature_zero(output, value == "1"))
            return metadata
        namespace["activate_reference"] = activate
        namespace["main"]()
    trace = namespace["_zero_trace"]
    checks = {"actual_complete_original_initializer_once": len(trace["calls"]) == 1 and trace["calls"][0]["status"] == "complete",
              "actual_native_all_upload_payloads_measured": trace["actual_native_HtoD_bytes"] > 0
                and trace["calls"][0]["native_API_matches_all_tensor_uploads"],
              "all_native_functions_restored": trace["native_functions_restored"],
              "selected_exact_native_route": trace["calls"][0]["route"] == ("direct_zero" if value == "1" else "native"),
              "all_own_sources_and_binary_unchanged": identities == [data_identity(path) for path in paths]}
    atomic_json(output / "ZERO_ADAPTER_CHECKS.json", {"checks": checks, "sources": identities,
        "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The actual complete zero-feature native contract failed")


if __name__ == "__main__":
    main()
