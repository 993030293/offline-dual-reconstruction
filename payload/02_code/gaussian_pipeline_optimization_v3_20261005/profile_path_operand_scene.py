"""Compare the full original Scene with one local path operand elimination."""

from pathlib import Path
import json
import shutil
import sys

import profile_q68_fused_fingerprint as composition
from runtime import atomic_json, data_identity

ROOT, CODE = composition.ROOT, composition.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/spr004"


def main():
    index = sys.argv.index("--operand-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The owned static path route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    enabled = value == "1"
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--identity-enabled") + 1] != "1":
        raise ValueError("Both fresh sides require the original complete input identity checks")
    paths = [Path(__file__), CODE / "prefetch_path_operand.py", CODE / "path_operand_scene_node.py",
             CODE / "run_path_operand_scene_campaign.py"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_path_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
    previous_out, previous_transform = composition.OUT, composition.transform_generated_scene

    def transform(source):
        after = previous_transform(source)
        for old, new in (
            ('out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"', 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/spr004"'),
            ('from mapped_image_linux_guard import LinuxRSSGuard', 'from mapped_image_linux_guard import LinuxRSSGuard\nfrom prefetch_path_operand import prefetch_path_operand'),
            ('prefetch_camera_images(4, 8) as prefetch', 'prefetch_path_operand(args.output, ' + repr(enabled) + ', 4, 8) as prefetch'),
        ):
            if after.count(old) != 1:
                raise ValueError("The full original Scene path consumer boundary changed")
            after = after.replace(old, new)
        return after

    composition.OUT, composition.transform_generated_scene = OUT, transform
    try:
        composition.main()
    finally:
        composition.OUT, composition.transform_generated_scene = previous_out, previous_transform
    trace = json.loads((output / "PATH_OPERAND_TRACE.json").read_text(encoding="utf-8"))
    checks = {
        "all_own_actual_sources_unchanged": identities == [data_identity(path) for path in paths],
        "all_original_functions_restored": trace["original_functions_restored"],
        "all1016_native_first_path_resolves_preserved": not enabled or trace["native_resolve_calls"] == 1016,
        "all1016_same_absolute_operands_and_no_fallback": not enabled or trace["same_absolute_operand"] == 1016 and trace["native_fallback_comparisons"] == 0,
        "no_cross_comparison_cache_and_original_full_identity_scope": trace["no_cross_comparison_cache"] and trace["static_registered_namespace_required"],
    }
    atomic_json(output / "PATH_ADAPTER_CHECKS.json", {"checks": checks, "sources": identities, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The complete path consumer did not preserve its original bindings")


if __name__ == "__main__":
    main()
