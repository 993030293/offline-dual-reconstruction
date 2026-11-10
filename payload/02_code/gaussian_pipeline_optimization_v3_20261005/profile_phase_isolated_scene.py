"""Run the exact full composition with an optional fresh CPU phase barrier."""

from pathlib import Path
import shutil
import sys

import profile_q68_fused_fingerprint as composition
from runtime import atomic_json, data_identity

ROOT, CODE = composition.ROOT, composition.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sip001"


def main():
    index = sys.argv.index("--phase-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The complete phase route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    enabled = value == "1"
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--identity-enabled") + 1] != "1":
        raise ValueError("Both fresh complete sides require the same accepted composition")
    paths = [Path(__file__), CODE / "phase_isolated_camera_prepare.py", CODE / "phase_isolated_scene_node.py",
        CODE / "run_phase_isolated_scene_campaign.py"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_phase_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
    previous_out, previous_transform = composition.OUT, composition.transform_generated_scene

    def transform(source):
        after = previous_transform(source)
        replacements = (
            ('out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"',
             'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sip001"'),
            ('from mapped_image_linux_guard import LinuxRSSGuard',
             'from mapped_image_linux_guard import LinuxRSSGuard\nfrom phase_isolated_camera_prepare import phase_isolated_camera_prepare'),
            ('        with (camera_loader_adapter(dataset) as manifest,',
             '        with (camera_loader_adapter(dataset) as manifest,\n                phase_isolated_camera_prepare(args.output, ' + repr(enabled) + ') as phase_bind,'),
            ('mapping as store, prefetch_camera_images(4, 8) as prefetch):',
             'mapping as store, prefetch_camera_images(4, 8) as prefetch, phase_bind()):'),
        )
        for old, new in replacements:
            if after.count(old) != 1:
                raise ValueError("The exact complete composition or prefetch binding changed")
            after = after.replace(old, new)
        return after

    composition.OUT, composition.transform_generated_scene = OUT, transform
    try:
        composition.main()
    finally:
        composition.OUT, composition.transform_generated_scene = previous_out, previous_transform
    trace = __import__("json").loads((output / "PHASE_PREPARATION_TRACE.json").read_text(encoding="utf-8"))
    checks = {
        "all_own_actual_sources_unchanged": identities == [data_identity(path) for path in paths],
        "all_original_prepare_and_camera_functions_restored": trace["original_functions_restored"],
        "all_own_CPU_workers_and_entries_closed": trace["original_CPU_workers_closed"] and trace["active_entries_after_close"] == 0,
        "all1016_fresh_original_images_prepared_and_consumed_once": not enabled or trace["prepared"] == trace["reused"] == 1016 and trace["groups"] == 1 and trace["native_fallback"] == 0 and not trace["nonadmitted_groups"] and trace["stored_original_errors"] == 0,
        "original4workers8pending_and_explicit_payload_bound": trace["workers"] == 4 and trace["capacity"] == 8 and trace["maximum_pending"] <= 8 and trace["payload_bytes"] <= trace["maximum_payload_bytes"],
    }
    atomic_json(output / "PHASE_ADAPTER_CHECKS.json", {"checks": checks, "sources": identities, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The complete fresh phase consumer did not preserve its owned bindings")


if __name__ == "__main__":
    main()
