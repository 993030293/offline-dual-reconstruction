"""Verify the original complete Scene while counting native source file opens."""

import json
from pathlib import Path
import shutil
import sys

import profile_q68_fused_fingerprint as composition
from runtime import atomic_json, data_identity

ROOT, CODE = composition.ROOT, composition.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001"


def main():
    index = sys.argv.index("--handoff-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The loaded image handoff route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    enabled = value == "1"
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--identity-enabled") + 1] != "1":
        raise ValueError("Both fresh sides require all original complete identity checks")
    paths = [Path(__file__), CODE / "open_image_handoff.py", CODE / "image_source_open_census.py",
             CODE / "open_image_handoff_node.py", CODE / "run_open_image_handoff_campaign.py"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_image_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
    previous_out, previous_transform = composition.OUT, composition.transform_generated_scene

    def transform(source):
        after = previous_transform(source)
        replacements = (
            ('out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"', 'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001"'),
            ('from mapped_image_linux_guard import LinuxRSSGuard', 'from mapped_image_linux_guard import LinuxRSSGuard\nfrom open_image_handoff import open_image_handoff\nfrom image_source_open_census import image_source_open_census'),
            ('with LinuxRSSGuard(args.output, atomic_json) as guard:', 'with LinuxRSSGuard(args.output, atomic_json) as guard, image_source_open_census(args.output, dataset):'),
            ('prefetch_camera_images(4, 8) as prefetch', 'open_image_handoff(args.output, ' + repr(enabled) + ', 4, 8) as prefetch'),
        )
        for old, new in replacements:
            if after.count(old) != 1:
                raise ValueError("The original complete image handoff boundary changed: " + old)
            after = after.replace(old, new)
        return after

    composition.OUT, composition.transform_generated_scene = OUT, transform
    try:
        composition.main()
    finally:
        composition.OUT, composition.transform_generated_scene = previous_out, previous_transform
    trace = json.loads((output / "IMAGE_HANDOFF_TRACE.json").read_text(encoding="utf-8"))
    work = json.loads((output / "SOURCE_IMAGE_OPEN_WORK.json").read_text(encoding="utf-8"))
    checks = {
        "all_own_actual_sources_unchanged": identities == [data_identity(path) for path in paths],
        "all_original_functions_and_threads_restored": trace["original_functions_restored"] and trace["owned_decode_threads_closed"] and work["native_functions_restored"],
        "all1016_loaded_native_images_or_exact_control": (trace["recorded"] == trace["handed_off"] == trace["closed_images"] == 1016 and not trace["failed_future_native_header_fallbacks"] and not trace["unsupported_native_header_fallbacks"]) if enabled else trace["recorded"] == trace["handed_off"] == trace["closed_images"] == 0,
        "bounded_owned_images_and_no_cache": trace["maximum_owned_images"] <= 8 and trace["remaining_images"] == 0 and trace["no_persistent_cache"],
        "actual_complete_source_file_work_census": work["registered_source_images"] == 1016 and not work["failed_open_attempts"] and work["actual_native_call_count_not_Image_open_estimate"],
    }
    atomic_json(output / "HANDOFF_ADAPTER_CHECKS.json", {"checks": checks, "sources": identities, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The complete image handoff changed an original native contract")


if __name__ == "__main__":
    main()
