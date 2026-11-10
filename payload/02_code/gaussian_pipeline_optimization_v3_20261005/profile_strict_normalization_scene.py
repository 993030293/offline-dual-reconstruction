"""Apply one exact arithmetic route to the original full Scene consumer."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

import profile_q68_fused_fingerprint as composition
from runtime import atomic_json, data_identity
from strict_image_normalization import strict_image_normalization

ROOT, CODE = composition.ROOT, composition.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nn001"


@contextmanager
def normalization_owner(output, library, enabled):
    trace = None
    try:
        with strict_image_normalization(library, enabled) as trace:
            yield trace
    finally:
        if trace is not None:
            atomic_json(Path(output) / "NORMALIZATION_TRACE.json", {"trace": trace,
                "exception_type": None if sys.exc_info()[0] is None else sys.exc_info()[0].__name__,
                "new_optimization_count": 0})


def main():
    index = sys.argv.index("--arithmetic-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The arithmetic route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    enabled = value == "1"
    index = sys.argv.index("--arithmetic-library")
    library = Path(sys.argv[index + 1]).resolve()
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    library.relative_to((OUT / "g0/build").resolve())
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--identity-enabled") + 1] != "1":
        raise ValueError("Both fresh sides require all original complete identity checks")
    generic = json.loads((OUT / "g0/RESULT.json").read_text(encoding="utf-8"))
    paths = [Path(__file__), CODE / "strict_image_normalization.py", CODE / "strict_image_normalization.c",
        CODE / "strict_normalization_scene_node.py", CODE / "run_strict_normalization_campaign.py", library]
    identities = [data_identity(path) for path in paths]
    if (not all(generic["checks"].values()) or generic["library"] != identities[-1]
            or generic["sources"][1:3] != identities[1:3]):
        raise ValueError("The checked strict arithmetic implementation changed")
    archive = output.with_name(output.name + "_normalization_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(paths):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    previous_out, previous_transform = composition.OUT, composition.transform_generated_scene

    def transform(source):
        after = previous_transform(source)
        replacements = (
            ('out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"',
             'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/nn001"'),
            ('from mapped_image_linux_guard import LinuxRSSGuard',
             'from mapped_image_linux_guard import LinuxRSSGuard\nfrom profile_strict_normalization_scene import normalization_owner'),
            ('mapping as store, prefetch_camera_images(4, 8) as prefetch',
             'mapping as store, normalization_owner(args.output, ' + repr(str(library)) + ', ' + repr(enabled) + ') as normalization, prefetch_camera_images(4, 8) as prefetch'),
        )
        for before, replacement in replacements:
            if after.count(before) != 1:
                raise ValueError("An original full Scene normalization boundary changed: " + before)
            after = after.replace(before, replacement)
        return after

    composition.OUT, composition.transform_generated_scene = OUT, transform
    try:
        composition.main()
    finally:
        composition.OUT, composition.transform_generated_scene = previous_out, previous_transform
    trace = json.loads((output / "NORMALIZATION_TRACE.json").read_text(encoding="utf-8"))
    checks = {
        "own_sources_and_checked_library_unchanged": identities == [data_identity(path) for path in paths],
        "original_binding_restored_after_complete_real_entry": trace["trace"]["original_binding_restored"],
        "all_original1016_calls_preserved": trace["trace"]["calls"] == 1016,
        "one_only_route_on_all_original1016_images": trace["trace"]["arithmetic_calls"] == (1016 if enabled else 0)
            and trace["trace"]["native_calls"] == (0 if enabled else 1016),
        "no_original_error_swallowed": trace["exception_type"] is None,
    }
    atomic_json(output / "NORMALIZATION_ADAPTER_CHECKS.json", {"checks": checks,
        "sources": identities, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The complete arithmetic consumer changed an original contract")


if __name__ == "__main__":
    main()
