"""Run an unchanged complete composition with private preparation timings."""

from pathlib import Path
import shutil
import sys

import profile_q68_fused_fingerprint as composition
from runtime import atomic_json, data_identity

ROOT, CODE = composition.ROOT, composition.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists() or sys.argv[sys.argv.index("--identity-enabled") + 1] != "1":
        raise ValueError("A fresh exact accepted-composition diagnostic is required")
    paths = [Path(__file__), CODE / "timed_camera_preparation.py", CODE / "composed_scene_cost_node.py",
             CODE / "run_composed_scene_cost.py", CODE / "composed_scene_cost_preflight.py"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_cost_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
    previous_out, previous_transform = composition.OUT, composition.transform_generated_scene

    def transform(source):
        after = previous_transform(source)
        replacements = (
            ('out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"',
             'out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"'),
            ('from mapped_image_linux_guard import LinuxRSSGuard',
             'from mapped_image_linux_guard import LinuxRSSGuard\nfrom timed_camera_preparation import timed_camera_preparation'),
            ('mapping as store, prefetch_camera_images(4, 8) as prefetch',
             'mapping as store, timed_camera_preparation(args.output) as preparation_cost, prefetch_camera_images(4, 8) as prefetch'),
        )
        for old, new in replacements:
            if after.count(old) != 1:
                raise ValueError("The complete composition or bound prefetch entry changed")
            after = after.replace(old, new)
        return after

    composition.OUT, composition.transform_generated_scene = OUT, transform
    try:
        composition.main()
    finally:
        composition.OUT, composition.transform_generated_scene = previous_out, previous_transform
    if identities != [data_identity(path) for path in paths]:
        raise AssertionError("A physical preparation diagnostic source changed")
    atomic_json(output / "COST_ADAPTER_SOURCES.json", {"sources": identities,
        "actual_adapter_sources_preserved_unchanged": True, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
