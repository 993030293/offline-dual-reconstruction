"""Reproduce the observed registered-subset snapshot mismatch on a real small map."""

import argparse
import json
from pathlib import Path
import time

import pycolmap

from check_grid_support import ROOT, POLICY, source_module
from grid_support import grid_support_execution
from binary_map_qa import image_arrays
from segmented_camera_support import native_support_many
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    folders = sorted(path for path in (ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/generic_binary_qa_001").iterdir()
                     if path.is_dir() and (path/"images.bin").is_file())
    if len(folders) != 9:
        raise ValueError("The existing nine complete native map fixtures are required")
    inputs = [data_identity(path) for path in sorted(folders[0].glob("*.bin"))]
    source_paths = [Path(__file__), Path(__file__).with_name("segmented_camera_support.py"),
                    Path(__file__).with_name("native_observation_index.py")]
    sources = [data_identity(path) for path in source_paths]
    model = pycolmap.Reconstruction(folders[0])
    first = next(image for image in model.images.values() if image.has_pose)
    model.deregister_frame(first.frame_id)
    images = [image for image in model.images.values() if image.has_pose]
    module = source_module()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["camera_geometry_admission"]
    error = None
    with grid_support_execution(module, True):
        expected = [module.camera_geometry_metrics(model, image, policy) for image in images]
        try:
            native_support_many(model, images, policy, args.output/"old_index", module.camera_geometry_metrics)
        except ValueError as failure:
            error = str(failure)
    written = image_arrays(args.output/"old_index/before/images.bin")
    checks = {
        "real_deregistration_leaves_more_in_memory_images": len(model.images) > len(images) > 0,
        "native_snapshot_contains_exact_registered_subset": {row["id"] for row in written} == {image.image_id for image in images},
        "unchanged_original_scalar_support_completes": len(expected) == len(images),
        "same_observed_old_index_failure_reproduced": error == "The native snapshot image identities changed",
        "original_fixture_files_unchanged": inputs == [data_identity(path) for path in sorted(folders[0].glob("*.bin"))],
        "current_failing_sources_preserved": sources == [data_identity(path) for path in source_paths],
    }
    result = {"status": "reproduced_actual_registered_subset_snapshot_assumption" if all(checks.values()) else "failed_reproduction",
        "checks": checks, "in_memory_images": len(model.images), "registered_images": len(images),
        "native_snapshot_images": len(written), "old_exception": error, "original_complete_rows": expected,
        "sources": sources, "inputs": inputs, "complete_seconds": time.perf_counter()-started,
        "new_optimization_count": 0, "later_full_pairs": "not_run"}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key:result[key] for key in ("status", "checks", "in_memory_images", "registered_images", "native_snapshot_images", "old_exception")}))
    if not all(checks.values()):
        raise AssertionError("The actual registration failure did not reproduce under its stated scope")


if __name__ == "__main__":
    main()
