"""Retain the complete original R140 worker while selecting its sampling route."""

import argparse
import difflib
import json
from pathlib import Path
import sys
import time

from profile_vector_full_export import BASE
from dataset_sampling_shared_carrier import cached_dataset
from vector_camera_sampling import library_supports_simd
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    output = args.contract.parent.resolve()
    output.relative_to(BASE)
    original_file = Path(__file__).with_name("export_qa_artifact_worker.py")
    before = original_file.read_text(encoding="utf-8")
    anchor = 'ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001"'
    if before.count(anchor) != 1: raise ValueError("The original complete worker scope anchor changed")
    source = before.replace(anchor, 'ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001"')
    (output / "COMPLETE_EXECUTED_ARTIFACT_WORKER.py").write_text(source, encoding="utf-8")
    (output / "ARTIFACT_WORKER_ADAPTER.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
        fromfile=original_file.name, tofile="owned_sampling_worker_scope")), encoding="utf-8")
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "retained_original_artifact_worker"}
    exec(compile(source, "<retained_original_complete_artifact_worker>", "exec"), namespace)
    native = namespace["native"]
    original = native.pycolmap.undistort_images
    calls = []

    def sampling(destination, model, images, **options):
        begin = time.perf_counter()
        library = Path(contract["r123_library"])
        native.pycolmap.undistort_images = original
        try:
            if contract["r123_sampling_enabled"] and library_supports_simd(library):
                expected = {"output_type": "COLMAP", "num_patch_match_src_images": 0, "jpeg_quality": 97, "num_threads": 8}
                if set(options) != set(expected) | {"image_names"} or any(options[k] != v for k, v in expected.items()):
                    raise ValueError("The original owned worker sampling options changed")
                trace = cached_dataset(destination, model, images, options["image_names"], library, output / "r123_sampling_proof")
            else:
                original(destination, model, images, **options)
                trace = {"route": "original_complete_native_undistortion", "closed": True}
            calls.append({"trace": trace, "seconds": time.perf_counter() - begin})
        finally:
            native.pycolmap.undistort_images = sampling

    native.pycolmap.undistort_images = sampling
    try:
        namespace["main"]()
    finally:
        native.pycolmap.undistort_images = original
        atomic_json(output / "R123_SAMPLING_TRACE.json", {"calls": calls,
            "original_binding_restored": native.pycolmap.undistort_images is original, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
