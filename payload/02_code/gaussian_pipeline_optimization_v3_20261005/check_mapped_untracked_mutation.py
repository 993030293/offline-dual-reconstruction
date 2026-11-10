"""Exercise a NumPy storage write that bypasses the native Tensor version counter."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from mapped_training_images import mapped_training_images
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json):
        activate_reference(ROOT/"04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        import torch
        from PIL import Image
        import scene.cameras as cameras
        native = cameras.PILtoTorch
        path = args.output/"input.png"
        Image.fromarray(np.arange(3072, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8)).save(path)
        source = data_identity(Path(__file__).with_name("mapped_training_images.py"))
        with Image.open(path) as image:
            reference = native(image, (32, 32))
        with mapped_training_images(args.output/"cache", data_identity) as store:
            with Image.open(path) as image:
                candidate = cameras.PILtoTorch(image, (32, 32))
                sibling = cameras.PILtoTorch(image, (32, 32))
            payload = next((args.output/"cache").glob("*.f32"))
            disk_before = data_identity(payload)
            version_before = candidate._version
            reference.numpy()[0, 0, 0] = -3.125
            candidate.numpy()[0, 0, 0] = -3.125
            version_after = candidate._version
            before_equal = torch.equal(reference, candidate)
            store.release_clean_pages(candidate)
            store.release_clean_pages(sibling)
            after_equal = torch.equal(reference, candidate)
            checks = {
                "actual_numpy_write_bypassed_version": version_before == version_after,
                "values_equal_before_release": before_equal,
                "numpy_private_write_survived_release": after_equal,
                "sibling_unchanged": sibling[0, 0, 0].item() == 0,
                "read_only_payload_unchanged": disk_before == data_identity(payload),
            }
            trace = store.receipt()
            witness = {"reference_first": reference[0, 0, 0].item(),
                "candidate_first_after_release": candidate[0, 0, 0].item(),
                "version_before": version_before, "version_after": version_after}
        checks["factory_restored"] = cameras.PILtoTorch is native
        checks["source_unchanged"] = source == data_identity(source["path"])
        status = "all_untracked_COW_contract_checks_passed" if all(checks.values()) else "failed_verification"
        atomic_json(args.output/"RESULT.json", {"status": status, "checks": checks,
            "source": source, "trace": trace, "witness": witness, "new_optimization_count": 0})
        print(json.dumps({"status": status, "checks": checks, "witness": witness}))
        if not all(checks.values()):
            raise AssertionError("A native storage mutation was lost after mapped-page release")


if __name__ == "__main__":
    main()
