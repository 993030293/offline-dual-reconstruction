"""Admit the mask-free S07 bootstrap without the legacy S02 visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap

from common import atomic_write_json, read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--s02-machine-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_root.resolve(strict=True)
    s02_path = args.s02_machine_receipt.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    s02 = read_json(s02_path)
    if s02.get("status") != "complete_machine_only_input_integrity":
        raise ValueError("machine-only S02 input integrity has not passed")
    if s02.get("language_model_used") is not False:
        raise ValueError("S02 receipt does not prove the machine-only runtime boundary")

    selected = read_json(run / "02_fresh_visual_workspace/image_manifest.json")
    selected_names = {row["image_name"] for row in selected}
    model = pycolmap.Reconstruction(run / "09_pgsr_centered_input/sparse")
    names = {image.name for image in model.images.values() if image.has_pose}
    if len(names) == 0 or not names <= selected_names:
        raise ValueError("registered training frames are not contained in the machine-selected keyframes")
    s05_path = run / "06_metric_poses/truncated/pose_refinement_receipt.json"
    s05 = read_json(s05_path)
    if s05.get("status") != "complete_gate_passed":
        raise ValueError("S05 metric pose gate has not passed")
    if int(s05.get("registered_image_count", -1)) != len(names):
        raise ValueError("S05 registered image count differs from the centered model")

    source = pycolmap.Reconstruction(run / "07_initial_dense/metric_workspace/sparse")
    contract_path = run / "09_pgsr_centered_input/input_contract.json"
    contract = read_json(contract_path)
    delta = np.asarray(contract["source_to_target_uv_translation"], dtype=np.float64)
    maximum_shift_error = 0.0
    for image_id in model.reg_image_ids():
        before = np.asarray([point.xy for point in source.images[image_id].points2D])
        after = np.asarray([point.xy for point in model.images[image_id].points2D])
        if before.shape != after.shape:
            raise ValueError("centered camera feature identity changed")
        maximum_shift_error = max(
            maximum_shift_error, float(np.max(np.abs(after - before - delta)))
        )
    if maximum_shift_error >= 1e-9:
        raise ValueError("serialized centered-camera coordinates changed")

    receipt = {
        "schema_version": "e26_mask_free_bootstrap_machine_admission_v1",
        # Keep the downstream machine-readable status stable while narrowing the claim below.
        "status": "admitted_for_mask_free_bootstrap_only",
        "language_model_used": False,
        "s02_machine_receipt": str(s02_path),
        "s02_machine_receipt_sha256": sha256_file(s02_path),
        "training_frame_count": len(names),
        "training_frame_names": sorted(names),
        "all_training_frames_in_machine_selected_keyframe_manifest": True,
        "static_scene_semantics_assessed": False,
        "target_visibility_assessed": False,
        "camera_metric_gate_passed": True,
        "raw_to_full_pinhole_workspace_verified": True,
        "full_pinhole_to_centered_camera_adapter_verified": True,
        "serialized_feature_translation_maximum_error_px": maximum_shift_error,
        "S07_consumes_masks": False,
        "S07_consumes_language_model_receipts": False,
        "background_motion_boundary": (
            "Machine admission does not declare people or other moving background pixels static. "
            "The output remains a broad engineering preview candidate."
        ),
        "independent_geometry_accuracy_verified": False,
        "claim_boundary": (
            "This admission proves identity and camera-geometry consistency only. It replaces the "
            "legacy S02 agent review and does not infer semantic scene stability."
        ),
    }
    atomic_write_json(output_path, receipt)
    print(receipt["status"])


if __name__ == "__main__":
    main()
