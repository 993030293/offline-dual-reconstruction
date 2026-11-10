"""Check independent camera-volume geometry and weighted-loss gradient oracles."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from full_intrinsics import Pinhole
from runtime import atomic_json, data_identity
from target_weight_adapter import TargetVolumeLoss, volume_ray_mask, weighted_l1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/"CONTRACT.json", {"scope": "32x24 synthetic camera and loss oracles only",
        "foreground_weight": 2, "background_weight": 1, "loss_atol": 1e-6, "gradient_atol": 1e-7,
        "torch_threads": 1, "uses_actual_images_or_GPU": False, "counted": False})
    torch.set_num_threads(1)
    checks = {}
    intr = Pinhole(32, 24, 40., 36., 15.25, 11.75)
    low, high = np.array([-.4, -.3, 2.]), np.array([.4, .3, 3.])
    identity = np.eye(4)
    u, v = np.meshgrid(np.arange(32)+.5, np.arange(24)+.5)
    rectangle = ((u >= intr.cx-8) & (u <= intr.cx+8) & (v >= intr.cy-5.4) & (v <= intr.cy+5.4)).astype(np.uint8)
    actual = volume_ray_mask(intr, identity, low, high)
    checks["independent_front_face_rectangle"] = np.array_equal(actual, rectangle)
    rotated = identity.copy()
    rotated[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated_rectangle = ((u >= intr.cx-6) & (u <= intr.cx+6) & (v >= intr.cy-7.2) & (v <= intr.cy+7.2)).astype(np.uint8)
    checks["independent_rotated_camera_rectangle"] = np.array_equal(volume_ray_mask(intr, rotated, low, high), rotated_rectangle)
    shifted = identity.copy()
    shifted[:3, 3] = [5., -7., 3.]
    checks["metric_world_translation_invariance"] = np.array_equal(actual, volume_ray_mask(intr, shifted, low+shifted[:3, 3], high+shifted[:3, 3]))
    checks["behind_camera_rejected"] = not volume_ray_mask(intr, identity, [-.4, -.3, -3], [.4, .3, -2]).any()
    checks["before_near_plane_rejected"] = not volume_ray_mask(intr, identity, [-1, -1, .1], [1, 1, .15]).any()
    checks["camera_inside_volume"] = bool(volume_ray_mask(intr, identity, [-1, -1, -1], [1, 1, 1]).all())
    parallel_intr = Pinhole(32, 24, 40, 40, 15.5, 11.5)
    parallel = volume_ray_mask(parallel_intr, identity, [.2, -.5, 2], [.4, .5, 3])
    checks["parallel_ray_outside_slab_rejected"] = not parallel[:, 15].any()
    for name, pose, lo, hi in [("invalid_volume", identity, high, low),
                               ("nonmetric_pose", np.diag([2., 1., 1., 1.]), low, high)]:
        try:
            volume_ray_mask(intr, pose, lo, hi)
            checks[name] = False
        except ValueError:
            checks[name] = True
    prediction = torch.linspace(-.5, 1.5, 3*24*32).reshape(3, 24, 32).requires_grad_()
    reference = torch.full_like(prediction, .273)
    mask = torch.from_numpy(actual)
    loss = weighted_l1(prediction, reference, mask)
    loss.backward()
    weights = 1+actual.astype(np.float64)
    difference = prediction.detach().numpy().astype(np.float64)-reference.numpy().astype(np.float64)
    expected = np.sum(np.abs(difference)*weights[None])/(3*weights.sum())
    gradient = np.sign(difference)*weights[None]/(3*weights.sum())
    loss_error = abs(float(loss)-expected)
    gradient_error = float(np.max(np.abs(prediction.grad.numpy()-gradient)))
    checks["independent_weighted_value"] = loss_error <= 1e-6
    checks["independent_weighted_gradient"] = gradient_error <= 1e-7
    checks["positive_background_gradient_retained"] = bool(np.all(np.abs(prediction.grad.numpy()[:, actual == 0]) > 0))
    plain = (prediction.detach()-reference).abs().mean()
    checks["empty_hint_preserves_plain_l1"] = abs(float(weighted_l1(prediction.detach(), reference, torch.zeros_like(mask))-plain)) <= 1e-7
    camera = SimpleNamespace(calibrated_intrinsics=asdict(intr),world_view_transform=torch.eye(4),
                             original_image=torch.zeros(3,24,32),image_name="synthetic")
    binding = TargetVolumeLoss([camera], low, high)
    binding.select(camera)
    try:
        binding.select(camera)
        checks["duplicate_render_selection_rejected"] = False
    except ValueError:
        checks["duplicate_render_selection_rejected"] = True
    binding.loss(prediction.detach(), reference)
    result = binding.finish(1)
    checks["binding_consumes_exact_camera_and_loss"] = result["calls"] == 1 and result["rows"][0]["pixels"] == int(actual.sum())
    try:
        binding.finish(2)
        checks["incomplete_denominator_rejected"] = False
    except ValueError:
        checks["incomplete_denominator_rejected"] = True
    checks = {name: bool(value) for name, value in checks.items()}
    receipt = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "maximum_weighted_loss_absolute_error": loss_error, "maximum_gradient_absolute_error": gradient_error,
        "sources": [data_identity(Path(__file__).parent/name) for name in ("validate_target_weight.py", "target_weight_adapter.py", "full_intrinsics.py")],
        "actual_212_cameras_validated": False, "training_quality_claim": False, "counted": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not all(checks.values()):
        raise RuntimeError("The independent target-weight geometry or gradient oracle failed")


if __name__ == "__main__":
    main()
