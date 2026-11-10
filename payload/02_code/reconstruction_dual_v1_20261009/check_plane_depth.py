"""Verify off-center plane depth and its differentiable normal/distance path."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from pgsr_calibration import camera_K, calibrated_render


def main():
    root = Path(sys.argv[1])
    camera = SimpleNamespace(image_width=64, image_height=48, corner_K=(60., 61., 24.1, 29.7))
    matrix = torch.tensor(camera_K(camera), dtype=torch.float64)
    y, x = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
    rays = torch.stack(((x-matrix[0, 2])/matrix[0, 0], (y-matrix[1, 2])/matrix[1, 1], torch.ones_like(x)), -1)
    camera.get_rays = lambda: rays
    normal = torch.tensor([.3, -.2, -.93], dtype=torch.float64, requires_grad=True)
    distance = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((1, 48, 64), .7, dtype=torch.float64)
    normal_map = normal[:, None, None] * alpha
    distance_map = distance * alpha
    centered = torch.stack(((x-32)/60, (y-24)/61, torch.ones_like(x)), -1)
    old_depth = distance_map / -(torch.sum(normal_map * centered.permute(2, 0, 1), 0, keepdim=True)+1e-8)
    fake = lambda *args, **kwargs: {"plane_depth": old_depth, "rendered_distance": distance_map,
        "rendered_normal": normal_map, "rendered_alpha": alpha}
    trace = {}
    result = calibrated_render(fake, trace)(camera, None, None, None, return_plane=True, return_depth_normal=False)
    expected = distance / -torch.sum(normal[:, None, None] * rays.permute(2, 0, 1), 0)
    error = float(torch.max(torch.abs(result["plane_depth"].squeeze()-expected)).detach())
    native_bias = float(torch.mean(torch.abs(old_depth.squeeze()-expected)).detach())
    result["plane_depth"].mean().backward()
    if error > 1e-6 or native_bias < .01 or not torch.isfinite(normal.grad).all() or not torch.isfinite(distance.grad):
        raise ValueError("Off-center differentiable plane-depth contract failed")
    record = {"status": "offcenter_plane_ratio_and_gradients_verified", "maximum_analytic_error_m": error,
        "native_centered_bias_m": native_bias, "normal_gradient": normal.grad.tolist(),
        "distance_gradient": float(distance.grad), "trace": trace,
        "evidence_level": "analytic_diagnostic_not_scene_accuracy_or_complete_CUDA_validation"}
    folder = root / "diagnostics/plane_depth"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "RESULT.json").write_text(json.dumps(record, indent=2)+"\n")
    print(record)


if __name__ == "__main__":
    main()
