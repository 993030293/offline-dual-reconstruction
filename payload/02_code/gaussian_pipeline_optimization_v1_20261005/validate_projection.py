"""Compare full intrinsics with OpenCV and isolated real CUDA Gaussian renders."""

import argparse
from dataclasses import asdict, replace
import time

import cv2
import numpy as np

from full_intrinsics import Pinhole
from runtime import activate_reference, atomic_json, data_identity


def projected_corner_pixels(points, intrinsic):
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ intrinsic.projection().T
    ndc = homogeneous[:, :2] / homogeneous[:, 3:4]
    # The CUDA rasterizer returns integer-centred pixels; add 0.5 for COLMAP.
    raster_pixels = (ndc + 1.0) * [intrinsic.width / 2, intrinsic.height / 2] - 0.5
    return raster_pixels + 0.5


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rasterizer-library", type=Path)
    parser.add_argument("--gsplat-build", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime, args.rasterizer_library)
    if args.gsplat_build:
        if args.rasterizer_library:
            raise ValueError("Select one candidate backend")
        from gsplat_adapter import activate_gsplat, rasterize_values
        metadata["gsplat"] = activate_gsplat(args.gsplat_build)

    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.colmap_loader import read_intrinsics_binary

    start = time.monotonic()
    source = args.dataset / "sparse/0/cameras.bin"
    cameras = read_intrinsics_binary(source)
    real = Pinhole.from_colmap(cameras[sorted(cameras)[0]])
    cases = [real.resized(812, 620), real.resized(601, 459)]
    analytic_rows, raster_rows = [], []
    limits = {"analytic_max_px": 1e-3, "raster_centroid_max_px": 0.03, "gradient_relative_error": 0.03}
    identity = torch.eye(4, device="cuda")

    def raster(point, intrinsic):
        settings = GaussianRasterizationSettings(
            image_height=intrinsic.height, image_width=intrinsic.width,
            tanfovx=intrinsic.width / (2 * intrinsic.fx),
            tanfovy=intrinsic.height / (2 * intrinsic.fy),
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=identity,
            projmatrix=torch.tensor(intrinsic.projection(), device="cuda", dtype=torch.float32).T.contiguous(),
            sh_degree=0, campos=torch.zeros(3, device="cuda"),
            prefiltered=False, debug=False, antialiasing=False)
        if args.gsplat_build:
            result = rasterize_values(point, torch.full((1, 3), .006, device="cuda"),
                torch.tensor([[1., 0., 0., 0.]], device="cuda"), torch.full((1, 1), .7, device="cuda"),
                torch.ones((1, 3), device="cuda"), torch.tensor(intrinsic.matrix(), dtype=torch.float32, device="cuda"),
                identity, intrinsic.width, intrinsic.height, torch.zeros(3, device="cuda"))
        else:
            result = GaussianRasterizer(settings)(
                means3D=point, means2D=torch.zeros_like(point, requires_grad=True),
                opacities=torch.full((1, 1), 0.7, device="cuda"),
                colors_precomp=torch.ones((1, 3), device="cuda"),
                scales=torch.full((1, 3), 0.006, device="cuda"),
                rotations=torch.tensor([[1., 0., 0., 0.]], device="cuda"))
        image = result[0][0].double()
        xs = torch.arange(intrinsic.width, device="cuda", dtype=image.dtype) + 0.5
        ys = torch.arange(intrinsic.height, device="cuda", dtype=image.dtype) + 0.5
        mass = image.sum()
        return torch.stack([(image.sum(0) * xs).sum() / mass, (image.sum(1) * ys).sum() / mass])

    for intrinsic in cases:
        uv = np.array([[24.25, 24.75], [intrinsic.width - 25.25, 24.75],
                       [24.25, intrinsic.height - 25.75],
                       [intrinsic.width - 25.25, intrinsic.height - 25.75],
                       [intrinsic.cx, intrinsic.cy], [intrinsic.width * .3, intrinsic.height * .7]])
        points = np.column_stack([(uv[:, 0] - intrinsic.cx) / intrinsic.fx * 2,
                                  (uv[:, 1] - intrinsic.cy) / intrinsic.fy * 2, np.full(len(uv), 2.)])
        oracle = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic.matrix(), None)[0][:, 0]
        old = replace(intrinsic, cx=intrinsic.width/2, cy=intrinsic.height/2)
        analytic_rows.append({"intrinsics": asdict(intrinsic),
            "old_max_px": float(np.max(np.linalg.norm(projected_corner_pixels(points, old)-oracle, axis=1))),
            "full_k_max_px": float(np.max(np.linalg.norm(projected_corner_pixels(points, intrinsic)-oracle, axis=1)))})
        for index, point in enumerate(points):
            value = torch.tensor(point[None], dtype=torch.float32, device="cuda", requires_grad=True)
            corrected, legacy = raster(value, intrinsic), raster(value, old)
            raster_rows.append({"size": [intrinsic.width, intrinsic.height], "point": index,
                "oracle_corner_px": oracle[index].tolist(), "full_k_corner_px": corrected.detach().cpu().tolist(),
                "old_error_px": float(np.linalg.norm(legacy.detach().cpu().numpy()-oracle[index])),
                "full_k_error_px": float(np.linalg.norm(corrected.detach().cpu().numpy()-oracle[index]))})

    intrinsic = cases[0]
    value = torch.tensor([[0.13, -0.17, 2.]], dtype=torch.float32, device="cuda", requires_grad=True)
    centroid = raster(value, intrinsic)
    objective = centroid[0] + 0.37 * centroid[1]
    gradient = torch.autograd.grad(objective, value)[0].detach().cpu().numpy()[0]
    sweep = []
    for epsilon in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        finite = []
        for axis in range(3):
            delta = torch.zeros_like(value)
            delta[0, axis] = epsilon
            plus, minus = raster(value.detach()+delta, intrinsic), raster(value.detach()-delta, intrinsic)
            finite.append(float(((plus[0]+.37*plus[1])-(minus[0]+.37*minus[1])).item()/(2*epsilon)))
        relative = float(np.linalg.norm(gradient-np.asarray(finite))/np.linalg.norm(finite))
        sweep.append({"epsilon_m": epsilon, "finite_difference": finite, "relative_error": relative})
    # Require agreement at both smallest scales, rather than selecting a best step.
    gradient_relative = max(row["relative_error"] for row in sweep[-2:])
    plateau = float(np.linalg.norm(np.asarray(sweep[-1]["finite_difference"])-sweep[-2]["finite_difference"])
                    / np.linalg.norm(sweep[-1]["finite_difference"]))
    torch.cuda.synchronize()
    passed = (max(x["full_k_max_px"] for x in analytic_rows) <= limits["analytic_max_px"]
              and max(x["full_k_error_px"] for x in raster_rows) <= limits["raster_centroid_max_px"]
              and gradient_relative <= limits["gradient_relative_error"] and plateau <= 0.01)
    receipt = {"status": "passed" if passed else "failed", "scope": "analytic_and_cuda_projection_not_training_quality",
        "runtime": metadata, "camera_source": data_identity(source), "limits": limits,
        "pixel_convention": "COLMAP corner; raster integer-centred output +0.5",
        "source_convention": "https://github.com/colmap/colmap/blob/main/doc/faq.rst",
        "analytic": analytic_rows, "real_cuda_renders": raster_rows,
        "gradient": {"autograd": gradient.tolist(), "step_sweep": sweep,
                     "relative_error": gradient_relative, "smallest_step_agreement": plateau,
                     "protocol_revision": "v2_double_reduction_and_two_smallest_step_agreement",
                     "reason": "Opacity support is thresholded; a large difference step can cross discontinuities. Original failed receipt is retained."},
        "elapsed_seconds": time.monotonic()-start,
        "old_source_modified": False, "counted_as_usable_optimization": False}
    atomic_json(args.output/"PROJECTION_RECEIPT.json", receipt)
    print({"status": receipt["status"], "analytic_max_px": max(x["full_k_max_px"] for x in analytic_rows),
        "raster_max_px": max(x["full_k_error_px"] for x in raster_rows), "gradient_relative": gradient_relative})
    if not passed:
        raise RuntimeError("Projection validation did not meet the preregistered limits")


if __name__ == "__main__":
    main()
