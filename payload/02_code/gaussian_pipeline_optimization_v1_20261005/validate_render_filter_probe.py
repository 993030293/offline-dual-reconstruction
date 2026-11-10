"""Audit actual browser alpha against analytic EWA and native CUDA single-Gaussian renders."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from full_intrinsics import Pinhole
from runtime import activate_reference, atomic_json, data_identity


def covariance(decoded):
    x, y, z, w = decoded["quaternion_xyzw"]
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    axes = np.diag([1., -1., -1.])
    return axes @ rotation @ np.diag(np.square(decoded["scales"])) @ rotation.T @ axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "probe", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = json.loads(args.probe.read_text())
    if (source["status"] != "rendered_pending_independent_oracle" or len(source["results"]) != 12
            or not source["drawingBufferAlpha"]):
        raise ValueError("The complete registered browser alpha probe is required")
    metadata = activate_reference(args.runtime)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = "cuda"
    identity = torch.eye(4, device=device)
    rows = []
    for index, case in enumerate(source["results"]):
        if case["index"] != index or len(case["variants"]) != 2:
            raise ValueError("Probe ordering or variant denominator changed")
        intrinsic = Pinhole(**{key: case[key] for key in ("width", "height", "fx", "fy", "cx", "cy")})
        center = np.asarray(case["decoded"]["center"])*[1., -1., -1.]
        xyz_cov = covariance(case["decoded"])
        x, y, z = center
        jacobian = np.array([[intrinsic.fx/z, 0., -intrinsic.fx*x/z**2],
                             [0., intrinsic.fy/z, -intrinsic.fy*y/z**2]])
        cov = jacobian @ xyz_cov @ jacobian.T + .3*np.eye(2)
        projected = np.array([intrinsic.fx*x/z+intrinsic.cx, intrinsic.fy*y/z+intrinsic.cy])
        yy, xx = np.mgrid[:intrinsic.height, :intrinsic.width]
        delta = np.stack([xx+.5-projected[0], yy+.5-projected[1]], axis=-1)
        exponent = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov), delta)
        central = exponent <= 4
        oracle = case["decoded"]["opacity"]*np.exp(-.5*exponent)
        if central.sum() < 1:
            raise ValueError("No central pixel in a registered probe")
        settings = GaussianRasterizationSettings(image_height=intrinsic.height, image_width=intrinsic.width,
            tanfovx=intrinsic.width/(2*intrinsic.fx), tanfovy=intrinsic.height/(2*intrinsic.fy),
            bg=torch.zeros(3, device=device), scale_modifier=1., viewmatrix=identity,
            projmatrix=torch.tensor(intrinsic.projection(), device=device, dtype=torch.float32).T.contiguous(),
            sh_degree=0, campos=torch.zeros(3, device=device), prefiltered=False, debug=False, antialiasing=False)
        packed_cov = xyz_cov[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
        with torch.no_grad():
            rendered = GaussianRasterizer(settings)(
                means3D=torch.tensor(center[None], dtype=torch.float32, device=device),
                means2D=torch.zeros((1, 3), device=device),
                colors_precomp=torch.ones((1, 3), device=device),
                opacities=torch.tensor([[case["decoded"]["opacity"]]], dtype=torch.float32, device=device),
                cov3D_precomp=torch.tensor(packed_cov[None], dtype=torch.float32, device=device))[0][0].cpu().numpy()
        row = {"index": index, "intrinsics": asdict(intrinsic), "decoded": case["decoded"],
            "central_pixels": int(central.sum()), "native_analytic_max_error": float(abs(rendered-oracle)[central].max()),
            "variants": []}
        for variant in case["variants"]:
            alpha = np.asarray(variant["alpha"], dtype=np.float64).reshape(intrinsic.height, intrinsic.width)/255
            row["variants"].append({"label": variant["label"], "preBlurAmount": variant["preBlurAmount"],
                "blurAmount": variant["blurAmount"],
                "analytic_max_error": float(abs(alpha-oracle)[central].max()),
                "native_max_error": float(abs(alpha-rendered)[central].max())})
        rows.append(row)
    checks = {
        "complete12cases": len(rows) == 12,
        "native_analytic_oracle": all(row["native_analytic_max_error"] <= 2/255 for row in rows),
        "actual_classic_uniforms": all(row["variants"][1]["label"] == "model_classic"
            and row["variants"][1]["preBlurAmount"] == .3 and row["variants"][1]["blurAmount"] == 0 for row in rows),
        "new_analytic_and_native_alpha": all(max(row["variants"][1]["analytic_max_error"],
            row["variants"][1]["native_max_error"]) <= 2/255 for row in rows),
        "old_default_has_large_failure": any(row["variants"][0]["analytic_max_error"] > 10/255 for row in rows),
    }
    receipt = {"status": "numeric_passed_actual_model_and_navigation_pending" if all(checks.values()) else "failed_numeric_oracle",
        "checks": checks, "limits": {"new_max_alpha_error": 2/255, "old_failure_min_alpha_error": 10/255},
        "rows": rows, "runtime": metadata, "input": data_identity(args.probe), "source": data_identity(Path(__file__)),
        "elapsed_seconds": time.monotonic()-started, "counted": False,
        "scope": "central_single_Gaussian_alpha_not_full_scene_RGB_or_GT_accuracy"}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": checks}))
    if not all(checks.values()):
        raise RuntimeError("The preregistered renderer filter oracle failed")


if __name__ == "__main__":
    main()
