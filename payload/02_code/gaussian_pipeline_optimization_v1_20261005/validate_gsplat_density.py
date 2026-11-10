"""Compare actual native and gsplat gradient and densification semantics."""

import argparse
import json
from pathlib import Path

import numpy as np

from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole
from gsplat_adapter import activate_gsplat, rasterize_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = activate_reference(args.runtime)
    candidate = activate_gsplat(args.build)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.colmap_loader import read_intrinsics_binary

    real = Pinhole.from_colmap(next(iter(read_intrinsics_binary(args.dataset / "sparse/0/cameras.bin").values())))
    limits = {"gradient_relative": .03, "pixel_mean_absolute": .0001, "pixel_max_absolute": .02,
              "density_radii_exact": True}
    atomic_json(args.output / "CONTRACT.json", {"limits": limits, "reference": reference, "candidate": candidate,
        "scope": "actual native dL/dNDC and 3-sigma bookkeeping compared with the gsplat adapter; RGB and SH degree3",
        "source": data_identity(Path(__file__)), "old_backend_not_modified": True})
    rows = []
    identity = torch.eye(4, device="cuda")
    for size in [(128, 96), (812, 620)]:
        intr = real.resized(*size)
        for sh_degree in [None, 3]:
            for background_values in [[0., 0., 0.], [.05, .03, .02]]:
                torch.manual_seed(18)
                originals = {
                    "means": torch.tensor([[.1, -.15, 2.], [-.2, .15, 2.5], [.08, -.12, 2.1]], device="cuda"),
                    "scales": torch.tensor([[.009, .006, .012], [.006, .014, .008], [.012, .007, .006]], device="cuda"),
                    "quats": torch.tensor([[1., .1, .05, 0.], [1., 0., .08, .07], [1., .03, -.02, .01]], device="cuda"),
                    "opacities": torch.tensor([[.65], [.5], [.8]], device="cuda"),
                    "colors": torch.rand((3, 3) if sh_degree is None else (3, 16, 3), device="cuda")*.3,
                }
                background = torch.tensor(background_values, device="cuda")
                outputs = {}
                for backend in ("native", "gsplat"):
                    values = {key: value.clone().requires_grad_() for key, value in originals.items()}
                    rotations = torch.nn.functional.normalize(values["quats"], dim=-1)
                    proxy = torch.zeros_like(values["means"], requires_grad=True)
                    if backend == "native":
                        settings = GaussianRasterizationSettings(image_height=intr.height, image_width=intr.width,
                            tanfovx=intr.width/(2*intr.fx), tanfovy=intr.height/(2*intr.fy), bg=background,
                            scale_modifier=1., viewmatrix=identity,
                            projmatrix=torch.tensor(intr.projection(), dtype=torch.float32, device="cuda").T.contiguous(),
                            sh_degree=sh_degree or 0, campos=torch.zeros(3, device="cuda"), prefiltered=False,
                            debug=False, antialiasing=False)
                        image, radii, _ = GaussianRasterizer(settings)(means3D=values["means"], means2D=proxy,
                            opacities=values["opacities"], scales=values["scales"], rotations=rotations,
                            colors_precomp=values["colors"] if sh_degree is None else None,
                            shs=values["colors"] if sh_degree is not None else None)
                    else:
                        image, radii, _ = rasterize_values(values["means"], values["scales"], rotations,
                            values["opacities"], values["colors"], torch.tensor(intr.matrix(), dtype=torch.float32, device="cuda"),
                            identity, intr.width, intr.height, background, sh_degree=sh_degree, screen_proxy=proxy)
                    x = torch.linspace(.1, 1., intr.width, device="cuda")[None, None, :]
                    y = torch.linspace(.15, .8, intr.height, device="cuda")[None, :, None]
                    weight = torch.sin(5*x) + torch.cos(4*y) + x*y
                    objective = (image*weight).mean() + image.square().mean()*.13
                    objective.backward()
                    if proxy.grad is None:
                        raise AssertionError("Actual density gradient was not populated")
                    outputs[backend] = {"image": image.detach(), "radii": radii.detach(),
                        "density": proxy.grad.detach(), "gradients": {key: value.grad.detach() for key, value in values.items()}}
                old, new = outputs["native"], outputs["gsplat"]
                difference = (old["image"]-new["image"]).abs()
                gradients = {key: float(torch.linalg.vector_norm(new["gradients"][key]-old["gradients"][key])/
                    torch.linalg.vector_norm(old["gradients"][key]).clamp_min(1e-10)) for key in originals}
                density_delta = float(torch.linalg.vector_norm(new["density"]-old["density"])/
                    torch.linalg.vector_norm(old["density"]).clamp_min(1e-10))
                checks = {"parameter_gradients": max(gradients.values()) <= limits["gradient_relative"],
                    "density_gradient": density_delta <= limits["gradient_relative"],
                    "radii_exact": torch.equal(old["radii"], new["radii"]),
                    "pixels": float(difference.mean()) <= limits["pixel_mean_absolute"] and float(difference.max()) <= limits["pixel_max_absolute"]}
                rows.append({"size": size, "sh_degree": sh_degree, "background": background_values,
                    "gradient_relative_errors": gradients, "density_gradient_relative_error": density_delta,
                    "native_radii": old["radii"].tolist(), "gsplat_bookkeeping_radii": new["radii"].tolist(),
                    "pixel_mean_absolute_error": float(difference.mean()), "pixel_max_absolute_error": float(difference.max()),
                    "checks": checks, "passed": all(checks.values())})
    result = {"status": "passed" if all(row["passed"] for row in rows) else "failed", "rows": rows,
              "candidate_adapter_sha256": candidate["adapter"]["sha256"],
              "limits": limits, "counted": False, "training_quality_claim": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("The candidate did not preserve the registered gradient and density guards")


if __name__ == "__main__":
    main()
