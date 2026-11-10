"""Bridge pinned gsplat to the existing metric cameras and INRIA training loop."""

import json
from pathlib import Path
import sys

from runtime import data_identity


def activate_gsplat(build):
    build = Path(build).resolve()
    result = json.loads((build / "RESULT.json").read_text())
    if result["status"] != "built_pending_projection_and_training":
        raise ValueError("gsplat did not complete the isolated build")
    modules = list((build / "lib/gsplat").glob("csrc*.so"))
    if len(modules) != 1 or data_identity(modules[0])["sha256"] != result["modules"][0]["sha256"]:
        raise ValueError("Compiled gsplat identity differs from its build receipt")
    if "gsplat" in sys.modules:
        raise RuntimeError("Select the isolated gsplat library before importing it")
    sys.path[:0] = [str(build / "lib"), str(build / "dependencies")]
    import gsplat
    from gsplat import csrc
    from gsplat.cuda import _backend
    if gsplat.__version__ != "1.5.3" or _backend._C is not csrc:
        raise RuntimeError("Expected the pinned precompiled backend, never an automatic JIT fallback")
    return {"build": data_identity(build / "RESULT.json"), "module": data_identity(modules[0]),
            "adapter": data_identity(Path(__file__)),
            "python_sources": [data_identity(path) for path in sorted((build / "lib/gsplat").rglob("*.py"))],
            "coordinate_normalization": False, "batch_size": 1, "packed": False,
            "sparse_gradients": False, "antialiasing": False, "radius_clip": 0.0,
            "depth_supervision_supported": False, "dataset_split_changed": False}


def rasterize_values(means, scales, rotations, opacities, colors, intrinsics, view, width, height,
                     background, sh_degree=None, screen_proxy=None):
    import torch
    from gsplat import rasterization

    rendered, alpha, info = rasterization(means=means, quats=rotations, scales=scales,
        opacities=opacities.reshape(-1), colors=colors, viewmats=view[None], Ks=intrinsics[None],
        width=width, height=height, near_plane=.2, far_plane=1e10, radius_clip=0.0, eps2d=.3,
        sh_degree=sh_degree, packed=False, backgrounds=background[None], render_mode="RGB",
        sparse_grad=False, absgrad=False, rasterize_mode="classic", camera_model="pinhole")
    if screen_proxy is not None and info["means2d"].requires_grad:
        def preserve_ndc_density_gradient(gradient):
            # INRIA's densifier consumes dL/dNDC, whereas gsplat exposes dL/dpixel.
            value = torch.zeros_like(screen_proxy)
            value[:, :2] = gradient.detach()[0] * gradient.new_tensor([width/2, height/2])
            screen_proxy.grad = value if screen_proxy.grad is None else screen_proxy.grad + value
            return gradient
        info["means2d"].register_hook(preserve_ndc_density_gradient)
    with torch.no_grad():
        conics = info["conics"][0].detach()
        determinant = conics[:, 0]*conics[:, 2]-conics[:, 1].square()
        visible = (info["radii"][0] > 0).all(dim=-1) & (determinant > 0)
        safe_det = torch.where(visible, determinant, torch.ones_like(determinant))
        xx, xy, yy = conics[:, 2]/safe_det, -conics[:, 1]/safe_det, conics[:, 0]/safe_det
        midpoint = .5*(xx+yy)
        discriminant = torch.clamp(midpoint.square()-(xx*yy-xy.square()), min=.1).sqrt()
        radii = torch.ceil(3*torch.clamp(midpoint+discriminant, min=0).sqrt()).to(torch.int32)
        radii = torch.where(visible, radii, torch.zeros_like(radii))
    return rendered[0].permute(2, 0, 1), radii, info


def render(camera, gaussian, pipeline, background, scaling_modifier=1.0, separate_sh=False,
           override_color=None, use_trained_exp=False):
    import torch

    if (use_trained_exp or separate_sh or pipeline.antialiasing or pipeline.compute_cov3D_python
            or pipeline.convert_SHs_python or not hasattr(camera, "calibrated_intrinsics")):
        raise ValueError("This first gsplat adapter requires the frozen full-K dense RGB configuration")
    intr = camera.calibrated_intrinsics
    key = (intr["fx"], intr["fy"], intr["cx"], intr["cy"], str(gaussian.get_xyz.device))
    if getattr(camera, "_gsplat_intrinsics_key", None) != key:
        camera._gsplat_intrinsics = gaussian.get_xyz.new_tensor([
            [intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
        camera._gsplat_intrinsics_key = key
    proxy = torch.zeros_like(gaussian.get_xyz, requires_grad=torch.is_grad_enabled())
    image, radii, _ = rasterize_values(gaussian.get_xyz, gaussian.get_scaling*scaling_modifier,
        gaussian.get_rotation, gaussian.get_opacity,
        gaussian.get_features if override_color is None else override_color,
        camera._gsplat_intrinsics, camera.world_view_transform.T.contiguous(), camera.image_width, camera.image_height,
        background, sh_degree=gaussian.active_sh_degree if override_color is None else None, screen_proxy=proxy)
    return {"render": image.clamp(0, 1), "viewspace_points": proxy,
            "visibility_filter": (radii > 0).nonzero(), "radii": radii, "depth": None}
