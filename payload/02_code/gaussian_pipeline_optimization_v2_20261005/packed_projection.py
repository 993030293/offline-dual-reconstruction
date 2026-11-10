"""Scatter packed gsplat observations into the preserved dense densifier identity."""

from contextlib import contextmanager


def rasterize_packed(means, scales, rotations, opacities, colors, intrinsics, view, width, height,
                     background, sh_degree=None, screen_proxy=None):
    import torch
    from gsplat import rasterization

    rendered, alpha, info = rasterization(means=means, quats=rotations, scales=scales,
        opacities=opacities.reshape(-1), colors=colors, viewmats=view[None], Ks=intrinsics[None],
        width=width, height=height, near_plane=.2, far_plane=1e10, radius_clip=0.0, eps2d=.3,
        sh_degree=sh_degree, packed=True, backgrounds=None, render_mode="RGB",
        sparse_grad=False, absgrad=False, rasterize_mode="classic", camera_model="pinhole")
    # gsplat 1.5.3 checks packed backgrounds against a missing image dimension.
    rendered = rendered+(1-alpha)*background
    ids = info["gaussian_ids"]
    if ids.ndim != 1 or info["means2d"].shape != (len(ids), 2):
        raise ValueError("The pinned single-camera packed projection layout changed")
    if screen_proxy is not None and info["means2d"].requires_grad:
        def preserve_ndc_density_gradient(gradient):
            value = torch.zeros_like(screen_proxy)
            value[:, :2].index_add_(0, ids, gradient.detach()*gradient.new_tensor([width/2, height/2]))
            screen_proxy.grad = value if screen_proxy.grad is None else screen_proxy.grad+value
            return gradient
        info["means2d"].register_hook(preserve_ndc_density_gradient)
    with torch.no_grad():
        conics = info["conics"].detach()
        determinant = conics[:, 0]*conics[:, 2]-conics[:, 1].square()
        visible = (info["radii"] > 0).all(dim=-1) & (determinant > 0)
        safe_det = torch.where(visible, determinant, torch.ones_like(determinant))
        xx, xy, yy = conics[:, 2]/safe_det, -conics[:, 1]/safe_det, conics[:, 0]/safe_det
        midpoint = .5*(xx+yy)
        discriminant = torch.clamp(midpoint.square()-(xx*yy-xy.square()), min=.1).sqrt()
        local = torch.ceil(3*torch.clamp(midpoint+discriminant, min=0).sqrt()).to(torch.int32)
        local = torch.where(visible, local, torch.zeros_like(local))
        radii = torch.zeros(len(means), device=means.device, dtype=torch.int32)
        radii.index_copy_(0, ids, local)
    return rendered[0].permute(2, 0, 1), radii, info


@contextmanager
def packed_projection(enabled=True):
    import gsplat_adapter
    original = gsplat_adapter.rasterize_values
    if enabled:
        gsplat_adapter.rasterize_values = rasterize_packed
    try:
        yield
    finally:
        gsplat_adapter.rasterize_values = original
