"""Fuse only the original dense detached conic-to-radius arithmetic."""

from contextlib import contextmanager
import inspect

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def radius_kernel(C, R, OUTPUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    mask = i < N
    a = tl.load(C+3*i, mask, other=0.)
    b = tl.load(C+3*i+1, mask, other=0.)
    c = tl.load(C+3*i+2, mask, other=0.)
    r0 = tl.load(R+2*i, mask, other=0)
    r1 = tl.load(R+2*i+1, mask, other=0)
    determinant = a*c-b*b
    visible = (r0 > 0) & (r1 > 0) & (determinant > 0)
    safe = tl.where(visible, determinant, 1.)
    xx = tl.div_rn(c, safe)
    xy = tl.div_rn(-b, safe)
    yy = tl.div_rn(a, safe)
    midpoint = .5*(xx+yy)
    discriminant = libdevice.sqrt_rn(tl.maximum(midpoint*midpoint-(xx*yy-xy*xy), .1, propagate_nan=tl.PropagateNan.ALL))
    radii = tl.ceil(3*libdevice.sqrt_rn(tl.maximum(midpoint+discriminant, 0., propagate_nan=tl.PropagateNan.ALL))).to(tl.int32)
    tl.store(OUTPUT+i, tl.where(visible, radii, 0), mask)


def native_radii(conics, original_radii):
    with torch.no_grad():
        determinant = conics[:, 0]*conics[:, 2]-conics[:, 1].square()
        visible = (original_radii > 0).all(dim=-1) & (determinant > 0)
        safe_det = torch.where(visible, determinant, torch.ones_like(determinant))
        xx, xy, yy = conics[:, 2]/safe_det, -conics[:, 1]/safe_det, conics[:, 0]/safe_det
        midpoint = .5*(xx+yy)
        discriminant = torch.clamp(midpoint.square()-(xx*yy-xy.square()), min=.1).sqrt()
        radii = torch.ceil(3*torch.clamp(midpoint+discriminant, min=0).sqrt()).to(torch.int32)
        return torch.where(visible, radii, torch.zeros_like(radii))


def admission(conics, original_radii):
    return (conics.is_cuda and original_radii.is_cuda and conics.device == original_radii.device
        and conics.dtype == torch.float32 and original_radii.dtype == torch.int32
        and conics.ndim == 2 and conics.shape[1] == 3 and original_radii.shape == (len(conics), 2)
        and conics.is_contiguous() and original_radii.is_contiguous())


def fused_radii(conics, original_radii):
    if not admission(conics, original_radii):
        raise ValueError("Select the original radius route before execution for unsupported tensor metadata")
    output = torch.empty(len(conics), dtype=torch.int32, device=conics.device)
    if len(conics):
        radius_kernel[(triton.cdiv(len(conics), 256),)](conics, original_radii, output, len(conics),
            BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


@contextmanager
def fused_radius_adapter(adapter, enabled=True):
    original = adapter.rasterize_values
    source = inspect.getsource(original)
    before = """        conics = info[\"conics\"][0].detach()
        determinant = conics[:, 0]*conics[:, 2]-conics[:, 1].square()
        visible = (info[\"radii\"][0] > 0).all(dim=-1) & (determinant > 0)
        safe_det = torch.where(visible, determinant, torch.ones_like(determinant))
        xx, xy, yy = conics[:, 2]/safe_det, -conics[:, 1]/safe_det, conics[:, 0]/safe_det
        midpoint = .5*(xx+yy)
        discriminant = torch.clamp(midpoint.square()-(xx*yy-xy.square()), min=.1).sqrt()
        radii = torch.ceil(3*torch.clamp(midpoint+discriminant, min=0).sqrt()).to(torch.int32)
        radii = torch.where(visible, radii, torch.zeros_like(radii))
"""
    if source.count(before) != 1:
        raise ValueError("The pinned original dense radius formula changed")
    after = """        conics = info[\"conics\"][0].detach()
        radii = fused_radii(conics, info[\"radii\"][0]) if radius_admission(conics, info[\"radii\"][0]) else native_radii(conics, info[\"radii\"][0])
"""
    namespace = dict(adapter.__dict__, fused_radii=fused_radii, native_radii=native_radii, radius_admission=admission)
    exec(compile(source.replace(before, after), "<isolated_original_dense_radius_fusion>", "exec"), namespace)
    if enabled:
        adapter.rasterize_values = namespace["rasterize_values"]
    try:
        yield
    finally:
        adapter.rasterize_values = original
