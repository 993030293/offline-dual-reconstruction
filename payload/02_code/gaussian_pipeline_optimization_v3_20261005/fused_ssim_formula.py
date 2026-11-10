"""Fuse only the original evaluation SSIM algebra without changing reductions."""

import inspect

import torch
import triton
import triton.language as tl

import reference_ssim
from interior_ssim_reuse import InteriorSSIM


@triton.jit
def _map_kernel(mp, mr, rs, ps, vr, cr, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    p = tl.load(mp+index, mask, other=0)
    r = tl.load(mr+index, mask, other=0)
    rr = tl.load(rs+index, mask, other=0)
    pp = p*p
    product = p*r
    vp = tl.load(ps+index, mask, other=0)-pp
    cov = tl.load(cr+index, mask, other=0)-product
    variance = tl.load(vr+index, mask, other=0)
    numerator = (2*product+.01**2)*(2*cov+.03**2)
    denominator = (pp+rr+.01**2)*(vp+variance+.03**2)
    value = tl.div_rn(numerator, denominator)
    tl.store(out+index, value, mask)


def formula(mu_pred, mu_ref, mu_ref_squared, pred_second, variance_ref, cross):
    values = (mu_pred, mu_ref, mu_ref_squared, pred_second, variance_ref, cross)
    supported = all(x.device.type == "cuda" and x.dtype == torch.float32 and x.is_contiguous()
                    and not x.requires_grad and x.shape == mu_pred.shape and x.device == mu_pred.device
                    for x in values) and mu_pred.numel() > 0
    if supported:
        result = torch.empty_like(mu_pred)
        _map_kernel[(triton.cdiv(mu_pred.numel(), 256),)](*values, result, mu_pred.numel(),
                                                       BLOCK=256, enable_fp_fusion=False)
        return result
    mu_pred_squared = mu_pred.pow(2)
    mean_product = mu_pred*mu_ref
    variance_pred = pred_second-mu_pred_squared
    covariance = cross-mean_product
    return ((2*mean_product+.01**2)*(2*covariance+.03**2))/(
        (mu_pred_squared+mu_ref_squared+.01**2)*(variance_pred+variance_ref+.03**2))


def map_factory(window_factory, maximum_bytes):
    source = inspect.getsource(reference_ssim.ReferenceSSIM)
    anchor = """        mu_pred_squared = mu_pred.pow(2)
        mean_product = mu_pred*mu_ref
        variance_pred = F.conv2d(prediction*prediction, window, padding=window_size//2, groups=channels)-mu_pred_squared
        covariance = F.conv2d(prediction*reference, window, padding=window_size//2, groups=channels)-mean_product
        result = ((2*mean_product+.01**2)*(2*covariance+.03**2))/(
            (mu_pred_squared+mu_ref_squared+.01**2)*(variance_pred+variance_ref+.03**2))"""
    return_anchor = "return result.mean() if size_average else result.mean(1).mean(1).mean(1)"
    if source.count(anchor) != 1 or source.count(return_anchor) != 1 or source.count("class ReferenceSSIM:") != 1:
        raise ValueError("The original complete SSIM formula source changed")
    replacement = """        pred_second = F.conv2d(prediction*prediction, window, padding=window_size//2, groups=channels)
        cross = F.conv2d(prediction*reference, window, padding=window_size//2, groups=channels)
        result = _formula(mu_pred, mu_ref, mu_ref_squared, pred_second, variance_ref, cross)"""
    source = source.replace(anchor, replacement).replace(return_anchor, "return result")
    source = source.replace("class ReferenceSSIM:", "class _FusedMapReferenceSSIM:")
    namespace = dict(vars(reference_ssim), _formula=formula)
    exec(compile(source, "<private-source-bound-fused-SSIM-map>", "exec"), namespace)
    return namespace["_FusedMapReferenceSSIM"](window_factory, maximum_bytes=maximum_bytes)


class FusedInteriorSSIM(InteriorSSIM):
    def __init__(self, window_factory, maximum_bytes=1024**3):
        super().__init__(window_factory, maximum_bytes)
        self.cache = map_factory(window_factory, maximum_bytes)
