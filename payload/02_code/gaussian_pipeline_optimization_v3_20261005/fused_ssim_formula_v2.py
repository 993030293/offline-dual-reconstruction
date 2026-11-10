"""Preserve mixed dense NCHW and channels-last SSIM statistics in one kernel."""

import inspect

import torch
import triton
import triton.language as tl

import fused_ssim_formula as original
from interior_ssim_reuse import InteriorSSIM


@triton.jit
def _offset(index, C, H, W, strides):
    x=index%W
    y=(index//W)%H
    c=(index//(W*H))%C
    n=index//(W*H*C)
    return n*strides[0]+c*strides[1]+y*strides[2]+x*strides[3]


@triton.jit(do_not_specialize=["N","C","H","W","STRIDES"])
def _map_kernel(mp,mr,rs,ps,vr,cr,out,N,C,H,W,STRIDES,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=index<N
    p=tl.load(mp+_offset(index,C,H,W,STRIDES[0]),mask,other=0)
    r=tl.load(mr+_offset(index,C,H,W,STRIDES[1]),mask,other=0)
    rr=tl.load(rs+_offset(index,C,H,W,STRIDES[2]),mask,other=0)
    pp=p*p
    product=p*r
    vp=tl.load(ps+_offset(index,C,H,W,STRIDES[3]),mask,other=0)-pp
    cov=tl.load(cr+_offset(index,C,H,W,STRIDES[5]),mask,other=0)-product
    variance=tl.load(vr+_offset(index,C,H,W,STRIDES[4]),mask,other=0)
    numerator=(2*product+.01**2)*(2*cov+.03**2)
    denominator=(pp+rr+.01**2)*(vp+variance+.03**2)
    tl.store(out+_offset(index,C,H,W,STRIDES[6]),tl.div_rn(numerator,denominator),mask)


def formula(mu_pred,mu_ref,mu_ref_squared,pred_second,variance_ref,cross):
    values=(mu_pred,mu_ref,mu_ref_squared,pred_second,variance_ref,cross)
    supported=mu_pred.ndim==4 and mu_pred.numel()>0 and all(
        x.device.type=='cuda' and x.dtype==torch.float32 and x.device==mu_pred.device and x.shape==mu_pred.shape
        and not x.requires_grad and (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
        for x in values)
    if not supported:return original.formula(*values)
    result=torch.empty_like(mu_pred)
    strides=tuple(tuple(x.stride()) for x in (*values,result))
    _map_kernel[(triton.cdiv(mu_pred.numel(),256),)](*values,result,mu_pred.numel(),*mu_pred.shape[1:],
        strides,BLOCK=256,enable_fp_fusion=False)
    return result


def map_factory(window_factory,maximum_bytes):
    namespace=dict(vars(original),formula=formula)
    exec(compile(inspect.getsource(original.map_factory),'<private-source-bound-layout-fused-map>','exec'),namespace)
    return namespace['map_factory'](window_factory,maximum_bytes)


class FusedInteriorSSIM(InteriorSSIM):
    def __init__(self,window_factory,maximum_bytes=1024**3):
        super().__init__(window_factory,maximum_bytes)
        self.cache=map_factory(window_factory,maximum_bytes)
