"""Check actual fused SSIM maps, fallback errors, public shapes and output work."""

import os
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup020/g0"
os.environ["TRITON_CACHE_DIR"] = str(OUT.parent / "triton_cache")
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=180, maximum_bytes=8*1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        import triton
        from utils import loss_utils
        from interior_ssim_reuse import InteriorSSIM
        from fused_ssim_formula import FusedInteriorSSIM
        from ssim_output_census import output_census, new_trace
        torch.manual_seed(18231)
        before_rng = torch.random.get_rng_state().clone()
        cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
        traces = {m:new_trace() for m in ("raw","fused")}
        rows = []
        for number in range(16):
            device = "cpu" if number == 12 else "cuda"
            dtype = torch.float64 if number in (12,13) else torch.float32
            channels = 1 if number == 11 else 3
            h,w = (43,57) if number != 10 else (9,13)
            values = torch.arange(channels*h*w,device=device,dtype=dtype).reshape(1,channels,h,w)
            reference = (values.remainder(257)/256).clone()
            prediction = ((values*7).remainder(253)/252).clone()
            if number == 0: prediction.zero_(); reference.zero_()
            if number == 1: prediction.fill_(1); reference.fill_(.5)
            if number == 2: prediction.copy_(reference)
            if number == 3: prediction[0,0,0,0] = float('nan')
            if number == 4: prediction[0,0,0,0] = float('inf')
            if number == 5: prediction=prediction.transpose(2,3);reference=reference.transpose(2,3)
            if number == 6: prediction=prediction[:,:,:,::2];reference=reference[:,:,:,::2]
            if number == 7: prediction=prediction.to(memory_format=torch.channels_last);reference=reference.to(memory_format=torch.channels_last)
            window = 1 if number == 8 else (5 if number == 9 else 11)
            average = number != 14
            saved = [x.contiguous().view(torch.uint8).clone() for x in (prediction,reference)]
            result, errors, receipts = {}, {}, {}
            for mode, factory in (("raw",InteriorSSIM),("fused",FusedInteriorSSIM)):
                inner = factory(loss_utils.create_window, maximum_bytes=1024 if number == 15 else 1024**3)
                try:
                    with output_census(traces[mode]):
                        result[mode] = [inner(prediction,reference,window,average).clone()]
                        if channels == 3 and window == 11 and min(prediction.shape[2:]) > 20:
                            result[mode].append(inner(prediction[:,:,3:-3,4:-4],reference[:,:,3:-3,4:-4],window,average).clone())
                    receipts[mode] = inner.receipt()
                except BaseException as exc:
                    errors[mode] = (type(exc).__name__,str(exc))
                finally: inner.clear()
                if inner.receipt()['resident_accounted_bytes'] != 0: raise AssertionError('Owned cache remained')
            same = errors.get('raw') == errors.get('fused') and len(result.get('raw',[])) == len(result.get('fused',[]))
            for a,b in zip(result.get('raw',[]),result.get('fused',[])):
                same = same and torch.allclose(a,b,atol=1e-7,rtol=0,equal_nan=True)
            unchanged = all(torch.equal(x.contiguous().view(torch.uint8),y) for x,y in zip((prediction,reference),saved))
            rows.append({'case':number,'outputs_within_original1e7':bool(same),'input_bits_unchanged':unchanged,'errors':errors,'receipts':receipts})
            if not same or not unchanged: raise AssertionError('Original SSIM generic quality changed')
        errors=[]
        for kind in ('grad','mismatch','window','budget'):
            pair=[]
            for factory in (InteriorSSIM,FusedInteriorSSIM):
                try:
                    cache=factory(loss_utils.create_window,maximum_bytes=0 if kind=='budget' else 1024**3)
                    p=torch.ones((1,3,17,23),device='cuda',requires_grad=kind=='grad')
                    q=torch.ones((1,3,17,24 if kind=='mismatch' else 23),device='cuda')
                    cache(p,q,window_size=2 if kind=='window' else 11)
                except BaseException as e: pair.append((type(e).__name__,str(e)))
            errors.append({'kind':kind,'pair':pair,'same':len(pair)==2 and pair[0]==pair[1]})
        import sqlite3
        connection=sqlite3.connect('file:'+str(ROOT/'04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/database.db')+'?mode=ro',uri=True)
        names=connection.execute('SELECT image_id,name FROM images ORDER BY image_id').fetchall();connection.close()
        public=[]
        for identifier,name in names:
            v=torch.arange(3*17*29,device='cuda',dtype=torch.float32).reshape(1,3,17,29)
            p=((v+identifier).remainder(251)/250).clone();q=((v*3+identifier).remainder(257)/256).clone()
            outputs=[]
            for mode,factory in (('raw',InteriorSSIM),('fused',FusedInteriorSSIM)):
                cache=factory(loss_utils.create_window)
                with output_census(traces[mode]): outputs.append(cache(p,q))
                cache.clear()
            public.append({'id':identifier,'name':name,'same':bool(torch.allclose(*outputs,atol=1e-7,rtol=0))})
        torch.cuda.synchronize()
        paths=[Path(__file__),CODE/'fused_ssim_formula.py',CODE/'ssim_output_census.py',CODE/'interior_ssim_reuse.py',
               ROOT/'02_code/gaussian_pipeline_optimization_v2_20261005/reference_ssim.py',Path(triton.__file__)]
        checks={'all16_generic_inputs_quality':all(x['outputs_within_original1e7'] and x['input_bits_unchanged'] for x in rows),
                'all4_original_errors':all(x['same'] for x in errors),
                'public360_synthetic_component_original_tolerance':len(public)==360 and all(x['same'] for x in public),
                'global_RNG_unchanged':torch.equal(before_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())),
                'actual_CUDA_API_output_reduction':traces['fused']['successful_new_CUDA_output_elements'] < traces['raw']['successful_new_CUDA_output_elements'],
                'owned_CUDA_under4GiB_device_under12GiB_free2GiB':torch.cuda.max_memory_reserved()<=4*1024**3 and torch.cuda.mem_get_info()[0]>=2*1024**3}
        result={'checks':checks,'rows':rows,'errors':errors,'public_synthetic_nonblind_not_image_or_GT_pipeline':public,
                'trace':traces,'metadata':metadata,'triton_version':triton.__version__,'sources':[data_identity(p) for p in paths],
                'seconds':time.perf_counter()-STARTED,'new_optimization_count':0}
        atomic_json(OUT/'RESULT.json',result)
        print({k:result[k] for k in ('checks','seconds','triton_version')},flush=True)
        if not all(checks.values()):raise AssertionError('A fused SSIM source, quality, work or resource check failed')


if __name__=='__main__':main()
