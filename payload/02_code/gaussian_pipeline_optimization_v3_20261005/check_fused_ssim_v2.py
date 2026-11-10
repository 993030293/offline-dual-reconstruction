"""Repeat generic checks and verify complete maps and reference mutation."""

import inspect
from pathlib import Path

import check_fused_ssim_020 as original


def main():
    source=inspect.getsource(original.main)
    source=source.replace('from fused_ssim_formula import FusedInteriorSSIM','from fused_ssim_formula_v2 import FusedInteriorSSIM')
    source=source.replace("CODE/'fused_ssim_formula.py'","CODE/'fused_ssim_formula.py',CODE/'fused_ssim_formula_v2.py'")
    anchor='        import sqlite3'
    extra='''        from interior_ssim_reuse import map_reference_factory
        from fused_ssim_formula_v2 import map_factory
        map_rows=[]
        for number in range(8):
            values=torch.arange(3*47*61,device='cuda',dtype=torch.float32).reshape(1,3,47,61)
            p=(values.remainder(257)/256).clone()
            q=((values*7).remainder(253)/252).clone()
            if number&1:p=p.to(memory_format=torch.channels_last)
            if number&2:q=q.to(memory_format=torch.channels_last)
            caches=[factory(loss_utils.create_window,1024**3) for factory in (map_reference_factory,map_factory)]
            for stage in range(3):
                if stage==1:q.add_(.03125)
                if stage==2:p.mul_(.875)
                outputs=[]
                for mode,cache in zip(('raw','fused'),caches):
                    with output_census(traces[mode]):outputs.append(cache(p,q))
                same=torch.equal(outputs[0].contiguous().view(torch.uint8),outputs[1].contiguous().view(torch.uint8))
                layout=outputs[0].stride()==outputs[1].stride() and outputs[0].dtype==outputs[1].dtype
                map_rows.append({'case':number,'stage':stage,'all_map_bits_equal':same,'original_layout':layout})
                if not same or not layout:raise AssertionError('Original complete map bits or layout changed')
            for cache in caches:cache.clear()
'''
    if source.count(anchor)!=1:raise ValueError('The original generic component changed')
    source=source.replace(anchor,extra+anchor)
    source=source.replace("'all16_generic_inputs_quality':", "'all24_complete_map_bits_layout_and_reference_mutation':all(x['all_map_bits_equal'] and x['original_layout'] for x in map_rows), 'all16_generic_inputs_quality':")
    source=source.replace("'checks':checks,'rows':rows", "'checks':checks,'map_rows':map_rows,'rows':rows")
    namespace=dict(original.__dict__,OUT=original.OUT.parent/'g1',__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
