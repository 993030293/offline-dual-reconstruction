"""Copy source-bound private compiled artifacts without overwriting any origin."""

from pathlib import Path
import shutil
import time

import run_fused_ssim_campaign as campaign
from runtime import atomic_json, data_identity

OWN=campaign.OWN


def main():
    started=time.perf_counter()
    target=OWN.parent/'triton_v2_shared_cache'
    target.mkdir(parents=True,exist_ok=False)
    origins=[OWN/f'c1/p{i}/interior/triton_cache' for i in (0,1)]
    rows=[]
    for origin in origins:
        for path in sorted(origin.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(origin);destination=target/relative
            identity=data_identity(path)
            if destination.exists():
                if data_identity(destination)['sha256']!=identity['sha256']:
                    raise ValueError('A private compiled-cache member disagreed between equivalent source kernels')
                copied=False
            else:
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination);copied=True
            if data_identity(destination)['sha256']!=identity['sha256']:raise ValueError('A complete compiled-cache copy changed')
            rows.append({'origin':identity,'copy':data_identity(destination),'copied':copied})
    if not rows:raise ValueError('No actual original compiled artifacts existed')
    atomic_json(OWN.parent/'COMPILED_CACHE_PREPARATION.json',{'rows':rows,'seconds':time.perf_counter()-started,
        'origin_frozen_V2_kernel':data_identity(campaign.CODE/'fused_ssim_formula_v2.py'),
        'private_output':str(target),'no_cached_scientific_result_or_input_data':True,'new_optimization_count':0})
    print({'copied_members':sum(x['copied'] for x in rows),'source_members':len(rows),'seconds':time.perf_counter()-started},flush=True)


if __name__=='__main__':main()
