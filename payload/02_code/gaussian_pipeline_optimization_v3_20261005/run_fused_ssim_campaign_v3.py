"""Run new complete pairs using one explicitly charged private compiled cache."""

import inspect
from pathlib import Path

import run_fused_ssim_campaign as original
import run_fused_ssim_campaign_v2 as previous
from runtime import data_identity


def identities():
    sources,inputs=previous.identities()
    paths=[Path(__file__),original.CODE/'evaluate_fused_ssim_v3.py',original.CODE/'fault_fused_ssim_v3.py',
        original.CODE/'fused_ssim_node_v3.py',original.CODE/'prepare_fused_cache.py',original.CODE/'prepare_fused_cache_v2.py']
    return sources+[data_identity(p) for p in paths],inputs


namespace=dict(vars(original),__file__=str(Path(__file__)))
source=inspect.getsource(original.invoke).replace("'evaluate_fused_ssim.py'","'evaluate_fused_ssim_v3.py'").replace(
    "'fault_fused_ssim.py'","'fault_fused_ssim_v3.py'").replace("'fused_ssim_node.py'","'fused_ssim_node_v3.py'")
exec(compile(source,str(Path(__file__)),'exec'),namespace)
invoke=namespace['invoke']


def main():
    preparation=original.original.read(original.OWN.parent/'COMPILED_CACHE_PREPARATION_V2.json')
    if preparation['origin_frozen_V2_kernel']!=data_identity(original.CODE/'fused_ssim_formula_v2.py'):
        raise ValueError('The compiler cache kernel source identity changed')
    source=inspect.getsource(original.main).replace("out=OWN/'c0'","out=OWN/'c2'").replace("OWN.parent/'g0/RESULT.json'","OWN.parent/'g1/RESULT.json'")
    namespace=dict(vars(original),identities=identities,invoke=invoke,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
