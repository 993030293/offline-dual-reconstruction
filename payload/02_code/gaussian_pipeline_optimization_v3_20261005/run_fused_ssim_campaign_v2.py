"""Run fresh complete V2 pairs after the preserved V1 unsupported-layout failure."""

import inspect
from pathlib import Path
from types import FunctionType

import run_fused_ssim_campaign as original
from runtime import data_identity


def identities():
    sources,inputs=original.identities()
    paths=[Path(__file__),original.CODE/'evaluate_fused_ssim_v2.py',original.CODE/'fused_ssim_formula_v2.py',
        original.CODE/'check_fused_ssim_v2.py',original.CODE/'fault_fused_ssim_v2.py',original.CODE/'fused_ssim_node_v2.py']
    return sources+[data_identity(p) for p in paths],inputs


invoke_namespace=dict(vars(original),__file__=str(Path(__file__)))
invoke_source=inspect.getsource(original.invoke).replace("'evaluate_fused_ssim.py'","'evaluate_fused_ssim_v2.py'").replace("'fault_fused_ssim.py'","'fault_fused_ssim_v2.py'").replace("'fused_ssim_node.py'","'fused_ssim_node_v2.py'")
exec(compile(invoke_source,str(Path(__file__)),'exec'),invoke_namespace)
invoke=invoke_namespace['invoke']


def main():
    source=inspect.getsource(original.main).replace("out=OWN/'c0'","out=OWN/'c1'").replace("OWN.parent/'g0/RESULT.json'","OWN.parent/'g1/RESULT.json'")
    namespace=dict(vars(original),identities=identities,invoke=invoke,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
