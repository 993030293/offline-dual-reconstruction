"""Keep abnormal drainage restricted to the exact owned V3 Linux entries."""

import inspect
from pathlib import Path

import fused_ssim_node as original


def compose():
    function=original.compose()
    source=inspect.getsource(original.original.linux_identity_check)
    before='("parallel_a_003/owned_matching_worker.py",)'
    if source.count(before)!=1:raise ValueError('The original Linux drain boundary changed')
    source=source.replace(before,'("evaluate_fused_ssim_v3.py", "fault_fused_ssim_v3.py")')
    namespace=function.__globals__;namespace['__file__']=str(Path(__file__))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    return function


if __name__=='__main__':compose()()
