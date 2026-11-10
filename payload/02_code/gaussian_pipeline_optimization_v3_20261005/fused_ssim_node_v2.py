"""Restrict abnormal Linux drainage to the exact owned V2 entry paths."""

import inspect
from pathlib import Path

import fused_ssim_node as original


def compose():
    function=original.compose()
    source=inspect.getsource(original.original.linux_identity_check)
    before='("parallel_a_003/owned_matching_worker.py",)'
    if source.count(before)!=1:raise ValueError('The original owned Linux drain boundary changed')
    source=source.replace(before,'("evaluate_fused_ssim_v2.py", "fault_fused_ssim_v2.py")')
    namespace=function.__globals__
    namespace['__file__']=str(Path(__file__))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    return function


if __name__=='__main__':compose()()
