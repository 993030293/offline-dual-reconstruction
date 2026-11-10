"""Use the original fifth-call real failure and binding restoration contract."""

import inspect
from pathlib import Path

import fault_interior_conv_work as original


def main():
    source=inspect.getsource(original.main)
    before='import evaluate_interior_conv_work as entry'
    if source.count(before)!=1:raise ValueError('The original SSIM fault source boundary changed')
    source=source.replace(before,'import evaluate_fused_ssim as entry')
    namespace=dict(original.__dict__,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)), 'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
