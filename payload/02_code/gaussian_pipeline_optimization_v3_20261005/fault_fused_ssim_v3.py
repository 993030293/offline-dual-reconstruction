"""Use the same original fault and real SIGINT contract with the V3 entry."""

import inspect
from pathlib import Path

import fault_interior_conv_work as original


def main():
    source=inspect.getsource(original.main)
    before='import evaluate_interior_conv_work as entry'
    if source.count(before)!=1:raise ValueError('The original complete SSIM fault source changed')
    namespace=dict(original.__dict__,__file__=str(Path(__file__)))
    exec(compile(source.replace(before,'import evaluate_fused_ssim_v3 as entry'),str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
