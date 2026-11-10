"""Validate the prepared compiled cache through the original generic consumer."""

import inspect
import os
from pathlib import Path

import check_fused_ssim_v2 as original


def main():
    os.environ['TRITON_CACHE_DIR']=str(original.original.OUT.parent/'triton_v2_shared_cache2')
    source=inspect.getsource(original.main).replace("OUT.parent/'g1'","OUT.parent/'g2'")
    namespace=dict(original.__dict__,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
