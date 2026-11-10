"""Use the same verified formula with source-bound private compiled artifacts."""

import inspect
from pathlib import Path

import evaluate_fused_ssim_v2 as original

entry=original.entry


def run(fault=None):
    source=inspect.getsource(original.original.run)
    before='os.environ["TRITON_CACHE_DIR"] = str(pair.parent / "triton_cache")'
    if source.count(before)!=1:raise ValueError('The original private compilation cache boundary changed')
    source=source.replace(before,'os.environ["TRITON_CACHE_DIR"] = str(OWN.parent / "triton_v2_shared_cache2")')
    namespace=dict(original.namespace,__file__=str(Path(__file__)),composed_main=original.composed_main)
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    return namespace['run'](fault)


if __name__=='__main__':run()
