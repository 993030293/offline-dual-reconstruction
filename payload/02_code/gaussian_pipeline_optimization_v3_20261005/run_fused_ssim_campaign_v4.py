"""Run fresh same-kernel complete pairs after a separately closed guard failure."""

import inspect
from pathlib import Path

import run_fused_ssim_campaign_v3 as original
from runtime import data_identity

invoke=original.invoke
ROOT,CODE,OWN=original.original.ROOT,original.original.CODE,original.original.OWN


def identities():
    sources,inputs=original.identities()
    return sources+[data_identity(Path(__file__))],inputs


def main():
    closure=original.original.original.read(original.original.OWN/'c2/p0/interior/ABORTED_NODE_CLOSURE.json')
    generic=original.original.original.read(original.original.OWN.parent/'g2/RESULT.json')
    if not all(closure['checks'].values()) or not all(generic['checks'].values()):raise AssertionError('Actual closure or cache consumer health failed')
    source=inspect.getsource(original.original.main).replace("out=OWN/'c0'","out=OWN/'c3'").replace("OWN.parent/'g0/RESULT.json'","OWN.parent/'g2/RESULT.json'")
    namespace=dict(vars(original.original),identities=identities,invoke=invoke,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
