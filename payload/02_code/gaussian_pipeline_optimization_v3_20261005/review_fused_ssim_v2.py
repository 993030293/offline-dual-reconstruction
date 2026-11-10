"""Read actual installed Linux sources on Linux while retaining the failed audit."""

import inspect
import json
from pathlib import Path
import subprocess

import review_fused_ssim as original


def check_sources(rows):
    linux=[x for x in rows if x['path'].startswith('/opt/')]
    windows=[x for x in rows if not x['path'].startswith('/opt/')]
    local=all(original.data_identity(original.common.windows(x['path']))['sha256']==x['sha256'] for x in windows)
    code="import json,pathlib,hashlib; rows=json.loads("+repr(json.dumps(linux))+ "); print(json.dumps([dict(path=x['path'],sha256=hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()) for x in rows]))"
    actual=json.loads(subprocess.check_output(['wsl','-d','E26-C2-LIV-20260907-134802','--',
        '/opt/e26_thin_surface_v1/venv/bin/python','-B','-c',code],text=True,timeout=30))
    return local and actual==[{'path':x['path'],'sha256':x['sha256']} for x in linux]


def main():
    source=inspect.getsource(original.main).replace("folder=OUT/'review0'","folder=OUT/'review1'")
    before="all(data_identity(common.windows(row['path']))['sha256']==row['sha256'] for row in generic['sources'])"
    if source.count(before)!=1:raise ValueError('The original source identity audit changed')
    source=source.replace(before,"_check_sources(generic['sources'])").replace(
        "if path.name=='REVIEW_LEASE.json':continue","if path.name=='REVIEW_V2_LEASE.json':continue")
    namespace=dict(original.__dict__,_check_sources=check_sources,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
