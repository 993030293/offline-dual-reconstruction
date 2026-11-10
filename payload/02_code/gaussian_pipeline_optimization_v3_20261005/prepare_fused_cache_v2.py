"""Copy one canonical compiled cache and rebase only its path manifests."""

import inspect
import json
from pathlib import Path

import prepare_fused_cache as original


def main():
    source=inspect.getsource(original.main).replace("'triton_v2_shared_cache'","'triton_v2_shared_cache2'").replace(
        "origins=[OWN/f'c1/p{i}/interior/triton_cache' for i in (0,1)]",
        "origins=[OWN/'c1/p0/interior/triton_cache']")
    anchor="    atomic_json(OWN.parent/'COMPILED_CACHE_PREPARATION.json',"
    extra='''    groups=[]
    origin_linux='/mnt/c/'+origins[0].as_posix()[3:]
    target_linux='/mnt/c/'+target.as_posix()[3:]
    for path in target.rglob('__grp__*.json'):
        payload=json.loads(path.read_text(encoding='utf-8'))
        if set(payload)!= {'child_paths'}:raise ValueError('An actual compiler group schema changed')
        old=data_identity(path)
        mapped={}
        for key,value in payload['child_paths'].items():
            if not value.startswith(origin_linux+'/'):raise ValueError('A compiler group points outside its canonical origin')
            relative=value[len(origin_linux)+1:]
            if not (target/relative).is_file():raise ValueError('A copied compiler child is missing')
            mapped[key]=target_linux+'/'+relative
        atomic_json(path,{'child_paths':mapped})
        groups.append({'before_copy_group':old,'after_rebased_paths':data_identity(path),'child_paths':mapped})
    for row in rows:row['current_copy']=data_identity(Path(row['copy']['path']))
'''
    if source.count(anchor)!=1:raise ValueError('The original private cache preparation changed')
    source=source.replace(anchor,extra+anchor.replace('COMPILED_CACHE_PREPARATION.json','COMPILED_CACHE_PREPARATION_V2.json'))
    source=source.replace("'rows':rows,'seconds':", "'rows':rows,'rebased_groups':groups,'seconds':")
    namespace=dict(original.__dict__,json=json,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
