"""Preserve an incomplete original guard and verify its actual owned exits."""

import json
from pathlib import Path
import subprocess

import psutil

import run_fused_ssim_campaign as campaign
from runtime import atomic_json


def ended(row):
    try:return psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess:return True


def main():
    folder=campaign.OWN/'c2/p0/interior'
    node=campaign.original.read(folder/'node/NODE.json')
    samples=sorted((folder/'node/guard_history').glob('*.json'))
    journal=campaign.original.read(samples[-1])
    windows=journal['owned_process_identities']+[node['owner']]
    resource=campaign.original.read(folder/'measurement/WSL_RESOURCE.json')
    identity={'pid':resource['pid'],'creation_time':resource['creation_time']}
    code="import psutil,json;row=json.loads("+repr(json.dumps(identity))+ ");alive=False\n"
    code+="try:\n p=psutil.Process(row['pid']);alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\nexcept psutil.NoSuchProcess:pass\nprint(json.dumps(dict(row,alive=alive)))"
    actual=json.loads(subprocess.check_output(['wsl','-d','E26-C2-LIV-20260907-134802','--',
        '/opt/e26_thin_surface_v1/venv/bin/python','-B','-c',code],text=True,timeout=30))
    checks={'all_saved_Windows_birth_instances_ended':all(ended(x) for x in windows),
        'actual_saved_Linux_birth_ended':not actual['alive'],
        'incomplete_original_guard_not_rewritten_or_accepted':not (folder/'node/guard.json').exists()
            and node['status']=='running' and journal['status']=='running'}
    if not all(checks.values()):raise AssertionError(checks)
    path=folder/'ABORTED_NODE_CLOSURE.json'
    if path.exists():raise FileExistsError(path)
    atomic_json(path,{'status':'closed_failed_guard_missing_final_receipt_reason_unconfirmed','checks':checks,
        'Windows':windows,'Linux':actual,'unknown_partial_CUDA_outputs_conservatively_charged_as_one_complete_original_side':1777725783,
        'no_scientific_or_performance_result_from_aborted_side':True,'new_optimization_count':0})
    print(checks,flush=True)


if __name__=='__main__':main()
