"""Run all three fresh complete SSIM pairs with interior reuse common to both."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import run_interior_conv_campaign as original
from runtime import atomic_json, data_identity

ROOT, CODE = original.ROOT, original.CODE
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup020/R172"


def invoke(folder, mode, fault=None):
    script='evaluate_fused_ssim.py' if fault is None else 'fault_fused_ssim.py'
    command=[sys.executable,str(CODE/'fused_ssim_node.py'),'--kind','quality','--mode','process',
             '--node',str(folder/'node'),'--output',str(folder),'--','wsl','-d','E26-C2-LIV-20260907-134802','--',
             '/opt/e26_thin_surface_v1/venv/bin/python','-B',original.linux(CODE/script),
             '--mode',mode,'--pair',original.linux(folder/'pair'),'--localization',original.linux(original.LOCALIZATION)]
    if fault is not None:command.extend(['--fault',fault])
    code=subprocess.run(command).returncode
    node=original.read(folder/'node/NODE.json')
    expected=0 if fault is None else 86
    if code!=(0 if fault is None else 1) or node['guard']['returncode']!=expected:
        raise AssertionError('The complete evaluator or expected fault returned unexpectedly')
    if node['live_Windows_after_wait'] or any(x['alive'] for x in node['actual_Linux_after_wait']) or node.get('controller_error'):
        raise AssertionError('An owned complete evaluator did not close')
    return node


def identities():
    sources, inputs=original.identity_set()
    sources += [Path(__file__),CODE/'evaluate_fused_ssim.py',CODE/'fused_ssim_formula.py',CODE/'ssim_output_census.py',
                CODE/'fused_ssim_node.py',CODE/'fault_fused_ssim.py']
    return [data_identity(p) for p in sources],[data_identity(p) for p in inputs]


def main():
    started=time.perf_counter();out=OWN/'c0';out.mkdir(parents=True,exist_ok=False)
    generic=original.read(OWN.parent/'g0/RESULT.json')
    if not all(generic['checks'].values()):raise AssertionError('Generic checks failed')
    sources,inputs=identities()
    atomic_json(out/'CONTRACT.json',{'sources':sources,'inputs':inputs,'common_Q30_Q61_R100_R135_R134':True,
        'original_full1016_24_48_96':True,'quality_SSIM_tolerance':1e-7,'each_whole_limit':1.1,'median_whole_limit':1.03,
        'net_registered_output_work_limit':.1,'unknown_extra_output_elements_reserve':1000000000,
        'remaining_text_interface_closure_seconds_reserve':120,'new_optimization_count':0})
    pairs=[]
    for trial in range(3):
        local={}
        for mode in (('native','interior') if trial%2==0 else ('interior','native')):
            before=time.perf_counter();folder=out/f'p{trial}'/mode
            original.prepare(folder);node=invoke(folder,mode)
            work=original.read(folder/'OUTPUT_WORK.json')
            local[mode]={'whole_seconds':time.perf_counter()-before,
                'actual_output_elements':work['trace']['successful_new_CUDA_output_elements'],
                'folder':str(folder),'node_seconds':node['complete_node_seconds']}
            if identities()!=(sources,inputs):raise AssertionError('A frozen complete SSIM source or input changed')
        comparison=original.compare(out/f'p{trial}/native',out/f'p{trial}/interior')
        comparison['checks']['each_original_whole_limit1_1']=local['interior']['whole_seconds']<=1.1*local['native']['whole_seconds']
        comparison['checks']['actual_output_work_reduction10percent']=local['interior']['actual_output_elements']<=.9*local['native']['actual_output_elements']
        row={'trial':trial,'sides':local,'comparison':comparison};pairs.append(row)
        atomic_json(out/'STATE.json',{'pairs':pairs,'new_optimization_count':0})
        print(json.dumps(row),flush=True)
        if not all(comparison['checks'].values()):raise AssertionError('An original complete SSIM science, work or whole guard failed')
        if time.perf_counter()-started>1700:raise RuntimeError('The registered campaign reserve was exceeded')
    medians={mode:statistics.median(x['sides'][mode]['whole_seconds'] for x in pairs) for mode in ('native','interior')}
    if medians['interior']>1.03*medians['native']:raise AssertionError('The preregistered raw whole median failed')
    atomic_json(out/'RESULT.json',{'status':'three_complete_current_pairs_pending_fault_recovery_root_cost','pairs':pairs,
        'raw_whole_medians_seconds':medians,'sources':sources,'inputs':inputs,'complete_seconds':time.perf_counter()-started,'new_optimization_count':0})


if __name__=='__main__':main()
