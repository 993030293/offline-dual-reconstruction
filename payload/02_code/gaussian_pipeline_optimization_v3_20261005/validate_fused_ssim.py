"""Check real complete-consumer errors, SIGINT and fresh full recoveries."""

import json
import math
from pathlib import Path
import time

import run_fused_ssim_campaign_v4 as campaign
import run_interior_conv_campaign as original
from runtime import atomic_json, data_identity

OWN=campaign.OWN


def prefix_compare(left,right):
    if len(left)!=4 or len(right)!=4:return False
    for a,b in zip(left,right):
        one,two=dict(a),dict(b)
        one.pop('seconds_including_complete_input_protection');two.pop('seconds_including_complete_input_protection')
        av,bv=one.pop('result_values'),two.pop('result_values')
        if not isinstance(av,(int,float)) or not isinstance(bv,(int,float)) or abs(av-bv)>1e-7:return False
        if one!=two:return False
    return True


def main():
    started=time.perf_counter();out=OWN/'r0';out.mkdir(parents=True,exist_ok=False)
    formal=original.read(OWN/'c3/RESULT.json')
    if len(formal['pairs'])!=3 or not all(all(p['comparison']['checks'].values()) for p in formal['pairs']):
        raise AssertionError('The complete formal campaign failed')
    sources,inputs=campaign.identities()
    if formal['sources']!=sources or formal['inputs']!=inputs:raise AssertionError('The actual full source/input revision changed')
    extra=0;rows=[];pairs=[]
    for kind in ('error','cancel','recover'):
        local={}
        for mode in ('native','interior'):
            folder=out/(kind+'_'+mode);original.prepare(folder)
            node=campaign.invoke(folder,mode,None if kind=='recover' else kind)
            work=original.read(folder/'OUTPUT_WORK.json')
            extra+=work['trace']['successful_new_CUDA_output_elements']
            if not all(work['checks'].values()):raise AssertionError('A real fault/recovery cache, binding or resource closure failed')
            if kind!='recover':
                fault=original.read(folder/'FAULT_RESULT.json')
                if not all(fault['checks'].values()):raise AssertionError('An original real fault/SIGINT contract failed')
            else:
                comparison=original.compare(OWN/'c3/p0/native',folder)
                if not all(comparison['checks'].values()):raise AssertionError('A fresh complete recovery changed the original control')
            local[mode]=work
            rows.append({'kind':kind,'mode':mode,'actual_output_elements':work['trace']['successful_new_CUDA_output_elements'],
                'actual_node_seconds':node['complete_node_seconds'],'folder':str(folder)})
            if campaign.identities()!=(sources,inputs):raise AssertionError('A frozen full recovery source/input changed')
        if kind!='recover':
            same=prefix_compare(local['native']['scientific_prefix_rows'],local['interior']['scientific_prefix_rows'])
            same=same and local['native']['complete_original_parameters_before_fault']==local['interior']['complete_original_parameters_before_fault']
            same=same and local['native']['attempted_ordinals']==local['interior']['attempted_ordinals']==[1,2,3,4,5]
            pairs.append({'kind':kind,'four_original_full_scientific_prefixes_parameters_and_attempt_order_match':same})
            if not same:raise AssertionError('The original complete scientific fault prefix changed')
    checks={'four_actual_RuntimeError_native_Linux_SIGINT_and_prefixes':len(pairs)==2 and all(x['four_original_full_scientific_prefixes_parameters_and_attempt_order_match'] for x in pairs),
        'both_fresh_full48_96_recoveries_original_control':len(rows)==6,
        'all_original_frozen_sources_and_inputs':campaign.identities()==(sources,inputs),
        'no_partial_outputs_used_as_complete_recovery':all(not (out/(kind+'_'+mode)/'measurement/RESULT.json').exists()
            for kind in ('error','cancel') for mode in ('native','interior'))}
    atomic_json(out/'RESULT.json',{'status':'all_actual_errors_SIGINT_fresh_complete_recovery_passed','checks':checks,
        'rows':rows,'prefix_comparisons':pairs,'extra_actual_output_elements':extra,'sources':sources,'inputs':inputs,
        'validation_source':data_identity(Path(__file__)),'seconds':time.perf_counter()-started,'new_optimization_count':0})
    print(json.dumps({'checks':checks,'extra_actual_output_elements':extra,'seconds':time.perf_counter()-started}),flush=True)
    if not all(checks.values()):raise AssertionError('A real full validation protection failed')


if __name__=='__main__':main()
