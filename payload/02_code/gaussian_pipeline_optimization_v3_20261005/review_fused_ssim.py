"""Separately audit actual full SSIM quality, output work, costs and closure."""

import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import psutil

import run_fused_ssim_campaign_v4 as campaign
import run_interior_conv_campaign as common
from runtime import atomic_json, data_identity

ROOT,CODE,OUT=campaign.ROOT,campaign.CODE,campaign.OWN
read=common.read


def ended(row):
    try:return psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess:return True


def census(trace):
    count=sum(math.prod(x['shape']) for row in trace['rows'] for x in row['outputs'])
    sizes=sum(x['bytes'] for row in trace['rows'] for x in row['outputs'])
    return count==trace['successful_new_CUDA_output_elements'] and sizes==trace['successful_new_CUDA_output_bytes'] and all(
        math.prod(x['shape'])==x['elements'] and x['bytes']>=x['elements'] for row in trace['rows'] for x in row['outputs'])


def main():
    started=time.perf_counter();folder=OUT/'review0';folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),folder/'REVIEW_SOURCE.py')
    formal,validation=read(OUT/'c3/RESULT.json'),read(OUT/'r0/RESULT.json')
    g0,g1,g2=[read(OUT.parent/name/'RESULT.json') for name in ('g0','g1','g2')]
    comparisons=[common.compare(OUT/f'c3/p{i}/native',OUT/f'c3/p{i}/interior') for i in range(3)]
    recoveries=[common.compare(OUT/'c3/p0/native',OUT/'r0'/('recover_'+mode)) for mode in ('native','interior')]
    traces=[value for generic in (g0,g1,g2) for value in generic['trace'].values()]
    formal_traces=[];negative_work=0
    for path in OUT.rglob('OUTPUT_WORK.json'):
        work=read(path);traces.append(work['trace'])
        if any(name in path.parts for name in ('c0','c1','c2')):negative_work+=work['trace']['successful_new_CUDA_output_elements']
        if 'c3' in path.parts:formal_traces.append(work)
    aborted=read(OUT/'c2/p0/interior/ABORTED_NODE_CLOSURE.json')
    negative_work+=aborted['unknown_partial_CUDA_outputs_conservatively_charged_as_one_complete_original_side']
    extra=sum(t['successful_new_CUDA_output_elements'] for generic in (g0,g1,g2) for t in generic['trace'].values())
    extra+=negative_work+validation['extra_actual_output_elements']+1000000000
    units={mode:statistics.median(p['sides'][mode]['actual_output_elements'] for p in formal['pairs']) for mode in ('native','interior')}
    net=(units['native']-units['interior'])/(units['native']+extra/6)
    sources,inputs=campaign.identities()
    source_checks=sources==formal['sources'] and inputs==formal['inputs'] and sources==validation['sources'] and inputs==validation['inputs']
    for generic in (g0,g1,g2):
        source_checks=source_checks and all(data_identity(common.windows(row['path']))['sha256']==row['sha256'] for row in generic['sources'])
    nodes=[read(path) for path in OUT.rglob('NODE.json')]
    win={};lin={}
    for node in nodes:
        if 'guard' not in node:continue
        for row in node['guard']['owned_process_identities']+[node['owner']]:win[(row['pid'],row['creation_time'])]=row
        for row in node['Linux_resource_receipts']:lin[(row['pid'],row['creation_time'])]={'pid':row['pid'],'creation_time':row['creation_time']}
    for row in aborted['Windows']:win[(row['pid'],row['creation_time'])]=row
    row=aborted['Linux'];lin[(row['pid'],row['creation_time'])]={'pid':row['pid'],'creation_time':row['creation_time']}
    leases=[]
    for path in OUT.parent.glob('*_LEASE.json'):
        if path.name=='REVIEW_LEASE.json':continue
        value=read(path);leases.append({'path':str(path),'seconds':value['child_elapsed_seconds'],'returncode':value['returncode'],'released':'lease_released_utc' in value})
        for prefix in ('guard','child'):
            win[(value[prefix+'_pid'],value[prefix+'_creation_time'])]={'pid':value[prefix+'_pid'],'creation_time':value[prefix+'_creation_time']}
    for name in ('G0_GUARD.json','G1_GUARD.json','G2_GUARD.json','FULL_V3_GUARD.json'):
        for row in read(OUT.parent/name)['owned_process_identities']:win[(row['pid'],row['creation_time'])]=row
    for name in ('g0','g1','g2'):
        row=read(OUT.parent/name/'WSL_RESOURCE.json');lin[(row['pid'],row['creation_time'])]={'pid':row['pid'],'creation_time':row['creation_time']}
    code='import json,psutil; rows=json.loads('+repr(json.dumps(list(lin.values())))+'); result=[]\n'
    code+="for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,alive=alive))\nprint(json.dumps(result))"
    actual_linux=json.loads(subprocess.check_output(['wsl','-d','E26-C2-LIV-20260907-134802','--','/opt/e26_thin_surface_v1/venv/bin/python','-B','-c',code],text=True,timeout=30))
    formal_seconds=sum(sum(x['whole_seconds'] for x in p['sides'].values()) for p in formal['pairs'])
    fee_seconds=sum(x['seconds'] for x in leases)-formal_seconds+120
    charged={mode:value+fee_seconds/6 for mode,value in formal['raw_whole_medians_seconds'].items()}
    cache=read(OUT.parent/'COMPILED_CACHE_PREPARATION_V2.json')
    cache_code_unchanged=all(data_identity(Path(x['current_copy']['path']))==x['current_copy'] for x in cache['rows'])
    checks={
        'all24_map_bits_layout_reference_mutation_and16_generic_original_errors':all(all(x['checks'].values()) for x in (g0,g1,g2)),
        'three_full1016_24_48_96_quality_JPEG_contract_parameters_inputs_RNG':len(comparisons)==3 and all(all(x['checks'].values()) for x in comparisons),
        'both_fresh_full48_96_recoveries_original_control':all(all(x['checks'].values()) for x in recoveries),
        'four_actual_fifth_call_errors_native_SIGINT_full_prefixes_and_binding_cleanup':all(validation['checks'].values()),
        'all_actual_successful_API_output_censuses':all(census(x) for x in traces),
        'all_frozen_current_sources_and_inputs':source_checks,
        'all_canonical_compiled_artifacts_and_rebased_groups_current':cache_code_unchanged,
        'all19_completed_or_expected_failed_nodes_and_one_aborted_node_closed':len(nodes)==20
            and sum('guard' in x for x in nodes)==19 and all(not x['live_Windows_after_wait'] and not any(y['alive'] for y in x['actual_Linux_after_wait']) for x in nodes if 'guard' in x)
            and all(aborted['checks'].values()),
        'all_saved_Windows_birth_instances_ended':all(ended(x) for x in win.values()),
        'all_saved_Linux_birth_instances_ended':all(not x['alive'] for x in actual_linux),
        'all_prior_exclusive_leases_released':all(x['released'] for x in leases),
        'all_formal_whole_ratio_within1_1':all(p['sides']['interior']['whole_seconds']<=1.1*p['sides']['native']['whole_seconds'] for p in formal['pairs']),
        'raw_and_charged_whole_median_within_V7_1_03':formal['raw_whole_medians_seconds']['interior']<=1.03*formal['raw_whole_medians_seconds']['native'] and charged['interior']<=1.03*charged['native'],
        'net_registered_API_output_work_at_least10percent':net>=.1,
        'all_failed_V1_V2_complete_work_and_time_preserved_charged':negative_work>0 and read(OUT/'c0/STATE.json')['pairs'][0]['comparison']['checks']['actual_output_work_reduction10percent'] is False
            and read(OUT/'c1/STATE.json')['pairs'][-1]['comparison']['checks']['each_original_whole_limit1_1'] is False,
        'only_actual_common_R134_independent_formula_mechanism_no_training_or_peak_claim':all(x['original_mean_and_conv_unchanged'] for x in formal_traces) and len(formal_traces)==6,
        'owned_family_disk_bound12GiB_and_free20GiB':sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file())<=12*1024**3 and shutil.disk_usage(OUT).free>=20*1024**3,
    }
    result={'status':'scoped_work_checks_passed_pending_actual_writer_exit_and_publication' if all(checks.values()) else 'failed_verification',
        'checks':checks,'original_output_elements':units,'negative_V1_V2_output_elements':negative_work,
        'complete_extra_output_elements_including1e9_reserve':extra,'net_registered_output_work_reduction':net,
        'raw_whole_medians_seconds':formal['raw_whole_medians_seconds'],'known_actual_fee_seconds_plus_registered120':fee_seconds,
        'charged_whole_medians_seconds':charged,'Windows':list(win.values()),'Linux':actual_linux,'leases':leases,
        'writer':{'pid':psutil.Process().pid,'creation_time':psutil.Process().create_time()},
        'separate_single_chat_self_review_not_independent':True,'review_seconds':time.perf_counter()-started,'new_optimization_count':0}
    atomic_json(folder/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','checks','net_registered_output_work_reduction','raw_whole_medians_seconds','review_seconds')}),flush=True)
    if not all(checks.values()):raise AssertionError('An original whole quality, source, resource, output-work or cost check failed')


if __name__=='__main__':main()
