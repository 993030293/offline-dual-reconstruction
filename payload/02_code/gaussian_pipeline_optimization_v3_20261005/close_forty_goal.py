"""Close the forty-item register after native text and actual writer-exit checks."""

import json
from pathlib import Path
import re
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'research/2026-10-05-gaussian-optimization-round3-50'
OUT=ROOT/'04_outputs/gaussian_pipeline_optimization_v3_20261005/oup020/R172'
sys.path.insert(0,str(ROOT/'02_code/gaussian_pipeline_optimization_v1_20261005'))
from runtime import atomic_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def ended(row):
    try:return psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess:return True


def main():
    started=time.perf_counter();state=read(DOC/'CURRENT_STATE.json')
    names=('FORTY_OPTIMIZATIONS_INDEX.md','R171_STRICT_SIFT_ENCODING_RESULTS.md','R172_FUSED_SSIM_RESULTS.md',
        'CANDIDATE_BOARD.md','PLAN.md','OPTIMIZE_CHECKLIST.md','TARGET_OVERRIDE_40.md')
    texts={n:(DOC/n).read_text(encoding='utf-8') for n in names}
    rows=re.findall(r'^\| (\d+) \| ([^|]+) \| ([^|]+) \| \[回执\]\((C:/[^)]+)\) \|$',texts[names[0]],re.MULTILINE)
    acceptance=read(OUT/'ACCEPTANCE_WORK_V7_001.json');review=read(OUT/'review1/RESULT.json')
    inventory=read(OUT/'FORTY_RECEIPT_INVENTORY.json')
    leases=[read(path) for path in OUT.parent.glob('*_LEASE.json')]
    links=[path for name in names[:3] for path in re.findall(r'\]\((C:/[^)]+)\)',texts[name])]
    aligned=len(rows)==40 and all(int(row[0])==i and row[1].strip()==name
        and Path(row[3])==ROOT/receipt for i,(row,name,receipt) in enumerate(zip(rows,state['accepted_mechanisms'],state['acceptance_receipts']),1))
    checks={
        'target40_exact_unique_40_current_mechanisms_and_receipts':state['additional_target']==40
            and state['additional_accepted_count']==len(state['accepted_mechanisms'])==len(set(state['accepted_mechanisms']))
            ==len(state['acceptance_receipts'])==len(set(state['acceptance_receipts']))==40,
        'all40_native_table_rows_and_inventory_exactly_align':aligned and inventory['count']==40
            and [x['mechanism'] for x in inventory['rows']]==state['accepted_mechanisms'],
        'all40_current_receipt_gate_content_passed_no_candidates_counted':all(all(read(ROOT/p).get('checks',{}).values()) for p in state['acceptance_receipts'])
            and state['R169_new_optimization_count']==state['R170_new_optimization_count']==0
            and state['R171_new_optimization_count']==state['R172_new_optimization_count']==1,
        'all_seven_native_texts_read_and_current_top40':all(text.strip() for text in texts.values())
            and all('40/40' in texts[n][:600] for n in names[3:]),
        'all_direct_native_report_and_index_links_exist':bool(links) and all(Path(p).is_file() for p in links),
        'last_two_actual_net_work_and_scope_stated':f"{100*read(ROOT/state['R171_acceptance'])['net_registered_array_output_reduction']:.9f}%" in texts[names[1]]
            and f"{100*acceptance['net_registered_output_work_reduction']:.9f}%" in texts[names[2]]
            and '原因未确认' in texts[names[2]] and '非独立代理' in texts[names[2]]
            and '未知物体' in texts[names[0]] and '未统一部署生产' in texts[names[0]],
        'actual_final_publisher_and_separate_reviewer_births_ended':ended(acceptance['writer']) and ended(review['writer']),
        'all_final_family_exclusive_leases_released_and_births_ended':all('lease_released_utc' in x and
            all(ended({'pid':x[p+'_pid'],'creation_time':x[p+'_creation_time']}) for p in ('guard','child')) for x in leases),
        'all_final17_quality_work_source_error_recovery_resource_gates':len(review['checks'])==17 and all(review['checks'].values())
            and all(acceptance['checks'].values()),
        'no41st_candidate_or_required_unfinished_accepted_branch':state['current_next_action'].startswith('stop_new_candidate_search_at_user40')
            and state['current_next_brief'].startswith('none_no_41st'),
    }
    if not all(checks.values()):raise AssertionError(json.dumps(checks))
    path=OUT/'FINAL_TEXT_STATE_POST_EXIT_001.json'
    if path.exists():raise FileExistsError(path)
    result={'status':'complete_user40_all_acceptance_native_text_counter_links_and_actual_writer_exit_verified',
        'checks':checks,'effective_count':40,'target':40,'new_optimization_count':0,
        'actual_ended_publisher':acceptance['writer'],'actual_ended_reviewer':review['writer'],
        'native_text_paths':[str(DOC/n) for n in names],
        'screenshots_OCR_document_hashes_or_independent_agent_review_used':False,
        'writer':{'pid':psutil.Process().pid,'creation_time':psutil.Process().create_time()},
        'final_closure_seconds_before_write':time.perf_counter()-started}
    atomic_json(path,result)
    state.update(status='completed_user40_all_acceptance_native_text_and_actual_publisher_exit_verified',
        completion_target=40,effective_completed_count=40,new_candidate_search_stopped=True,
        final_text_state_post_exit=path.relative_to(ROOT).as_posix(),
        forty_optimization_index=(DOC/'FORTY_OPTIMIZATIONS_INDEX.md').relative_to(ROOT).as_posix(),
        R172_status='accepted_optional_scoped_work_V7_actual_publisher_exit_and_final40_verified',
        R163_current_status='unaccepted_candidate0_unknown_GPU_tail_cost_no_forced_acceptance',
        current_next_action='none_user_target40_achieved_stop_goal_after_actual_finalizer_exit',current_next_brief=None)
    atomic_json(DOC/'CURRENT_STATE.json',state)
    print(json.dumps({'status':result['status'],'effective_count':40,'target':40,'checks':checks}),flush=True)


if __name__=='__main__':main()
