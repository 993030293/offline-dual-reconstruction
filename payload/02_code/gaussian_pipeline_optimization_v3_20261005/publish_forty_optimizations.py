"""Publish the final scoped mechanism and an evidence-linked forty-item index."""

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

DESCRIPTIONS=[
    '直接处理原生二进制观测数组，减少检查时的转换和复制',
    '完整checkpoint使用已有Zstandard库做无损传输',
    '将精确整数SIFT距离按批计算，减少重复匹配工作',
    '并行调度训练后的评价和归档任务',
    '持有不可变输入的完整身份，复用已验证的输入核对结果',
    '为原生观测建立不可变索引，减少重复查找',
    '完整checkpoint使用私有内存映射读取',
    '复用原始相机标定主体，保留完整标定契约',
    '在设备上完成最终展示产物的精确量化',
    '按序准备参考数据，与查询定位衔接',
    '通过精确灰度直方图计算选帧质量统计',
    '增量维护选帧新颖度索引，保留原选择结果',
    '使用相机网格支持索引减少重复支持统计',
    '有界调度多文件身份计算，保留完整输入检查',
    '并行执行原生CPU查询SIFT提取，避免GPU并发争用',
    '预备连续数组后调用原生几何计算，保持原位姿容差',
    '分块读取完整Gaussian PLY并核对原字段',
    '对完整地图分段统计相机支持',
    '使用已有原生主机完成完整Gaussian文件操作',
    '按视角评价多个模型并及时释放参考统计',
    '将不可变地图检查及产物任务组织为有依赖的DAG',
    '为原生质量线程使用私有目标平面，减少共享和重复工作',
    '预览中避免不参与当前视角的绘制工作',
    '使用已有NVML接口优化资源监测阶段',
    '融合RGGB到质量输入的转换，保留原像素结果',
    '直接处理平面角点响应，减少质量评价中间转换',
    '使用严格SIMD计算原灰度统计',
    '优化实际点拒绝判断，保留原可见点集',
    '融合原生相机像素摘要计算，保留完整输入身份',
    '交接已加载图像，减少重复原文件读取',
    '复用完整图和目标裁剪内部的SSIM计算，只重算边界',
    '按CPU依赖并行恢复完整训练状态',
    '使用严格SIMD完成全量RGB采样',
    '按依赖调度离散浏览视角的CPU收敛计算',
    '复用预览排序的临时数组，保留全部原排序结果',
    '用隔离分片并行执行完整原生SfM匹配并事务合并全列',
    '保留全部CUDA比较，减少恢复时主机布尔实体化',
    '在设备上创建初始化的已知零特征，减少主机上传',
    '用CPU单扫描严格SIFT编码，减少临时数组输出',
    '融合评价SSIM末段代数，减少成功新CUDA tensor输出',
]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def ended(row):
    try:return psutil.Process(row['pid']).create_time()!=row['creation_time']
    except psutil.NoSuchProcess:return True


def main():
    started=time.perf_counter()
    review=read(OUT/'review1/RESULT.json');lease=read(OUT.parent/'REVIEW_V2_LEASE.json')
    state=read(DOC/'CURRENT_STATE.json')
    texts={name:(DOC/name).read_text(encoding='utf-8') for name in ('R171_STRICT_SIFT_ENCODING_RESULTS.md',
        'R172_FUSED_SSIM_RESULTS.md','R172_FUSED_SSIM_TASK.md','R172_V2_MIXED_LAYOUT_ADDENDUM.md','R172_V3_COMPILED_CACHE_ADDENDUM.md')}
    fee=review['known_actual_fee_seconds_plus_registered120']+lease['child_elapsed_seconds']
    charged={mode:value+fee/6 for mode,value in review['raw_whole_medians_seconds'].items()}
    acceptance=OUT/'ACCEPTANCE_WORK_V7_001.json'
    mechanism='R172_COMPLETE_EVALUATION_SSIM_STRIDED_ELEMENTWISE_FUSION_WORK'
    links=[link for text in texts.values() for link in re.findall(r'\]\((C:/[^)]+)\)',text)]
    historical=[]
    for key in ('R162_final_post_exit','R164_final_post_exit','R165_final_cost_and_actual_post_exit',
                'R168_final_text_state_closure','R157_V7_final_post_exit'):
        receipt=read(ROOT/state[key]);historical.append({'key':key,'path':state[key],
            'checks_passed':all(receipt.get('checks',{}).values())})
    checks={
        'actual17_final_work_quality_recovery_cost_resource_checks_passed':len(review['checks'])==17 and all(review['checks'].values()),
        'actual_review_and_lease_writer_births_ended':ended(review['writer']) and all(ended({'pid':lease[p+'_pid'],
            'creation_time':lease[p+'_creation_time']}) for p in ('guard','child')),
        'actual_review_exit0_and_lease_released':lease['returncode']==0 and 'lease_released_utc' in lease,
        'final_time_fee_whole_V7_and_registered_work_gate':charged['interior']<=1.03*charged['native'] and review['net_registered_output_work_reduction']>=.1,
        '39_prior_unique_current_acceptances_target40_and_one_family_only':state['additional_target']==40
            and state['additional_accepted_count']==len(set(state['accepted_mechanisms']))==len(set(state['acceptance_receipts']))==39
            and mechanism not in state['accepted_mechanisms'] and not acceptance.exists(),
        'native_five_texts_read_scope_failures_net_values_and_links':all(text.strip() for text in texts.values())
            and '24.185665727%' in texts['R172_FUSED_SSIM_RESULTS.md'] and '原因未确认' in texts['R172_FUSED_SSIM_RESULTS.md']
            and bool(links) and all(Path(p).exists() for p in links),
        'historical_conditional_receipts_have_actual_post_closure_checks':all(x['checks_passed'] for x in historical),
        'exactly40_explanations_and_all39_prior_receipt_gate_content':len(DESCRIPTIONS)==40
            and all(all(read(ROOT/p).get('checks',{}).values()) for p in state['acceptance_receipts']),
    }
    if not all(checks.values()):raise AssertionError(json.dumps(checks))
    result={'status':'accepted_optional_scoped_work_V7_final40_pending_external_publication_exit',
        'mechanism':mechanism,'new_optimization_count':1,'checks':checks,'root_review':str(OUT/'review1/RESULT.json'),
        'report':str(DOC/'R172_FUSED_SSIM_RESULTS.md'),'raw_output_reduction':1-review['original_output_elements']['interior']/review['original_output_elements']['native'],
        'complete_extra_output_elements_including_registered1e9':review['complete_extra_output_elements_including1e9_reserve'],
        'net_registered_output_work_reduction':review['net_registered_output_work_reduction'],
        'raw_whole_medians_seconds':review['raw_whole_medians_seconds'],
        'known_actual_fee_seconds_plus_registered120':fee,'charged_whole_medians_seconds':charged,
        'full_speed_training_GT_unknown_object_whole_pipeline_peak_memory_or_production_accepted':False,
        'single_chat_no_subagents_separate_self_review_not_independent':True,'historical_post_closures':historical,
        'registered120_seconds_includes_remaining_text_publication_and_post_exit_execution':True,
        'writer':{'pid':psutil.Process().pid,'creation_time':psutil.Process().create_time()},
        'publication_seconds_before_write':time.perf_counter()-started}
    atomic_json(acceptance,result)
    state['accepted_mechanisms'].append(mechanism)
    state['acceptance_receipts'].append(acceptance.relative_to(ROOT).as_posix())
    state.update(additional_accepted_count=40,status='effective40_pending_final_native_text_and_actual_publisher_exit',
        R172_status=result['status'],R172_new_optimization_count=1,R172_acceptance=acceptance.relative_to(ROOT).as_posix(),
        R172_report=(DOC/'R172_FUSED_SSIM_RESULTS.md').relative_to(ROOT).as_posix(),
        current_next_action='stop_new_candidate_search_at_user40_only_final_text_counter_links_and_actual_publisher_exit_remain',
        current_next_brief='none_no_41st_optimization_authorized_under_completed40_target')
    atomic_json(DOC/'CURRENT_STATE.json',state)
    lines=['# 本轮新增40项有效优化清单','',
        '2026-10-08，按当前用户40项停止条件汇总；原有15+20项不重复计数，候选、失败、参数和验证修正均不计数。以下每行对应一个独立接纳回执。',
        '', '这些是各自已验证边界内的工程优化，包括速度、工作量、等待或资源取舍；不能把各百分比相加、称为40项整体提速或未知物体/GT画质均已验证。很多项仍为独立目录中的可选实现，未统一部署生产。',
        '', '最后两项：R171严格SIFT编码净数组输出工作少23.485606562%；R172评价SSIM融合净CUDA API输出元素工作少24.185665727%。两项完整消费者质量与错误/中断/新目录恢复通过，均以工作项接纳，净速度未接纳。',
        '', '| 序号 | 唯一机制 | 简单说明 | 原接纳证据 |','| ---: | --- | --- | --- |']
    inventory=[]
    for i,(name,path,description) in enumerate(zip(state['accepted_mechanisms'],state['acceptance_receipts'],DESCRIPTIONS),1):
        row=read(ROOT/path)
        inventory.append({'ordinal':i,'mechanism':name,'description':description,'receipt':path,
            'original_receipt_status':row['status'],'all_present_gate_checks_passed':all(row.get('checks',{}).values())})
        lines.append(f'| {i} | {name} | {description} | [回执]({(ROOT/path).as_posix()}) |')
    lines.extend(['','失败和范围限制保留在各报告与CURRENT_STATE：R169/R170、R172 V1/V2及启动残缺均贡献0；R163费用尾段未闭合仍不接纳。公开GT和未知物体完整链实验没有因本次达标自动开展。','',
        '代码、数据、环境和原始产物边界依照各回执；文本复核直接读取本文与报告，最后实际publisher退出另有闭合回执。'])
    index=DOC/'FORTY_OPTIMIZATIONS_INDEX.md'
    if index.exists():raise FileExistsError(index)
    with index.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    atomic_json(OUT/'FORTY_RECEIPT_INVENTORY.json',{'additional_target':40,'count':40,'rows':inventory,
        'existing35_excluded':True,'new_optimization_count':0,'publisher':result['writer']})
    print(json.dumps({'count':40,'target':40,'checks':checks,'net_work':result['net_registered_output_work_reduction'],
        'charged_seconds':charged,'status':result['status']}),flush=True)


if __name__=='__main__':main()
