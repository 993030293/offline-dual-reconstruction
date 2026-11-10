"""Publish current target acceptance without erasing historical stops and costs."""

import argparse
from pathlib import Path
import shutil
import time

from common import atomic_json, read_json


def finalize(root, target):
    target.relative_to(root)
    delivered = read_json(target / 'DELIVERY_RESULT.json')
    if delivered['status'] != 'adaptive_sensor_dual_target_outputs_and_viewers_verified':
        raise ValueError('The adaptive target binaries and real interactions must complete first')
    coverage = read_json(target / 'stable_lidar/stable_lidar_receipt.json')['anchor_coverage']
    shared = read_json(root / 'shared/RESULT.json')
    regression = read_json(target / 'ACTUAL_EVIDENCE_FINAL_REGRESSION.json')
    frame_check = read_json(target / 'VISUAL_ADMISSION_FINAL_REGRESSION.json')
    if (coverage['visual_frame_count'] != shared['registered_frames'] or
            regression['status'] != 'actual_visual_set_and_raw_lidar_membership_verified' or
            frame_check['status'] != 'optimized_visual_admission_preserved_and_sensor_abstention_verified'):
        raise ValueError('The optimized visual frame set or genuine sensor membership changed')
    archive = target / 'prior_acceptance'
    archive.mkdir(exist_ok=False)
    for name in ('STATE.json', 'RESULT.json', 'DELIVERY_RESULT.json', 'EXCEPTION_LEDGER.json', 'RUN_REPORT.md', 'QUALITY_REVIEW.json'):
        if (root / name).is_file():
            shutil.copyfile(root / name, archive / name)
    state = read_json(archive / 'STATE.json')
    ledger = read_json(archive / 'EXCEPTION_LEDGER.json')
    correction_failures = []
    diagnostic = root / 'corrections/target_contract_v2_001'
    for path in (diagnostic / 'ANCHOR_PROCESS.json', diagnostic / 'SUPPORTED_ANCHOR_PROCESS.json',
                 diagnostic / 'stages/m3_visual_target/PROCESS.json'):
        item = read_json(path)
        if not item['status'].startswith('failed'):
            raise ValueError('A preserved failed branch has been relabeled')
        correction_failures.append({'process_receipt': str(path), 'status': item['status'],
            'elapsed_seconds': item['elapsed_seconds'], 'returncode': item.get('returncode'),
            'profile_passed_after_contract_change': False})
    new_failures = []
    for path in (target / 'stages').rglob('PROCESS.json'):
        item = read_json(path)
        if item['status'].startswith('failed'):
            new_failures.append({'process_receipt': str(path), 'status': item['status'],
                'elapsed_seconds': item.get('elapsed_seconds'), 'returncode': item.get('returncode')})
    atomic_json(target / 'MANUAL_INTERVENTION_INTEGRATION.json', {
        'status': 'development_assistant_intervention',
        'reason': 'Correct wrong all-scene point scope and adapt sensor admission to the existing optimized visual frame set.',
        'new_contract_frozen_before_its_evaluation': str(target / 'SENSOR_CONTRACT.json'),
        'old_per_frame_lidar_failures_reclassified_as_success': False,
        'trained_1197_frame_models_reused_from_same_raw_run': True,
        'algorithm_autonomous_recovery': False, 'new_full_cold_start_performed': False})
    ledger.update(status='adaptive_target_delivery_verified_with_historical_exceptions_and_assistant_correction',
        rejected_prior_point_scope='all_scene_TSDF_instead_of_ground_removed_target',
        prior_all_scene_delivery_accepted=False, correction_native_failures=correction_failures,
        adaptive_correction_native_failures=new_failures,
        correction_assistant_intervention=str(target / 'MANUAL_INTERVENTION_INTEGRATION.json'),
        adaptive_sensor_contract_completed=True, strict_ReliableLidar_profile_completed=False,
        legacy_LiDAR_gate_failures_preserved=True, unassisted_cold_start_verified=False,
        correction_diagnostic_failures=[str(target / 'ADAPTER_IMPORT_FAILURE.json')],
        unassisted_single_raw_to_delivery_verified=False,
        independent_geometry_accuracy_verified=False,
        current_target_delivery_result=str(target / 'DELIVERY_RESULT.json'))
    atomic_json(root / 'EXCEPTION_LEDGER.json', ledger)
    state.update(status='adaptive_target_delivery_verified_with_recorded_failures_and_assistant_correction',
        current_stage='target_delivery_correction_complete', current_target_root=str(target),
        original_all_scene_stages_are_not_target_acceptance=True,
        unassisted_cold_start_verified=False, independent_geometry_accuracy_verified=False,
        adaptive_sensor_contract_completed=True, strict_ReliableLidar_profile_completed=False,
        correction_completed_unix=time.time())
    atomic_json(root / 'STATE.json', state)
    atomic_json(root / 'RESULT.json', state)
    atomic_json(root / 'DELIVERY_RESULT.json', {**delivered,
        'prior_all_scene_acceptance': str(archive / 'DELIVERY_RESULT.json'),
        'prior_all_scene_delivery_accepted': False})
    shutil.copyfile(target / 'QUALITY_REVIEW.json', root / 'QUALITY_REVIEW.json')
    stages = []
    for path in sorted((target / 'stages').glob('*/PROCESS.json')):
        row = read_json(path)
        stages.append((path.parent.name, row['status'], row['elapsed_seconds']))
    tsdf = read_json(target / '07_reliable_lidar_tsdf/reliable_lidar_tsdf_receipt.json')
    selection = tsdf['surface_candidate_selection']
    n = len(ledger.get('failed_native_processes', []))
    lines = ['# Box1 新选帧与目标表面整合记录', '',
        '当前验收：本轮新生成模型的 AdaptiveSensor 下游目标表面、M3、支持保持、二进制与实际浏览器交互已经验证。旧全场景点云交付语义错误，原产物与原成本保留为撤回的对照。', '',
        '默认统一入口已改用 schema 3；环境与传感器预检通过。最终版完整无人干预冷启动没有重跑，不得将本次下游修复写成无异常端到端完成。', '',
        '## 选帧保持', '',
        '原始 8,421 帧 → 质量/粗位姿合格 7,906 帧 → 初选 1,151 帧（轨迹质量 743、位置/朝向补充 389、时间桥接 19）→ 视觉注册补入 46 帧 → 两路训练共用 1,197 帧。', '',
        '新默认保留全部 46 张补充帧。旧式 LiDAR 时间删帧仅会留下 8 张，现只用于显式旧契约对照。选帧判断、顺序和真实雷达数组回归已验证。', '',
        '## 传感器与目标表面', '',
        f"全部 {coverage['visual_frame_count']} 帧视觉观测保留；{coverage['lidar_contributing_frame_count']} 帧可参与稳定雷达约束；{coverage['time_unpaired_frame_count']} 帧无同步扫描，{coverage['paired_weak_anchor_frame_count']} 帧锚点不足，仅该帧雷达项 abstain。没有借用越界扫描或伪造锚点。", '',
        '旧每帧覆盖契约失败继续保留。新契约在独立目录评估前冻结，不等于旧门槛通过。独立扫描去重缓存与真实锚点数组核对通过。', '',
        '三候选实际评估后选择：`' + selection['selected_mode'] + '`；选择沿用原候选规则，不按最终外观手动挑选。', '',
        '本例雷达直接融合候选碎片更多，未作为最终表面。最终表面与此前显式视觉诊断相同；本次主要验证新选帧和完整候选后端的衔接，不能称为额外外观提升。', '',
        f"目标网格 {delivered['mesh_vertices']:,} 顶点、{delivered['mesh_triangles']:,} 三角形；显示采样 {delivered['surface_points']:,} 点；高斯 {delivered['gaussian_count']:,} 个。采样点数不代表新增几何观测。", '',
        '预览：' + delivered['target_url'], '', '旧样式预览：' + delivered['reference_style_url'], '',
        '实际交付目录：`' + delivered['delivery_directory'] + '`。', '',
        '## 异常与助手修复', '',
        f'原始运行保留 {n} 个原生失败过程及其诊断和人工干预记录。目标纠正另保留三个失败过程：全 1,197 帧的旧时间配对失败、时间支持子集仍违反旧锚点覆盖、Windows PowerShell 5.1 的 M3 启动参数异常。', '',
        '随后切换已安装 PowerShell 7，并适配新传感器契约，均是开发助手介入，不是原算法自主恢复。metric BA 原失败保留；已有纯 Sim(3) 恢复只支持米制接口。', '',
        '新前端子入口的导入冒烟另发现 augmentation_graph 模块路径缺失，已记录并由助手修复加载路径；实际模块与 CLI 回归通过不等于完整前端重跑。', '',
        '历史报告与验收回执：`' + str(archive) + '`。原始成本仍在历史报告，不能拿下游重用模型后的时间当作从零总耗时。', '',
        '## 本次下游成本', '', '|阶段|实际过程状态|秒数|', '|---|---|---:|']
    lines += [f'|{label}|{status}|{seconds:.3f}|' for label, status, seconds in stages]
    lines += ['', '这些是本次下游重用训练模型的阶段成本，包含目标 ROI、传感器读取和表面处理；不是完整从原始视频重新训练的公平性能对照。', '',
        '## 证据边界', '',
        '运行阶段只使用已有本地程序与文件，无大模型、云 API 或下载。Python 子进程和浏览器执行本机网络限制。首轮前端早于守卫启用的时间边界继续保留。', '',
        '稀疏支持、接触保持、像素投影与二进制读回属于工程自一致性，不能代替独立 GT。原始里程计生产者未确认；尺寸精度、新视角质量、多物体泛化、全部历史优化项对两条分支的有效性和移机部署均未充分验证。', '',
        '实际概览和俯视可见完整堆叠目标；目标底部仍有采样孔隙与粗糙区域，细文字模糊。外观证据见当前 QUALITY_REVIEW.json 和目标 browser_review/。', '',
        '本轮单对话分离自审，没有使用 subagent。', '']
    (root / 'RUN_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(target / 'CORRECTION_FINALIZATION.json', {'status': 'current_acceptance_corrected_prior_evidence_preserved',
        'archive': str(archive), 'report': str(root / 'RUN_REPORT.md'),
        'new_unassisted_cold_start_claimed': False})
    print('current target acceptance published', delivered['target_url'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run_root.resolve(strict=True), args.target_root.resolve(strict=True))
