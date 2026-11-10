"""Verify native report text, cost, parent state and closed saved publishers."""

from pathlib import Path
import re

import close_native_zero_negative as parent


def main():
    out = parent.OUT / 'v7'
    acceptance = parent.read(out / 'ACCEPTANCE_WORK_V7_001.json')
    review = parent.read(out / 'REVIEW_001.json')
    docs = parent.ROOT / 'research/2026-10-05-gaussian-optimization-round3-50'
    master = parent.read(docs / 'CURRENT_STATE.json')
    text = (docs / 'R157_V7_REASSESSMENT_RESULTS.md').read_text(encoding='utf-8')
    fee = dict(acceptance['metrics']['fee_components'])
    fee['actual_followup_metadata_tool_seconds'] = .224976 + .2047572 + .2256003
    fee['actual_separate_publisher_tool_seconds'] = .2582907
    total = sum(fee.values())
    metrics = dict(acceptance['metrics'])
    B, C = metrics['actual_native_HtoD_bytes_B_C']
    tB, tC = metrics['complete_whole_seconds_B_C']
    metrics.update(fee_components=fee, known_fees_plus_reserve_seconds=total,
                   charged_whole_ratio=(tC + total / 6) / (tB + total / 6),
                   net_work_index_under_registered_symmetric_execution_model=(B - C) / (B + total / 6 * B / tB))
    links = re.findall(r'\]\((C:/[^)]+)\)', text)
    checks = {'review_and_actual_publisher_saved_births_ended': all(parent.windows_closed(
              {'pid': value['pid'], 'creation_time': value['creation_time']}) for value in (review, acceptance)),
              'parent38_unique_mechanisms_and_receipts': master['additional_accepted_count'] == 38
              and len(set(master['accepted_mechanisms'])) == len(master['acceptance_receipts']) == 38,
              'all38_acceptance_receipts_resolve': all((parent.ROOT / path).is_file() for path in master['acceptance_receipts']),
              'all_report_local_links_resolve': bool(links) and all(Path(path).is_file() for path in links),
              'raw_net_work_and_charged_whole_V7_boundaries': metrics['raw_whole_ratio'] <= 1.03
              and metrics['charged_whole_ratio'] <= 1.03 and metrics['net_work_index_under_registered_symmetric_execution_model'] >= .10,
              'text_preserves_speed_slowdown_failures_scope_and_unrun_boundaries':
              all(word in text for word in ('慢2.141724%', '不接受速度项', '旧V5/V6', '源码与旧三完整对', '未默认部署', '未测研究/文本思考工时不是0')),
              'V7_uniform_policy_not_single_candidate_threshold': '适用于所有同类旧候选及后续候选' in (docs / 'V7_WORK_TRADEOFF_POLICY.md').read_text(encoding='utf-8'),
              'control_frontier_text_current38': all('38/50' in (docs / name).read_text(encoding='utf-8').splitlines()[2]
                                                   for name in ('PLAN.md', 'CANDIDATE_BOARD.md', 'OPTIMIZE_CHECKLIST.md')),
              'old_failed_original_results_preserved': parent.read(parent.OUT / 'c2/RESULT.json')['status'] == 'closed_negative_whole_median_guard0',
              'goal_remains_active_and_next_R163_not_accepted': not acceptance['goal_complete']
              and not any(name.startswith('R163_') for name in master['accepted_mechanisms'])}
    result = {'status': 'closed_scoped_optional_work_acceptance' if all(checks.values()) else 'failed',
              'checks': checks, 'metrics': metrics, 'local_links': links,
              'report_validation': 'complete_native_text_and_links_no_document_hash_screenshot_or_OCR',
              'new_optimization_count': 1, 'parent_accepted_count': 38, 'remaining': 12, 'goal_complete': False,
              'actual_publisher_birth': {'pid': acceptance['pid'], 'creation_time': acceptance['creation_time']},
              'source': parent.data_identity(Path(__file__))}
    parent.atomic_json(out / 'FINAL_POST_EXIT_V7_001.json', result)
    print({'status': result['status'], 'checks': checks, 'count': 38, 'fee': total,
           'net_work': metrics['net_work_index_under_registered_symmetric_execution_model']})
    if not all(checks.values()):
        raise AssertionError('A current native text, state or closure boundary failed')


if __name__ == '__main__':
    main()
