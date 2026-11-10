"""Publish one scoped work acceptance after the actual review writer exited."""

import os
from pathlib import Path

import psutil

import close_native_zero_negative as parent


def main():
    out = parent.OUT / 'v7'
    review = parent.read(out / 'REVIEW_001.json')
    if not all(review['checks'].values()) or not parent.windows_closed({'pid': review['pid'], 'creation_time': review['creation_time']}):
        raise AssertionError('The complete separate review has not actually closed')
    master = parent.read(parent.ROOT / 'research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json')
    if master['additional_accepted_count'] != 37 or len(set(master['accepted_mechanisms'])) != 37:
        raise AssertionError('The parent acceptance frontier changed')
    if any('ZERO_FEATURE' in name or 'R157_' in name for name in master['accepted_mechanisms']):
        raise AssertionError('The native zero-feature family was already counted')
    fee = dict(review['V7']['fee_components'])
    seed = parent.read(parent.OUT.parent / 'oup015/p0_LEASE.json')
    if fee['related_original_seed015_diagnostic_seconds'] != seed['child_elapsed_seconds'] or seed['returncode'] != 0:
        raise AssertionError('The related seed diagnostic cost lacks its actual receipt')
    fee['actual_separate_review_tool_including_publication_exit_seconds'] = 3.8829483
    fee['actual_followup_metadata_tool_seconds'] = .224976 + .208
    total = sum(fee.values())
    metrics = dict(review['V7'])
    B, C = metrics['actual_native_HtoD_bytes_B_C']
    tB, tC = metrics['complete_whole_seconds_B_C']
    metrics.update(fee_components=fee, known_fees_plus_reserve_seconds=total,
                   charged_whole_ratio=(tC + total / 6) / (tB + total / 6),
                   net_work_index_under_registered_symmetric_execution_model=(B - C) / (B + total / 6 * B / tB))
    if not (metrics['raw_whole_ratio'] <= 1.03 and metrics['charged_whole_ratio'] <= 1.03
            and metrics['net_work_index_under_registered_symmetric_execution_model'] >= .10):
        raise AssertionError('The uniform V7 complete-cost boundary failed')
    result = {'status': 'scoped_optional_work_accepted_pending_external_publication_close',
              'mechanism': 'R157_COMPLETE_NATIVE_INITIALIZER_DEVICE_ZERO_FEATURE_WORK',
              'policy': 'V7_WORK_TRADEOFF_POLICY', 'metrics': metrics,
              'review': str(out / 'REVIEW_001.json'), 'actual_review_writer_closed': True,
              'new_optimization_count': 1, 'parent_accepted_count_before': 37, 'parent_accepted_count_after': 38,
              'goal_complete': False, 'production_default_deployed': False, 'full_speed_accepted': False,
              'historical_V5_V6_whole_failed_evidence_retained': True,
              'pid': os.getpid(), 'creation_time': psutil.Process().create_time(),
              'source': parent.data_identity(Path(__file__))}
    parent.atomic_json(out / 'ACCEPTANCE_WORK_V7_001.json', result)
    print({'status': result['status'], 'count': 38, 'net_work': metrics['net_work_index_under_registered_symmetric_execution_model'],
           'fees': total, 'raw_whole_ratio': metrics['raw_whole_ratio'], 'full_speed_accepted': False})


if __name__ == '__main__':
    main()
