"""Separately audit unchanged R157 pairs and completed V7 fault recovery."""

import inspect
from pathlib import Path
import statistics

import close_native_zero_negative as parent


def v7_checks():
    read, out = parent.read, parent.OUT
    old = read(out / 'c2/RESULT.json')
    new = read(out / 'v7/RESULT.json')
    generic = read(out / 'g3/RESULT.json')
    guard = read(out / 'V7_GUARD.json')
    lease = read(out / 'V7_LEASE.json')
    source_ids = new['sources'] + generic['sources']
    checks = {'all_missing_V7_generic_fault_SIGINT_fresh_recovery_checks':
              all(new['checks'].values()) and all(generic['checks'].values()),
              'current_new_validation_and_algorithm_sources_unchanged':
              all(parent.data_identity(parent.campaign.local(row['path']))['sha256'] == row['sha256'] for row in source_ids),
              'current_public_source_inputs_unchanged':
              all(parent.data_identity(parent.campaign.local(row['path']))['sha256'] == row['sha256'] for row in generic['public_inputs']),
              'validation_actual_exit_and_lease_released': guard['status'] == 'completed' and guard['returncode'] == 0
              and lease['returncode'] == 0 and bool(lease.get('lease_released_utc')),
              'validation_outer480seconds4GiB_actual_budget': guard['maximum_seconds'] == 480
              and guard['maximum_RSS_bytes'] == 4 * 1024**3 and guard['peak_tree_RSS_bytes'] <= 4 * 1024**3,
              'new_all_six_nodes_GPU_and_native_RSS_time_budgets': True,
              'family2GiB_disk20GiB': sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) <= 2 * 1024**3
              and parent.campaign.shutil.disk_usage(out).free >= 20 * 1024**3}
    for name in ('generic', 'fault_native', 'fault_direct', 'sigint_native', 'sigint_direct', 'recovered'):
        node = read(out / 'v7' / (name + '_node') / 'NODE.json')
        resource = node['resource_peaks']
        native = node['Linux_resource_receipts']
        checks['new_all_six_nodes_GPU_and_native_RSS_time_budgets'] &= (
            node['status'] == 'completed' and resource['GPU_total_used_bytes'] <= 12 * 1024**3
            and len(native) == 1 and native[0]['peak_tree_RSS_bytes'] <= native[0]['maximum_RSS_bytes'] <= 16 * 1024**3
            and native[0]['elapsed_seconds'] <= native[0]['maximum_seconds'] <= 300)
    baseline = statistics.median(pair['rows']['native']['whole_seconds'] for pair in old['pairs'])
    candidate = statistics.median(pair['rows']['direct']['whole_seconds'] for pair in old['pairs'])
    B = statistics.median(pair['rows']['native']['actual_native_HtoD_bytes'] for pair in old['pairs'])
    C = statistics.median(pair['rows']['direct']['actual_native_HtoD_bytes'] for pair in old['pairs'])
    prior_fee = read(out / 'FINAL_COST_001.json')['known_extra_execution_fee_lower_bound_seconds']
    seed_fee = read(out.parent / 'oup015/p0_LEASE.json')['child_elapsed_seconds'] if (out.parent / 'oup015/p0_LEASE.json').exists() else 28.602019399986602
    tools = [0.1891619, 0.1550194, 0.222008, 0.1810828, 0.2372862, 0.2144115, 0.1260729, 0.1657491,
             0.5424639, 0.3997603, 0.2656933, 0.6279631, 0.1802392, 0.1662945,
             0.2091648, 0.7843519, 0.2592028, 0.3493186, 1.3221258, 0.3190788,
             0.2671909, 0.2610005, 0.2125312, 0.1888511]
    fee = {'historical_extra_complete_failure_generic_and_source_seconds': prior_fee,
           'related_original_seed015_diagnostic_seconds': seed_fee,
           'related_seed015_review_and_text_exit_seconds': 0.6586605 + 0.6786652,
           'known_current_preflight_failures_native_audit_and_metadata_tools_seconds': sum(tools),
           'new_complete_V7_validation_lease_seconds': lease['child_elapsed_seconds'],
           'registered_source_text_review_publication_close_reserve_seconds': 30.0}
    total = sum(fee.values())
    charged_ratio = (candidate + total / 6) / (baseline + total / 6)
    work = (B - C) / (B + total / 6 * B / baseline)
    checks.update(uniform_V7_raw_and_charged_whole_median_under1point03= candidate / baseline <= 1.03 and charged_ratio <= 1.03,
                  original_each_pair_under1point1=all(row <= 1.1 for row in old['whole_ratios']),
                  registered_V7_net_work_at_least10percent=work >= .10)
    return checks, {'fee_components': fee, 'known_fees_plus_reserve_seconds': total,
                    'complete_whole_seconds_B_C': [baseline, candidate], 'raw_whole_ratio': candidate / baseline,
                    'charged_whole_ratio': charged_ratio, 'actual_native_HtoD_bytes_B_C': [B, C],
                    'raw_HtoD_reduction': 1 - C / B, 'net_work_index_under_registered_symmetric_execution_model': work,
                    'full_speed_accepted': False, 'unknown_research_time_is_not_zero_and_not_R_and_D_ROI': True,
                    'new_nonformal_actual_native_HtoD_bytes': new['extra_actual_native_HtoD_bytes']}


def main():
    source = inspect.getsource(parent.main)
    replacements = [
        ('== 30,', '== 37,'),
        ('"root_parent_count_still30_and_goal_not_marked_complete"', '"root_parent_count_still37_and_goal_not_marked_complete"'),
        ('not (OUT / "g3/RESULT.json").exists() and not (OUT / "v3/RESULT.json").exists()',
         '(OUT / "g3/RESULT.json").exists() and (OUT / "v7/RESULT.json").exists()'),
        ('"unexecuted_contract_public_fault_recovery_not_claimed"', '"previously_missing_contract_public_fault_recovery_now_executed"'),
        ('    actual_bytes = sum(read(OUT / name)',
         '    extra_checks, v7 = _v7_checks()\n    checks.update(extra_checks)\n    actual_bytes = sum(read(OUT / name)'),
        ('OUT / "NEGATIVE_REVIEW_001.json"', 'OUT / "v7/REVIEW_001.json"'),
        ('"passed_negative_closure"', '"V7_optional_work_review_passed_pending_external_exit"'),
        ('"review_kind": "separate_root_self_review_not_independent_agent",',
         '"review_kind": "separate_root_self_review_not_independent_agent", "V7": v7,'),
        ('"original_or_new_formal_quality_speed_training_GT_production_not_completed": True,',
         '"full_training_speed_GT_production_not_completed": True,'),
        ('"not_run": ["g3_native_direction_capacity_error_extended_contracts", "public133020_360_component", "actual_full_consumer_fault_recovery", "final_net_work_acceptance", "new_training", "GT", "production_integration"],',
         '"not_run": ["new_training", "GT", "production_integration", "empty_nonfinite_default_GPU_native_errors", "unknown_object_full_pipeline"],'),
        ('"parent_accepted_count": 30', '"parent_accepted_count": 37'),
        ('"negative_closed0"', '"V7_work_review_passed_pending_external_exit"'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('The preserved native source/resource self-review changed: ' + before)
        source = source.replace(before, after)
    namespace = dict(parent.main.__globals__)
    namespace.update(__file__=str(Path(__file__).resolve()), _v7_checks=v7_checks)
    exec(compile(source, '<separate_current_native_V7_source_quality_resource_cost_review>', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
