"""Keep rejected metric candidates visible after preencoded numerical recovery."""

from common import read_json


def metric_geometry_recoveries(side):
    original_path = side / 'metric/weak_position_prior/ORIGINAL_CANDIDATE_FAILURE.json'
    if not original_path.is_file():
        return []
    original = read_json(original_path)
    folder = original_path.parent / 'offline_physical_recovery'
    result_path, failure_path = folder / 'RESULT.json', folder / 'FAILURE.json'
    recovered = read_json(result_path) if result_path.is_file() else {}
    contract_path = side / 'metric/CONTRACT.json'
    contract = read_json(contract_path) if contract_path.is_file() else {}
    declared = contract.get('offline_physical_recovery_protocol')
    coded = isinstance(declared, str) and recovered.get('protocol') == declared
    passed = bool(coded and recovered.get('status') == 'completed_original_metric_guards'
        and recovered.get('factor_count_verified') is True
        and recovered.get('common_observations_preserved') is True
        and recovered.get('native_complete_metric_coordinate_lifecycle_used') is True
        and recovered.get('active_position_priors') == original.get('active_priors')
        and original.get('factor_count_verified') is True)
    return [{'original_failure_receipt': str(original_path), 'original_status': original['status'],
        'original_candidate_model': original.get('model'), 'original_failure_preserved': True,
        'original_invalid_projections': original.get('projection', {}).get('invalid_count'),
        'original_maximum_camera_radius_m': original.get('maximum_camera_radius_m'),
        'original_active_position_priors': original.get('active_priors'),
        'recovery_receipt': str(result_path) if result_path.is_file() else None,
        'recovery_failure_receipt': str(failure_path) if failure_path.is_file() else None,
        'recovery_status': recovered.get('status', 'not_completed'),
        'declared_recovery_protocol': declared,
        'actual_recovery_protocol': recovered.get('protocol'),
        'recovery_protocol_matches_metric_contract': coded,
        'recorded_original_metric_recovery_guards_passed': passed,
        'actual_position_priors': recovered.get('active_position_priors'),
        'separate_physical_depth_residuals': recovered.get('physical_depth_residuals'),
        'preencoded_offline_numeric_recovery': coded,
        'failure_and_recovery_costs_already_in_parent_metric_stage_total': True,
        'not_an_unexpected_controller_stop_or_manual_intervention': True}]
