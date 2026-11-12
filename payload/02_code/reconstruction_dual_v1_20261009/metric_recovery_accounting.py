"""Keep rejected metric candidates visible after preencoded numerical recovery."""

from pathlib import Path

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


def component_metric_alignments(side):
    """Report the selected component gauge separately from physical retriangulation."""
    metric_path = side / 'metric/RESULT.json'
    result_path = side / 'metric/weak_position_prior/offline_component_metric_alignment/RESULT.json'
    if not metric_path.is_file() or not result_path.is_file():
        return []
    metric = read_json(metric_path)
    selected = metric.get('results', {}).get('component_metric_aligned', {})
    reference = selected.get('offline_component_metric_alignment')
    if not isinstance(reference, dict):
        return []
    result = read_json(result_path)
    contract_path = side / 'metric/CONTRACT.json'
    contract = read_json(contract_path) if contract_path.is_file() else {}
    declared = contract.get('offline_metric_recovery_protocol')
    referenced = Path(reference.get('result', '')).resolve() == result_path.resolve()
    verified = bool(
        metric.get('selected') == 'component_metric_aligned'
        and metric.get('development_guard_passed') is True
        and selected.get('factor_count_verified') is True
        and selected.get('retained_observation_identity_preserved') is True
        and selected.get('new_observations_invented') is False
        and referenced
        and isinstance(declared, str)
        and reference.get('protocol') == declared == result.get('protocol')
        and result.get('status') == 'completed_component_metric_guards'
        and result.get('development_guard_passed') is True
        and result.get('retained_observation_identity_preserved') is True
        and result.get('new_observations_invented') is False
        and result.get('invalid_projection_count') == 0
        and result.get('source_geometry_unchanged') is True
        and result.get('full_registered_denominator_preserved') is True
        and result.get('thresholds_relaxed') is False
        and result.get('absolute_accuracy_claim') is False
        and result.get('GT_used') is False
    )
    return [{
        'metric_receipt': str(metric_path),
        'component_receipt': str(result_path),
        'declared_protocol': declared,
        'actual_protocol': result.get('protocol'),
        'selected_component_metric_alignment': metric.get('selected') == 'component_metric_aligned',
        'recorded_component_metric_guards_passed': verified,
        'source_visual_BA_passed': False,
        'source_visual_geometry_authority': result.get('source_visual_geometry_authority'),
        'full_registered_cameras': result.get('retained_registered_cameras'),
        'metric_gauged_registered_cameras': result.get('metric_gauged_registered_cameras'),
        'nontraining_witness_cameras': len(result.get('nontraining_witness_camera_names', [])),
        'retained_observations': result.get('retained_observations'),
        'maximum_reprojection_error_delta_px': result.get('maximum_reprojection_error_delta_px'),
        'invalid_projection_count': result.get('invalid_projection_count'),
        'thresholds_relaxed': result.get('thresholds_relaxed'),
        'absolute_accuracy_claim': result.get('absolute_accuracy_claim'),
    }]
