"""Report target delivery scope, native stops, and manual interventions separately."""

from pathlib import Path
from collections import Counter
import math

from common import COLMAP, atomic_json, read_json
from metric_recovery_accounting import component_metric_alignments, metric_geometry_recoveries
from native_packing import packing_recovery_accounting


def registration_checks(side, initial_names):
    checks = []
    for label in ('sfm', 'recovery', 'temporal', 'pose_gap'):
        path = side / label / 'RESULT.json'
        if not path.is_file():
            continue
        value = read_json(path)
        flag = value.get('original_registration_guard_passed', value.get('all_initial_registered'))
        if flag is None:
            continue
        models = value.get('models', [])
        best = max(models, key=lambda row: row['registered_count']) if models else {}
        missing = set(best.get('original_failures_remaining', best.get('failed_images',
            best.get('remaining_failures', [])))) & set(initial_names)
        partial = side / label / 'TEMPORAL_RECOVERY_INPUT.json'
        checks.append({'stage': label, 'receipt': str(path), 'original_registration_guard_passed': flag,
            'required_initial_still_unregistered': sorted(missing),
            'partial_recovery_input_receipt': str(partial) if partial.is_file() else None,
            'candidate_evaluation_is_not_final_training_admission': True})
    path = side / 'export/TRAINING_ADMISSION.json'
    if path.is_file():
        value = read_json(path)
        checks.append({'stage': 'final_training_admission', 'receipt': str(path),
            'original_registration_guard_passed': not value['initial_not_registered'],
            'required_initial_still_unregistered': value['initial_not_registered'],
            'partial_recovery_input_receipt': None, 'candidate_evaluation_is_not_final_training_admission': False})
    return checks


def camera_recovery_stages(side):
    backend = side / 'camera_backend'
    for name, key in (('RESULT.json', 'stages'), ('FAILURE.json', 'stages'), ('STAGES.json', 'rows')):
        path = backend / name
        if path.is_file():
            return [{**{field: row[field] for field in ('label', 'returncode', 'seconds',
                    'expected_failure_recovery', 'failure_receipt') if field in row},
                'journal_receipt': str(path), 'scope': 'camera_input_preparation_not_final_BA_or_delivery'}
                for row in read_json(path)[key]]
    return []


def gpu_reboot_recoveries(root):
    """Keep prior-boot resource recovery separate from geometry and process failures."""
    rows = []
    for lease_path in sorted(Path(root).rglob('*GPU_LEASE*.json')):
        lease = read_json(lease_path)
        reference = lease.get('offline_reboot_owner_recovery_receipt')
        if reference is None:
            continue
        proof_path = Path(reference).resolve(strict=True)
        proof = read_json(proof_path)
        required_true = ('exclusive_byte_lock_acquired', 'all_recorded_identities_before_current_boot',
            'all_recorded_identities_ended', 'native_COLMAP_only')
        required_false = ('WSL_process_termination_inferred', 'original_lease_receipt_modified',
            'original_failed_or_interrupted_attempt_became_pass', 'runtime_language_model_used',
            'manual_intervention_used', 'original_process_exit_codes_known')
        identities = proof.get('recorded_process_identities', [])
        boot = proof.get('current_windows_boot_unix')
        valid = (proof.get('status') == 'offline_previous_boot_native_GPU_owner_reclaimed'
            and Path(proof['new_lease_receipt']).resolve(strict=True) == lease_path.resolve(strict=True)
            and all(proof.get(key) is True for key in required_true)
            and all(proof.get(key) is False for key in required_false)
            and isinstance(boot, (int, float)) and math.isfinite(boot) and identities
            and all(0 < row['creation_time'] < boot for row in identities))
        original = read_json(proof['original_lease_receipt'])
        owner = proof['original_owner']
        valid = bool(valid and all(original.get(key) == owner.get(key) for key in
            ('status', 'pid', 'process_birth', 'receipt', 'label', 'source', 'acquired_unix'))
            and original['status'] == 'acquired' and len(identities) >= 2
            and any(row['pid'] == owner['pid'] and row['creation_time'] == owner['process_birth']
                    for row in identities))
        rows.append({'current_lease_receipt': str(lease_path), 'recovery_receipt': str(proof_path),
            'original_interrupted_lease_receipt': proof['original_lease_receipt'],
            'current_windows_boot_unix': boot,
            'recorded_offline_reboot_recovery_verified': valid,
            'original_interrupted_attempt_not_converted_to_pass': True})
    return rows


def gpu_interruption_recoveries(root):
    """Bind native-owner recovery to the unchanged failed supervisor and lease."""
    rows = []
    for lease_path in sorted(Path(root).rglob('*GPU_LEASE*.json')):
        lease = read_json(lease_path)
        reference = lease.get('offline_interrupted_owner_recovery_receipt')
        if reference is None:
            continue
        proof_path = Path(reference).resolve(strict=True)
        proof = read_json(proof_path)
        original = read_json(proof['original_lease_receipt'])
        guard = read_json(proof['failed_guard_receipt'])
        lifecycle = read_json(proof['native_command_evidence'])
        owner, identities = proof['original_owner'], proof.get('recorded_process_identities', [])
        pair = {(owner['pid'], owner['process_birth']), (original['child_pid'], original['child_birth'])}
        owned = {(row['pid'], row['creation_time']) for row in guard.get('owned_process_identities', [])}
        boot = proof.get('current_windows_boot_unix')
        required_true = ('exclusive_byte_lock_acquired', 'all_recorded_identities_ended',
            'all_recorded_identities_in_current_boot', 'failed_supervisor_owned_native_parent_and_child',
            'native_COLMAP_only')
        required_false = ('original_lease_receipt_modified', 'original_failed_guard_modified',
            'WSL_process_termination_inferred', 'original_native_command_exit_code_known',
            'original_failed_or_interrupted_attempt_became_pass', 'runtime_language_model_used',
            'manual_intervention_used')
        valid = (proof.get('status') == 'offline_interrupted_native_GPU_owner_reclaimed'
            and Path(proof['new_lease_receipt']).resolve(strict=True) == lease_path.resolve(strict=True)
            and all(proof.get(key) is True for key in required_true)
            and all(proof.get(key) is False for key in required_false)
            and original == proof['original_lease'] and original['status'] == 'acquired'
            and guard == proof['original_failed_guard'] and guard['status'] == 'failed_controller'
            and guard.get('returncode') not in (None, 0) and pair <= owned
            and lifecycle == proof['original_native_lifecycle']
            and lifecycle['status'] == 'running' and lifecycle['command'] == original['command']
            and lifecycle['child_pid'] == original['child_pid']
            and proof['native_commands'] == [original['command']]
            and Path(original['command'][0]).is_file() and Path(original['command'][0]).samefile(COLMAP)
            and isinstance(boot, (int, float)) and math.isfinite(boot) and len(identities) >= 2
            and pair <= {(row['pid'], row['creation_time']) for row in identities}
            and all(boot <= row['creation_time'] <= proof['recovered_unix'] for row in identities)
            and all(original.get(key) == owner.get(key) for key in
                ('status', 'pid', 'process_birth', 'receipt', 'label', 'source', 'acquired_unix')))
        rows.append({'current_lease_receipt': str(lease_path), 'recovery_receipt': str(proof_path),
            'original_interrupted_lease_receipt': proof['original_lease_receipt'],
            'original_failed_guard_receipt': proof['failed_guard_receipt'],
            'recorded_offline_interruption_recovery_verified': bool(valid),
            'original_failed_attempt_not_converted_to_pass': True})
    return rows


def heartbeat_publication_recoveries(root):
    """Require immutable heartbeat failures and the actual final guard counters."""
    root = Path(root).resolve(strict=True)
    groups = {}
    for path in sorted(root.rglob('*.HEARTBEAT_SHARING_*.json')):
        proof = read_json(path)
        receipt = Path(proof['process_receipt']).resolve(strict=True)
        receipt.relative_to(root)
        groups.setdefault(receipt, []).append((path, proof))
    rows = []
    for receipt, events in groups.items():
        final = read_json(receipt)
        stats = final.get('receipt_publication_before_current_write', {})
        valid = (final.get('status') in ('completed', 'failed_child', 'failed_controller')
            and stats.get('deferred_heartbeat_publications') == len(events)
            and stats.get('terminal_publication_failure_is_fatal') is True
            and stats.get('original_process_guard_unchanged') is True
            and stats.get('last_deferred_heartbeat_receipt') in [str(path.resolve()) for path, _ in events])
        for path, proof in events:
            evidence = proof.get('actual_sharing_evidence', {})
            valid = bool(valid and proof.get('status') == 'offline_heartbeat_publication_deferred_final_receipt_required'
                and path.name.startswith(receipt.name + '.HEARTBEAT_SHARING_')
                and proof.get('previous_complete_running_receipt_exists') is True
                and proof.get('final_receipt_publication_still_required') is True
                and proof.get('runtime_language_model_used') is False
                and proof.get('manual_intervention_used') is False
                and evidence.get('verified_delete_sharing_conflict') is True
                and evidence.get('delete_access_probe_winerror') in (32, 33)
                and Path(evidence['path']).resolve(strict=True) == receipt
                and proof['maximum_seconds_unchanged'] == final['maximum_seconds']
                and proof['maximum_RSS_bytes_unchanged'] == final['maximum_RSS_bytes'])
        rows.append({'process_receipt': str(receipt), 'original_error_receipts': [str(path) for path, _ in events],
            'deferred_heartbeat_publications': len(events), 'final_status': final.get('status'),
            'original_heartbeat_errors_preserved_and_final_receipt_verified': valid})
    return rows


def focus_numerical_recoveries(root):
    """Preserve the rejected focus boundary and verify unchanged original constraints."""
    path = Path(root) / 'target/vehicle_roi_receipt.json'
    if not path.is_file():
        return []
    roi = read_json(path)
    fit = roi.get('roi_derivation', {}).get('forward_focus_fit', {})
    correction = fit.get('numerical_constraint_correction', {})
    if correction.get('used') is not True:
        return []
    before = fit.get('original_candidate_constraint_evidence', {})
    tolerance = correction.get('original_guard_tolerance_m', 0)
    near = fit.get('renderer_near_plane_m', 0)
    shift = correction.get('actual_maximum_focus_component_shift_m', float('inf'))
    valid = (roi['status'] == 'complete' and fit.get('solver_success') is True
        and correction.get('LP_success') is True and correction.get('all_selected_halfspaces_retained') is True
        and all(correction.get(key) is False for key in ('original_guard_tolerance_relaxed',
            'camera_denominator_reduced', 'camera_inputs_or_directions_modified', 'manual_focus_coordinates_used'))
        and fit.get('required_forward_fraction') == .95
        and fit.get('camera_count') == roi['roi_derivation']['camera_count'] == before.get('camera_count')
        and fit.get('required_positive_count') == before.get('required_positive_count')
        and fit.get('actual_positive_count', 0) >= fit.get('required_positive_count', 1)
        and tolerance == before.get('original_guard_tolerance_m') and near == before.get('renderer_near_plane_m')
        and isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance > 0
        and before.get('original_maximum_near_plane_deficit_m', 0) > tolerance
        and correction.get('original_maximum_near_plane_deficit_m') == before['original_maximum_near_plane_deficit_m']
        and 0 < shift <= correction.get('maximum_numerical_shift_m', 0)
        and correction.get('actual_minimum_selected_depth_m', 0) >= near - tolerance
        and fit.get('minimum_constrained_forward_depth_m', 0) >= near - tolerance)
    return [{'roi_receipt': str(path), 'original_focus_boundary_rejection_preserved': True,
        'original_maximum_near_plane_deficit_m': before.get('original_maximum_near_plane_deficit_m'),
        'original_guard_tolerance_m': tolerance, 'actual_focus_component_shift_m': shift,
        'original_fraction_camera_denominator_and_near_plane_preserved': bool(valid),
        'predefined_numerical_focus_correction_verified': bool(valid)}]


def destination_admission_recoveries(side):
    """Bind final image-space rejection and paired admission to the unchanged policy."""
    folder = Path(side) / 'export'
    path = folder / 'DESTINATION_JOINT_ADMISSION.json'
    if not path.is_file():
        return []
    value = read_json(path)
    if value.get('used') is not True:
        return []
    admission = read_json(folder / 'TRAINING_ADMISSION.json')
    rejection_path = folder / 'ORIGINAL_DESTINATION_SUPPORT_REJECTION.json'
    original = read_json(rejection_path)
    policy = read_json(admission['policy']['path'])
    camera, points = policy['camera_geometry_admission'], policy['point_filter']
    retained = {row['image_name'] for row in admission['rows'] if row['training_role'] == 'admitted'}
    excluded = {row['image_name'] for row in admission['rows'] if row['training_role'] == 'excluded_insufficient_support'}
    fraction = len(retained) / admission['full_registered_denominator']
    valid = (value['status'] == 'source_and_destination_camera_point_admission_stable'
        and value['protocol'] == 'camera_point_destination_admission_v3'
        and admission['status'] == 'export_and_admission_verified_training_pending'
        and admission.get('destination_joint_admission') == value
        and original['status'] == 'original_final_image_support_rejected_not_relabelled'
        and value['original_candidate_rejection'] == original['rejected_support']
        and original['thresholds_relaxed'] is False and value['thresholds_unchanged'] is True
        and value['complete_registered_denominator_unchanged'] is True
        and value['full_registered_denominator'] == admission['full_registered_denominator'] == original['full_registered_denominator']
        and value['original_camera_policy'] == original['original_camera_policy'] == camera
        and value['original_point_policy'] == original['original_point_policy'] == points
        and value['retained_cameras'] == len(retained) == admission['training_count']
        and value['retained_points'] == admission['training_points'] >= points['minimum_retained_sparse_points']
        and value['retained_observations'] == admission['training_observations']
        and value['retained_fraction_of_full_map'] == fraction >= camera['minimum_retained_fraction_of_sim3_inliers']
        and value['minimum_full_map_retained_fraction'] == camera['minimum_retained_fraction_of_sim3_inliers']
        and len(retained) >= camera['minimum_retained_cameras']
        and not retained & excluded and len(retained | excluded) == admission['full_registered_denominator']
        and excluded == set(admission['excluded_names']) and set(value['new_excluded_names']) <= excluded
        and value['original_final_support_guard_passed'] is False
        and value['original_support_failure_not_relabelled'] is True
        and value['source_and_destination_support_passed'] is True
        and value['paired_observation_identity_verified'] is True
        and all(value[key] is False for key in ('full_metric_model_modified',
            'camera_poses_or_point_coordinates_modified', 'source_or_delivered_calibration_modified',
            'new_observations_invented', 'training_admission_granted', 'runtime_language_model_used')))
    return [{'admission_receipt': str(path), 'original_rejection_receipt': str(rejection_path),
        'original_rejected_camera_count': len(value['original_candidate_rejection']),
        'new_excluded_camera_count': len(value['new_excluded_names']),
        'full_registered_denominator': admission['full_registered_denominator'],
        'retained_cameras': len(retained), 'retained_fraction': fraction,
        'original_policy_and_rejection_binding_verified': bool(valid)}]


def ground_support_sampling_recovery(target, raw, run_root):
    """Retain failed sampled evidence and verify a bounded same-source full pass."""
    path = target / '09_ground_target_m3/ground_plane_receipt.json'
    if not path.is_file():
        return None, True
    support = read_json(path).get('fit', {}).get('registered_lidar_spatial_support') or {}
    attempts = support.get('support_sampling_attempts', [])
    if not attempts and not support.get('automatic_same_source_evidence_expansion'):
        return None, True
    valid = (support.get('automatic_same_source_evidence_expansion') is True
        and isinstance(attempts, list) and len(attempts) == 2
        and all(isinstance(row, dict) for row in attempts))
    if valid:
        first, final = attempts
        gates = ('minimum_supported_frames', 'minimum_points_per_supported_frame',
            'minimum_near_plane_points', 'near_plane_distance_m',
            'near_sparse_ground_height_half_window_m', 'minimum_xy_extent_m',
            'minimum_roi_span_fraction', 'grid_cell_m', 'minimum_occupied_area_m2')
        valid = (first.get('status') == 'insufficient_support' and final.get('status') == 'accepted'
            and first.get('sample_every_nth_frame') == 10 and final.get('sample_every_nth_frame') == 1
            and all(first.get(key) == final.get(key) and key in first for key in gates)
            and first.get('frame_count') == final.get('frame_count')
            and first.get('coordinate_frame') == final.get('coordinate_frame')
            and Path(first['source_mcap']).resolve() == Path(final['source_mcap']).resolve() == Path(raw).resolve()
            and final['supported_frame_count'] >= final['minimum_supported_frames']
            and final['near_plane_point_count'] >= final['minimum_near_plane_points']
            and final['supported_frame_count'] <= final['sampled_frame_count'] <= final['frame_count']
            and len(final['xy_extent_m']) == 2 and len(final['roi_span_fraction']) == 2
            and all(math.isfinite(v) and v >= final['minimum_xy_extent_m'] for v in final['xy_extent_m'])
            and all(math.isfinite(v) and v >= final['minimum_roi_span_fraction'] for v in final['roi_span_fraction'])
            and final['occupied_area_m2'] >= final['minimum_occupied_area_m2'])
        summary_keys = ('status', 'sample_every_nth_frame', 'frame_count', 'sampled_frame_count',
            'supported_frame_count', 'near_plane_point_count', 'source_mcap', 'coordinate_frame')
        valid = bool(valid and all(support.get(key) == final.get(key) for key in summary_keys))
        if final.get('source_kind') == 'same_attempt_raw_motion_registered_scans':
            valid = bool(valid and first.get('source_registered_scan_receipt') == final.get('source_registered_scan_receipt')
                and first.get('source_registered_scan_receipt_sha256') == final.get('source_registered_scan_receipt_sha256')
                and Path(final['source_run_root']).resolve() == Path(run_root).resolve()
                and final['sampled_frame_count'] == final['eligible_scan_count']
                and final['frame_count'] == final['raw_scan_count'])
    return {'ground_plane_receipt': str(path), 'original_sampled_support_insufficient': bool(
            isinstance(attempts, list) and attempts and isinstance(attempts[0], dict)
            and attempts[0].get('status') == 'insufficient_support'),
        'actual_sampling_attempts': attempts, 'same_source_full_evidence_recovery_verified': bool(valid),
        'quality_sample_rejection_is_not_a_native_process_stop': True,
        'original_support_thresholds_not_relaxed': bool(valid)}, bool(valid)


def review(root):
    root = Path(root).resolve(strict=True)
    config, state = read_json(root / 'CONFIG.json'), read_json(root / 'STATE.json')
    receipts = list((root / 'stages').rglob('PROCESS.json'))
    target = root / 'target'
    if (target / 'stages').is_dir():
        receipts += list((target / 'stages').rglob('PROCESS.json'))
    side = root / config['case'] / 'optimized'
    receipts += list(side.glob('*_process.json'))
    if (root / 'failures').is_dir():
        receipts += list((root / 'failures').rglob('PROCESS.json'))
        receipts += list((root / 'failures').rglob('*_process.json'))
    failures = []
    for path in sorted(set(receipts)):
        item = read_json(path)
        if item['status'].startswith('failed'):
            metric_result = path.parent / 'metric/RESULT.json'
            predefined_recovery = (path.name == 'metric_process.json' and metric_result.is_file()
                and read_json(metric_result).get('status') == 'failed_projection_validation')
            failures.append({'receipt': str(path), 'status': item['status'],
                'returncode': item.get('returncode'), 'elapsed_seconds': item.get('elapsed_seconds'),
                'predefined_metric_recovery_branch': predefined_recovery})
    interventions = [str(path) for path in root.glob('MANUAL_INTERVENTION_*.json')]
    interventions += [str(path) for path in target.glob('MANUAL_INTERVENTION_*.json')]
    result_path = target / 'DELIVERY_RESULT.json'
    result = read_json(result_path) if result_path.is_file() else None
    complete = result is not None and result['status'] in (
        'dual_target_outputs_and_viewers_verified', 'visual_target_diagnostic_and_dual_viewers_verified',
        'adaptive_sensor_dual_target_outputs_and_viewers_verified') and not (
            state['status'].startswith('failed') or state['status'] == 'running')
    strict = complete and result['strict_ReliableLidar_profile_completed']
    adaptive = complete and result.get('adaptive_sensor_contract_completed', False)
    stage_costs = []
    for path in sorted((root / 'stages').glob('*/PROCESS.json')):
        item = read_json(path)
        stage_costs.append({'stage': path.parent.name, 'status': item['status'],
            'seconds': item.get('elapsed_seconds', 0), 'peak_tree_RSS_bytes': item.get('peak_tree_RSS_bytes'),
            'receipt': str(path)})
    archived_costs = []
    for path in sorted((root / 'failures').glob('*/retained/stages/*/PROCESS.json')):
        item = read_json(path)
        archived_costs.append({'stage': path.parent.name, 'status': item['status'],
            'seconds': item.get('elapsed_seconds', 0), 'receipt': str(path)})
    diagnostic_costs = []
    diagnostic_failures = []
    for path in sorted((root / 'diagnostics/stages').glob('*/PROCESS.json')):
        item = read_json(path)
        row = {'stage': path.parent.name, 'status': item['status'],
            'seconds': item.get('elapsed_seconds', 0), 'returncode': item.get('returncode'),
            'receipt': str(path), 'not_an_unassisted_pipeline_attempt': True}
        diagnostic_costs.append(row)
        if item['status'].startswith('failed'):
            diagnostic_failures.append(row)
    initial_path = side / 'initial/INITIAL_FRAME_MANIFEST.json'
    selection = None
    initial_names = set()
    if initial_path.is_file():
        initial = read_json(initial_path)
        initial_names = {row['image_name'] for row in initial['selected_frames']}
        selection = {'raw_frames': initial['receipt']['raw_frame_count'],
            'eligible_frames': initial['receipt']['quality_and_coarse_pose_eligible_count'],
            'initial_frames': initial['receipt']['selected_count'],
            'initial_selection_reasons': dict(Counter(row['selection_reason'] for row in initial['selected_frames']))}
    gate_checks = registration_checks(side, initial_names)
    backend_stages = camera_recovery_stages(side)
    metric_recoveries = metric_geometry_recoveries(side)
    component_alignments = component_metric_alignments(side)
    component_alignment_verified = bool(component_alignments and all(
        row['recorded_component_metric_guards_passed'] for row in component_alignments))
    metric_recovery_path_verified = bool(not metric_recoveries or all(
        row['recorded_original_metric_recovery_guards_passed'] for row in metric_recoveries)
        or component_alignment_verified)
    gpu_recoveries = gpu_reboot_recoveries(root)
    gpu_recoveries_verified = all(row['recorded_offline_reboot_recovery_verified'] for row in gpu_recoveries)
    interrupted_gpu_recoveries = gpu_interruption_recoveries(root)
    interrupted_gpu_verified = all(row['recorded_offline_interruption_recovery_verified']
        for row in interrupted_gpu_recoveries)
    heartbeat_recoveries = heartbeat_publication_recoveries(root)
    heartbeat_verified = all(row['original_heartbeat_errors_preserved_and_final_receipt_verified']
        for row in heartbeat_recoveries)
    focus_recoveries = focus_numerical_recoveries(root)
    focus_verified = all(row['predefined_numerical_focus_correction_verified'] for row in focus_recoveries)
    destination_recoveries = destination_admission_recoveries(side)
    destination_verified = all(row['original_policy_and_rejection_binding_verified'] for row in destination_recoveries)
    ground_recovery, ground_recovery_verified = ground_support_sampling_recovery(target, config['raw'], root)
    browser_attempts_path = target / 'browser_review/ATTEMPTS.json'
    browser_attempts = read_json(browser_attempts_path)['attempts'] if browser_attempts_path.is_file() else []
    browser_failures = [row for row in browser_attempts if row['returncode']]
    browser_recovery_verified = all(row.get('returncode') == 1
        and row.get('retryable_initialization') is True and row.get('owned_tree_drained') is True
        for row in browser_failures)
    packing = packing_recovery_accounting(root)
    ledger = {'status': 'dual_target_outputs_checked' if complete else 'target_delivery_incomplete',
        'native_process_failures': failures, 'manual_interventions': interventions,
        'original_initial_registration_checks': gate_checks,
        'camera_backend_substage_processes': backend_stages,
        'offline_metric_geometry_recoveries': metric_recoveries,
        'offline_component_metric_alignments': component_alignments,
        'predefined_component_metric_alignment_verified': component_alignment_verified,
        'predefined_metric_recovery_path_verified': metric_recovery_path_verified,
        'offline_gpu_owner_reboot_recoveries': gpu_recoveries,
        'predefined_gpu_owner_reboot_recovery_verified': gpu_recoveries_verified,
        'offline_gpu_owner_interruption_recoveries': interrupted_gpu_recoveries,
        'predefined_gpu_owner_interruption_recovery_verified': interrupted_gpu_verified,
        'offline_heartbeat_publication_recoveries': heartbeat_recoveries,
        'predefined_heartbeat_publication_recovery_verified': heartbeat_verified,
        'offline_focus_numerical_corrections': focus_recoveries,
        'predefined_focus_numerical_correction_verified': focus_verified,
        'offline_destination_camera_admission': destination_recoveries,
        'predefined_destination_camera_admission_verified': destination_verified,
        'offline_ground_support_evidence_expansion': ground_recovery,
        'predefined_ground_support_evidence_expansion_verified': ground_recovery_verified,
        'offline_browser_review_attempts': browser_attempts,
        'original_browser_initialization_failures': browser_failures,
        'predefined_browser_initialization_recovery_verified': browser_recovery_verified,
        'offline_native_packing_attempts': packing['attempts'],
        'original_native_packing_metadata_fetch_failures': packing['failures'],
        'predefined_native_packing_metadata_fetch_recovery_verified': packing['verified'],
        'explicit_native_packing_retry_contract_present': packing['explicit_retry_contract_present'],
        'known_original_metric_guard_failure_count': len(metric_recoveries),
        'zero_failure_claim_allowed': not (failures or metric_recoveries or gpu_recoveries
            or interrupted_gpu_recoveries or heartbeat_recoveries or focus_recoveries or destination_recoveries or browser_failures or packing['failures'] or ground_recovery
            or any(not row['original_registration_guard_passed'] for row in gate_checks)
            or any(row['returncode'] for row in backend_stages)),
        'failed_quality_checks_and_expected_recovery_are_not_erased_by_zero_controller_stops': True,
        'nested_propagation_records_are_not_independent_root_causes': True,
        'top_level_stage_costs': stage_costs, 'stage_costs_include_nested_work_do_not_sum_children_again': True,
        'archived_top_level_stage_costs': archived_costs,
        'separate_development_diagnostic_costs': diagnostic_costs,
        'separate_development_diagnostic_failures': diagnostic_failures,
        'measured_development_diagnostic_seconds': sum(row['seconds'] for row in diagnostic_costs),
        'development_diagnostics_not_counted_as_unassisted_pipeline_runtime': True,
        'measured_current_attempt_stage_seconds': sum(row['seconds'] for row in stage_costs),
        'measured_owned_stage_seconds': sum(row['seconds'] for row in stage_costs + archived_costs),
        'controller_wall_seconds': state.get('wall_seconds'), 'frame_selection': selection,
        'strict_ReliableLidar_profile_completed': strict, 'delivery_result': str(result_path) if complete else None,
        'adaptive_sensor_contract_completed': adaptive,
        'unassisted_single_raw_to_delivery_verified': bool((strict or adaptive) and not interventions
            and state['status'] == 'dual_target_outputs_complete' and state.get('controller_attempt_count') == 1
            and browser_recovery_verified
            and packing['verified']
            and gpu_recoveries_verified
            and interrupted_gpu_verified and heartbeat_verified and focus_verified and destination_verified
            and ground_recovery_verified
            and metric_recovery_path_verified),
        'independent_geometry_accuracy_verified': False,
        'failed_BA_can_be_recovered_by_preexisting_explicit_Sim3_policy_but_is_not_a_BA_pass':
            component_alignment_verified}
    atomic_json(root / 'EXCEPTION_LEDGER.json', ledger)
    lines = ['# 目标表面双输出运行记录', '', '当前状态：`' + state['status'] + '`。', '',
        '输入：`' + config['raw'] + '`。目标点云必须来自数值 ROI 和 M3 去地面的观测表面，不能用全场景 TSDF 代替。', '',
        'AdaptiveSensor 路线完成：' + str(adaptive) + '。严格 ReliableLidar 路线完成：' + str(strict)
        + '。独立几何 GT 验收：False。', '',
        '原生失败过程记录：' + str(len(failures)) + '；人工/开发助手干预记录：' + str(len(interventions))
        + '。嵌套传播可能对应同一根因，不把过程记录数量当作独立缺陷数。', '',
        '完整初选注册尚未通过的阶段检查：' + str(sum(not row['original_registration_guard_passed'] for row in gate_checks))
        + '；相机准备子过程非零退出：' + str(sum(bool(row['returncode']) for row in backend_stages))
        + '。这些含预编码恢复分支和同一缺失名单的阶段快照，不等于独立异常停止数；零个控制器停止也不表示所有检查均通过。', '',
        '本次单次无人干预完成：' + str(ledger['unassisted_single_raw_to_delivery_verified']) + '。', '',
        '原始米制候选检查失败：' + str(len(metric_recoveries))
        + '；允许宣称零失败：' + str(ledger['zero_failure_claim_allowed']) + '。自动恢复不抹去失败。', '',
        '粗里程计仅用于米制接口；相机中心拟合残差、体素大小和模型点数不能替代尺寸精度。', '']
    if selection:
        lines += ['原始图像 ' + str(selection['raw_frames']) + '；质量/粗位姿合格 '
            + str(selection['eligible_frames']) + '；初选 ' + str(selection['initial_frames']) + '。', '']
    lines += ['本次外层阶段实际耗时如下，包含子阶段，不重复累加嵌套计时：', '',
        '| 阶段 | 状态 | 秒 |', '|---|---|---:|']
    for row in stage_costs:
        lines += ['| ' + row['stage'] + ' | ' + row['status'] + ' | ' + format(row['seconds'], '.3f') + ' |']
    for row in archived_costs:
        lines += ['| ' + row['stage'] + '（保留的失败尝试） | ' + row['status'] + ' | ' + format(row['seconds'], '.3f') + ' |']
    lines += ['', '本根目录各次过程累计（包含归档失败尝试）：' + format(ledger['measured_owned_stage_seconds'], '.3f') + ' 秒。', '',
        '当前根控制器墙钟：' + str(ledger['controller_wall_seconds']) + ' 秒；此前失败目录的成本另记，不包含在本表。', '']
    if diagnostic_costs:
        lines += ['开发诊断另列，不能视为原流程预定义恢复，也不能隐藏失败分支成本：', '',
            '| 诊断 | 状态 | 秒 |', '|---|---|---:|']
        for row in diagnostic_costs:
            lines += ['| ' + row['stage'] + ' | ' + row['status'] + ' | ' + format(row['seconds'], '.3f') + ' |']
        lines += ['', '有过程回执的开发诊断累计：' + format(ledger['measured_development_diagnostic_seconds'], '.3f')
            + ' 秒。未有过程回执的代码阅读、编辑和局部检查不包含在该计时中。', '']
    for row in failures:
        lines += ['- `' + row['receipt'] + '`：' + row['status'] + '，退出码 ' + str(row['returncode']) + '。']
    for row in gpu_recoveries:
        lines += ['- 系统重启后的GPU所属记录离线恢复：`' + row['recovery_receipt']
            + '`；机器验证 ' + str(row['recorded_offline_reboot_recovery_verified'])
            + '；原中断回执 `' + row['original_interrupted_lease_receipt'] + '` 保留，不改为成功。']
    for row in interrupted_gpu_recoveries:
        lines += ['- 原生监控中断后的GPU所属记录离线恢复：`' + row['recovery_receipt']
            + '`；机器验证 ' + str(row['recorded_offline_interruption_recovery_verified'])
            + '；原中断租约和失败监控回执保持原文，不改为成功。']
    for row in heartbeat_recoveries:
        lines += ['- 状态文件真实共享冲突使心跳发布延后 ' + str(row['deferred_heartbeat_publications'])
            + ' 次；原错误与最终发布回读验证 '
            + str(row['original_heartbeat_errors_preserved_and_final_receipt_verified'])
            + '；实际最终回执 `' + row['process_receipt'] + '`。时间、内存及最终发布要求保持。']
    for row in focus_recoveries:
        lines += ['- 方向焦点原候选近裁面越界 ' + str(row['original_maximum_near_plane_deficit_m'])
            + ' m，原容差 ' + str(row['original_guard_tolerance_m']) + ' m；有界数值位移 '
            + str(row['actual_focus_component_shift_m']) + ' m；原相机分母、95%和近裁面检查保持，验证 '
            + str(row['predefined_numerical_focus_correction_verified']) + '。原候选不追溯为通过。']
    for row in destination_recoveries:
        lines += ['- 最终针孔表示原支撑拒绝 ' + str(row['original_rejected_camera_count'])
            + ' 张；按原源与目标表示的联合准入条件排除 ' + str(row['new_excluded_camera_count'])
            + ' 张，完整分母 ' + str(row['full_registered_denominator']) + '，保留比例 '
            + str(row['retained_fraction']) + '；原政策和拒绝绑定验证 '
            + str(row['original_policy_and_rejection_binding_verified']) + '。原候选不追溯为通过。']
    for row in gate_checks:
        if not row['original_registration_guard_passed']:
            lines += ['- `' + row['receipt'] + '`：完整初选注册未通过，必需初选仍缺 '
                + str(len(row['required_initial_still_unregistered'])) + ' 张；原失败保留。']
    for row in backend_stages:
        if row['returncode']:
            lines += ['- 相机准备 `' + row['label'] + '`：退出码 ' + str(row['returncode'])
                + '；预编码恢复 ' + str(row.get('expected_failure_recovery', False))
                + '；子过程记录 `' + row['journal_receipt'] + '`。']
    for row in packing['failures']:
        lines += ['- 原生预览打包的本机元数据读取失败：`' + row['failure_receipt']
            + '`；原记录复核 ' + str(row['actual_raw_failure_revalidated'])
            + '；所属树结束 ' + str(row['owned_tree_drained'])
            + '。有限重试不代表传输根因已证明。', '']
    for row in browser_failures:
        lines += ['- 预览初始化原始失败：退出码 ' + str(row['returncode'])
            + '；离线有限重试资格 ' + str(row['retryable_initialization'])
            + '；失败回执 `' + row['failure_receipt'] + '`。恢复不证明根因已修复。']
    for row in metric_recoveries:
        lines += ['- 原始米制候选 `' + row['original_failure_receipt'] + '`：投影无效 '
            + str(row['original_invalid_projections']) + '，最大相机半径 '
            + str(row['original_maximum_camera_radius_m']) + ' m；物理/重新三角化恢复 '
            + row['recovery_status'] + '，实际位置先验 ' + str(row['actual_position_priors'])
            + '，另外的物理深度残差 ' + str(row['separate_physical_depth_residuals'])
            + '。失败与恢复耗时已包含在父级米制阶段，不能重复累加。']
    for row in component_alignments:
        lines += ['- 分量 Sim(3) 米制规范 `' + row['component_receipt'] + '`：验证 '
            + str(row['recorded_component_metric_guards_passed']) + '，完整注册相机 '
            + str(row['full_registered_cameras']) + '，实际米制分量相机 '
            + str(row['metric_gauged_registered_cameras']) + '，非训练见证相机 '
            + str(row['nontraining_witness_cameras']) + '，保留观测 '
            + str(row['retained_observations']) + '，最大重投影变化 '
            + str(row['maximum_reprojection_error_delta_px'])
            + ' px。原视觉 BA 仍不算通过；此恢复不构成独立精度证明。']
    if ground_recovery:
        attempts = ground_recovery['actual_sampling_attempts']
        lines += ['', '地面支撑曾抽样不足，原始与完整证据检查均保留：`'
            + ground_recovery['ground_plane_receipt'] + '`。同来源恢复验证：'
            + str(ground_recovery_verified) + '；抽样质量拒绝不等于原生过程异常停止。', '']
        for row in attempts:
            lines += ['- 每' + str(row['sample_every_nth_frame']) + '扫描检查：' + row['status']
                + '；实测支撑扫描' + str(row['supported_frame_count'])
                + '；近地面点' + str(row['near_plane_point_count'])
                + '。独立证据重新计数，不累加重复扫描，不改变门槛。']
    if complete:
        lines += ['', '目标点云预览：' + result['target_url'], '',
            '目标表面顶点：' + str(result['mesh_vertices']) + '；三角形：' + str(result['mesh_triangles'])
            + '；显示采样点：' + str(result['surface_points']) + '。', '',
            '本机自由移动交互回执：`' + result['actual_browser_behavior'] + '`。']
    if interventions:
        lines += ['', '人工/开发助手干预原记录：', '']
        lines += ['- `' + path + '`。' for path in interventions]
    (root / 'RUN_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return ledger
