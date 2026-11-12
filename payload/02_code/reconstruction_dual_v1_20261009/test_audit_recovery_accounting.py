"""Retain expected recovery failures even when the overall controller finishes."""

from pathlib import Path
import tempfile
import unittest

from audit_target_run import (registration_checks, camera_recovery_stages, gpu_reboot_recoveries,
    gpu_interruption_recoveries, heartbeat_publication_recoveries, review)
from audit_target_run import focus_numerical_recoveries, destination_admission_recoveries
from common import COLMAP, atomic_json, read_json
from metric_recovery_accounting import component_metric_alignments, metric_geometry_recoveries


class RecoveryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.side = Path(self.scratch.name)

    def write(self, name, value):
        path = self.side / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)

    def test_optional_failures_do_not_replace_required_initial_failure_count(self):
        self.write('temporal/RESULT.json', {'original_registration_guard_passed': False,
            'models': [{'registered_count': 5, 'remaining_failures': ['b', 'optional'],
                'original_failures_remaining': ['b']}]})
        self.write('temporal/TEMPORAL_RECOVERY_INPUT.json', {'scope': 'recovery_input'})
        value = registration_checks(self.side, {'a', 'b'})[0]
        self.assertFalse(value['original_registration_guard_passed'])
        self.assertEqual(value['required_initial_still_unregistered'], ['b'])
        self.assertIsNotNone(value['partial_recovery_input_receipt'])

    def test_final_pass_keeps_preceding_failure_snapshot(self):
        self.write('sfm/RESULT.json', {'all_initial_registered': False,
            'models': [{'registered_count': 1, 'failed_images': ['b']}]})
        self.write('export/TRAINING_ADMISSION.json', {'initial_not_registered': []})
        checks = registration_checks(self.side, {'a', 'b'})
        self.assertEqual([row['original_registration_guard_passed'] for row in checks], [False, True])

    def test_failed_journal_keeps_last_failing_substage(self):
        success = {'label': 'matching', 'returncode': 0, 'seconds': 1, 'expected_failure_recovery': False}
        failure = {'label': 'triangulation', 'returncode': 1, 'seconds': 2, 'expected_failure_recovery': False}
        self.write('camera_backend/STAGES.json', {'rows': [success]})
        self.write('camera_backend/FAILURE.json', {'stages': [success, failure]})
        self.assertEqual([row['returncode'] for row in camera_recovery_stages(self.side)], [0, 1])

    def test_completed_recovery_keeps_expected_PnP_failure(self):
        self.write('camera_backend/RESULT.json', {'stages': [{'label': 'genuine_pnp', 'returncode': 1,
            'seconds': 2, 'expected_failure_recovery': True, 'failure_receipt': 'UNVERIFIED_FRAMES.json'},
            {'label': 'bridge', 'returncode': 0, 'seconds': 1, 'expected_failure_recovery': False}]})
        rows = camera_recovery_stages(self.side)
        self.assertTrue(rows[0]['expected_failure_recovery'])
        self.assertEqual(rows[0]['returncode'], 1)
        self.assertEqual(rows[1]['returncode'], 0)

    def recovery(self, **changes):
        self.write('metric/CONTRACT.json', {
            'offline_physical_recovery_protocol': changes.get('protocol',
                'sensor_position_warmup_active_depth_joint_BA_v1')})
        self.write('metric/weak_position_prior/ORIGINAL_CANDIDATE_FAILURE.json',
            {'status': 'original_candidate_rejected_known_physical_or_extent_failure',
                'factor_count_verified': True, 'active_priors': 23,
                'projection': {'invalid_count': 0}, 'maximum_camera_radius_m': 36.})
        self.write('metric/weak_position_prior/offline_physical_recovery/RESULT.json',
            {'status': 'completed_original_metric_guards',
                'protocol': 'sensor_position_warmup_active_depth_joint_BA_v1',
                'factor_count_verified': True, 'common_observations_preserved': True,
                'native_complete_metric_coordinate_lifecycle_used': True,
                'active_position_priors': 23, 'physical_depth_residuals': 7, **changes})

    def test_absent_original_failure_adds_no_recovery(self):
        self.assertEqual(metric_geometry_recoveries(self.side), [])

    def test_success_preserves_original_failure_and_dynamic_prior_count(self):
        self.recovery()
        value = metric_geometry_recoveries(self.side)[0]
        self.assertTrue(value['original_failure_preserved'])
        self.assertTrue(value['recorded_original_metric_recovery_guards_passed'])
        self.assertEqual(value['actual_position_priors'], 23)
        self.assertEqual(value['separate_physical_depth_residuals'], 7)
        self.assertTrue(value['failure_and_recovery_costs_already_in_parent_metric_stage_total'])

    def test_current_v3_recovery_is_bound_to_its_actual_metric_contract(self):
        self.recovery(protocol='sensor_position_warmup_active_depth_joint_BA_v3')
        value = metric_geometry_recoveries(self.side)[0]
        self.assertTrue(value['recorded_original_metric_recovery_guards_passed'])
        self.assertEqual(value['declared_recovery_protocol'], value['actual_recovery_protocol'])
        self.assertTrue(value['recovery_protocol_matches_metric_contract'])

    def test_recovery_with_missing_or_mismatched_protocol_contract_is_not_accepted(self):
        self.recovery(protocol='sensor_position_warmup_active_depth_joint_BA_v3')
        self.write('metric/CONTRACT.json', {})
        self.assertFalse(metric_geometry_recoveries(self.side)[0]['recorded_original_metric_recovery_guards_passed'])
        self.write('metric/CONTRACT.json', {
            'offline_physical_recovery_protocol': 'sensor_position_warmup_active_depth_joint_BA_v1'})
        self.assertFalse(metric_geometry_recoveries(self.side)[0]['recorded_original_metric_recovery_guards_passed'])

    def test_mismatched_prior_count_or_missing_native_world_restoration_is_rejected(self):
        for field, value in [('active_position_priors', 22),
                ('native_complete_metric_coordinate_lifecycle_used', False)]:
            with self.subTest(field=field):
                self.recovery(**{field: value})
                self.assertFalse(metric_geometry_recoveries(self.side)[0][
                    'recorded_original_metric_recovery_guards_passed'])

    def test_incomplete_recovery_keeps_original_failure(self):
        self.recovery(status='failed')
        value = metric_geometry_recoveries(self.side)[0]
        self.assertTrue(value['original_failure_preserved'])
        self.assertFalse(value['recorded_original_metric_recovery_guards_passed'])

    def component_alignment(self):
        result_path = self.side / 'metric/weak_position_prior/offline_component_metric_alignment/RESULT.json'
        self.write('metric/CONTRACT.json', {
            'offline_metric_recovery_protocol': 'visual_track_component_metric_gauge_v1'})
        self.write('metric/weak_position_prior/ORIGINAL_CANDIDATE_FAILURE.json', {
            'status': 'original_candidate_rejected_known_physical_or_extent_failure',
            'factor_count_verified': True, 'active_priors': 23,
            'projection': {'invalid_count': 3}, 'maximum_camera_radius_m': 31.})
        self.write('metric/RESULT.json', {
            'selected': 'component_metric_aligned', 'development_guard_passed': True,
            'results': {'component_metric_aligned': {
                'factor_count_verified': True,
                'retained_observation_identity_preserved': True,
                'new_observations_invented': False,
                'offline_component_metric_alignment': {
                    'protocol': 'visual_track_component_metric_gauge_v1',
                    'result': str(result_path)}}}})
        self.write('metric/weak_position_prior/offline_component_metric_alignment/RESULT.json', {
            'status': 'completed_component_metric_guards',
            'protocol': 'visual_track_component_metric_gauge_v1',
            'development_guard_passed': True,
            'retained_observation_identity_preserved': True,
            'new_observations_invented': False,
            'invalid_projection_count': 0,
            'source_geometry_unchanged': True,
            'full_registered_denominator_preserved': True,
            'thresholds_relaxed': False,
            'absolute_accuracy_claim': False,
            'GT_used': False,
            'source_visual_geometry_authority': 'verified_aligned_input_before_metric_BA',
            'retained_registered_cameras': 1226,
            'metric_gauged_registered_cameras': 1201,
            'nontraining_witness_camera_names': ['witness'],
            'retained_observations': 1096032,
            'maximum_reprojection_error_delta_px': 1.7e-10})

    def test_component_alignment_is_distinct_from_unrun_physical_recovery(self):
        self.component_alignment()
        physical = metric_geometry_recoveries(self.side)[0]
        component = component_metric_alignments(self.side)[0]
        self.assertEqual(physical['recovery_status'], 'not_completed')
        self.assertTrue(component['recorded_component_metric_guards_passed'])
        self.assertEqual(component['metric_gauged_registered_cameras'], 1201)
        self.assertEqual(component['nontraining_witness_cameras'], 1)

    def test_verified_component_alignment_can_complete_unassisted_without_erasing_BA_failure(self):
        root = self.side / 'component_delivery_fixture'
        self.side = root / 'case' / 'optimized'
        self.component_alignment()
        self.side = root
        self.write('CONFIG.json', {'case': 'case', 'raw': 'metadata_fixture.mcap'})
        self.write('STATE.json', {'status': 'dual_target_outputs_complete', 'controller_attempt_count': 1})
        self.write('target/DELIVERY_RESULT.json', {
            'status': 'adaptive_sensor_dual_target_outputs_and_viewers_verified',
            'strict_ReliableLidar_profile_completed': False,
            'adaptive_sensor_contract_completed': True,
            'target_url': 'http://127.0.0.1/fixture', 'mesh_vertices': 1,
            'mesh_triangles': 1, 'surface_points': 1,
            'actual_browser_behavior': 'metadata_fixture.json'})
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_component_metric_alignment_verified'])
        report = (self.side / 'RUN_REPORT.md').read_text(encoding='utf-8')
        self.assertIn('物理/重新三角化恢复 not_completed', report)
        self.assertIn('分量 Sim(3) 米制规范', report)

    def test_completed_target_can_be_unassisted_but_cannot_be_zero_failure(self):
        root = self.side / 'delivery_fixture'
        self.side = root / 'case' / 'optimized'
        self.recovery()
        self.side = root
        self.write('CONFIG.json', {'case': 'case', 'raw': 'metadata_fixture.mcap'})
        self.write('STATE.json', {'status': 'dual_target_outputs_complete', 'controller_attempt_count': 1})
        self.write('target/DELIVERY_RESULT.json',
            {'status': 'adaptive_sensor_dual_target_outputs_and_viewers_verified',
                'strict_ReliableLidar_profile_completed': False, 'adaptive_sensor_contract_completed': True,
                'target_url': 'http://127.0.0.1/fixture', 'mesh_vertices': 1, 'mesh_triangles': 1,
                'surface_points': 1, 'actual_browser_behavior': 'metadata_fixture.json'})
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertEqual(result['known_original_metric_guard_failure_count'], 1)

    def test_completed_browser_retry_retains_exit_one_and_blocks_zero_failure_claim(self):
        self.write('CONFIG.json', {'case': 'case', 'raw': 'metadata_fixture.mcap'})
        self.write('STATE.json', {'status': 'dual_target_outputs_complete', 'controller_attempt_count': 1})
        self.write('target/DELIVERY_RESULT.json',
            {'status': 'adaptive_sensor_dual_target_outputs_and_viewers_verified',
                'strict_ReliableLidar_profile_completed': False, 'adaptive_sensor_contract_completed': True,
                'target_url': 'http://127.0.0.1/fixture', 'mesh_vertices': 1, 'mesh_triangles': 1,
                'surface_points': 1, 'actual_browser_behavior': 'metadata_fixture.json'})
        self.write('target/browser_review/ATTEMPTS.json', {'attempts': [
            {'returncode': 1, 'owned_tree_drained': True, 'retryable_initialization': True,
                'failure_receipt': 'original_initialization_failure.json'}, {'returncode': 0}]})
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertEqual(result['original_browser_initialization_failures'][0]['returncode'], 1)
        self.write('target/browser_review/ATTEMPTS.json', {'attempts': [
            {'returncode': 1, 'owned_tree_drained': False, 'retryable_initialization': True,
                'failure_receipt': 'original_initialization_failure.json'}, {'returncode': 0}]})
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])

    def reboot_delivery_fixture(self):
        self.write('CONFIG.json', {'case': 'case', 'raw': 'metadata_fixture.mcap'})
        self.write('STATE.json', {'status': 'dual_target_outputs_complete', 'controller_attempt_count': 1})
        self.write('target/DELIVERY_RESULT.json',
            {'status': 'adaptive_sensor_dual_target_outputs_and_viewers_verified',
                'strict_ReliableLidar_profile_completed': False, 'adaptive_sensor_contract_completed': True,
                'target_url': 'http://127.0.0.1/fixture', 'mesh_vertices': 1, 'mesh_triangles': 1,
                'surface_points': 1, 'actual_browser_behavior': 'metadata_fixture.json'})
        old = self.side / 'original_interrupted_lease.json'
        current = self.side / 'features_GPU_LEASE.json'
        proof = self.side / 'reboot_recovery.json'
        owner = {'status': 'acquired', 'pid': 900001, 'process_birth': 50., 'label': 'features',
            'receipt': str(old), 'source': {'path': 'native_gpu_lease.py'}, 'acquired_unix': 60.}
        atomic_json(old, owner)
        atomic_json(current, {'status': 'released', 'offline_reboot_owner_recovery_receipt': str(proof)})
        record = {'status': 'offline_previous_boot_native_GPU_owner_reclaimed',
            'new_lease_receipt': str(current), 'original_lease_receipt': str(old), 'original_owner': owner,
            'current_windows_boot_unix': 100., 'recorded_process_identities': [
                {'pid': 900001, 'creation_time': 50.}, {'pid': 900002, 'creation_time': 55.}]}
        for key in ('exclusive_byte_lock_acquired', 'all_recorded_identities_before_current_boot',
                'all_recorded_identities_ended', 'native_COLMAP_only'):
            record[key] = True
        for key in ('WSL_process_termination_inferred', 'original_lease_receipt_modified',
                'original_failed_or_interrupted_attempt_became_pass', 'runtime_language_model_used',
                'manual_intervention_used', 'original_process_exit_codes_known'):
            record[key] = False
        atomic_json(proof, record)
        return old, current, proof, record

    def test_reboot_recovery_is_recorded_without_converting_the_interrupted_lease(self):
        old, _, _, _ = self.reboot_delivery_fixture()
        original = old.read_bytes()
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_gpu_owner_reboot_recovery_verified'])
        self.assertEqual(len(result['offline_gpu_owner_reboot_recoveries']), 1)
        self.assertEqual(old.read_bytes(), original)
        self.assertIn('GPU', (self.side / 'RUN_REPORT.md').read_text(encoding='utf-8'))

    def test_unverified_reboot_or_other_lease_binding_blocks_unassisted_completion(self):
        _, _, proof, record = self.reboot_delivery_fixture()
        record['WSL_process_termination_inferred'] = True
        atomic_json(proof, record)
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])
        record['WSL_process_termination_inferred'] = False
        other = self.side / 'other_GPU_LEASE.json'
        atomic_json(other, {'status': 'released'})
        record['new_lease_receipt'] = str(other)
        atomic_json(proof, record)
        self.assertFalse(gpu_reboot_recoveries(self.side)[0]['recorded_offline_reboot_recovery_verified'])
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])

    def heartbeat_fixture(self):
        _, current, _, _ = self.reboot_delivery_fixture()
        atomic_json(current, {'status': 'released'})
        process = self.side / 'stages/front/PROCESS.json'
        event = process.with_name(process.name + '.HEARTBEAT_SHARING_1.json')
        self.write('stages/front/PROCESS.json', {'status': 'completed', 'returncode': 0,
            'elapsed_seconds': 1, 'maximum_seconds': 12, 'maximum_RSS_bytes': 1024**3,
            'receipt_publication_before_current_write': {'deferred_heartbeat_publications': 1,
                'terminal_publication_failure_is_fatal': True, 'original_process_guard_unchanged': True,
                'last_deferred_heartbeat_receipt': str(event)}})
        value = {'status': 'offline_heartbeat_publication_deferred_final_receipt_required',
            'process_receipt': str(process), 'previous_complete_running_receipt_exists': True,
            'final_receipt_publication_still_required': True, 'runtime_language_model_used': False,
            'manual_intervention_used': False, 'maximum_seconds_unchanged': 12,
            'maximum_RSS_bytes_unchanged': 1024**3, 'actual_sharing_evidence': {
                'verified_delete_sharing_conflict': True, 'delete_access_probe_winerror': 32,
                'path': str(process)}}
        atomic_json(event, value)
        return process, event, value

    def test_heartbeat_recovery_is_recorded_and_blocks_zero_failure_claim(self):
        self.heartbeat_fixture()
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_heartbeat_publication_recovery_verified'])
        self.assertEqual(len(result['offline_heartbeat_publication_recoveries']), 1)
        self.assertIn('心跳', (self.side / 'RUN_REPORT.md').read_text(encoding='utf-8'))

    def test_heartbeat_unverified_sharing_or_missing_final_counter_blocks_completion(self):
        process, event, value = self.heartbeat_fixture()
        value['actual_sharing_evidence']['verified_delete_sharing_conflict'] = False
        atomic_json(event, value)
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])
        value['actual_sharing_evidence']['verified_delete_sharing_conflict'] = True
        atomic_json(event, value)
        final = read_json(process)
        final['receipt_publication_before_current_write']['deferred_heartbeat_publications'] = 0
        atomic_json(process, final)
        self.assertFalse(heartbeat_publication_recoveries(self.side)[0]
            ['original_heartbeat_errors_preserved_and_final_receipt_verified'])

    def interrupted_fixture(self):
        old, current, proof, value = self.reboot_delivery_fixture()
        owner = value['original_owner']
        original = dict(owner, child_pid=900002, child_birth=55., command=[str(COLMAP), 'matches_importer'])
        guard = self.side / 'original_failed_guard.json'
        lifecycle = self.side / 'original_native_lifecycle.json'
        failed = {'status': 'failed_controller', 'returncode': 15, 'owned_process_identities': [
            {'pid': 900001, 'creation_time': 50.}, {'pid': 900002, 'creation_time': 55.}]}
        native = {'status': 'running', 'command': original['command'], 'child_pid': 900002}
        atomic_json(old, original)
        atomic_json(guard, failed)
        atomic_json(lifecycle, native)
        value.update(status='offline_interrupted_native_GPU_owner_reclaimed', original_lease=original,
            failed_guard_receipt=str(guard), original_failed_guard=failed, native_command_evidence=str(lifecycle),
            original_native_lifecycle=native, native_commands=[original['command']],
            current_windows_boot_unix=10., recovered_unix=100.)
        for key in ('all_recorded_identities_in_current_boot', 'failed_supervisor_owned_native_parent_and_child'):
            value[key] = True
        for key in ('original_failed_guard_modified', 'original_native_command_exit_code_known'):
            value[key] = False
        atomic_json(proof, value)
        atomic_json(current, {'status': 'released', 'offline_interrupted_owner_recovery_receipt': str(proof)})
        return old, guard, proof, value

    def test_interrupted_owner_recovery_preserves_failed_guard_and_original_lease(self):
        old, guard, _, _ = self.interrupted_fixture()
        payload, failed = old.read_bytes(), guard.read_bytes()
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_gpu_owner_interruption_recovery_verified'])
        self.assertEqual(old.read_bytes(), payload)
        self.assertEqual(guard.read_bytes(), failed)

    def test_interrupted_recovery_must_match_unchanged_failed_guard(self):
        _, guard, _, _ = self.interrupted_fixture()
        changed = read_json(guard)
        changed['returncode'] = 0
        atomic_json(guard, changed)
        self.assertFalse(gpu_interruption_recoveries(self.side)[0]['recorded_offline_interruption_recovery_verified'])
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])

    def numerical_focus_fixture(self):
        _, current, _, _ = self.reboot_delivery_fixture()
        atomic_json(current, {'status': 'released'})
        correction = {'used': True, 'LP_success': True, 'all_selected_halfspaces_retained': True,
            'original_guard_tolerance_relaxed': False, 'camera_denominator_reduced': False,
            'camera_inputs_or_directions_modified': False, 'manual_focus_coordinates_used': False,
            'original_guard_tolerance_m': 1e-13, 'original_maximum_near_plane_deficit_m': 1e-12,
            'actual_maximum_focus_component_shift_m': 2e-12, 'maximum_numerical_shift_m': 1e-8,
            'actual_minimum_selected_depth_m': .01 + 1e-13}
        before = {'camera_count': 20, 'required_positive_count': 19, 'renderer_near_plane_m': .01,
            'original_guard_tolerance_m': 1e-13, 'original_maximum_near_plane_deficit_m': 1e-12}
        value = {'status': 'complete', 'roi_derivation': {'camera_count': 20, 'forward_focus_fit': {
            'camera_count': 20, 'required_forward_fraction': .95, 'required_positive_count': 19,
            'actual_positive_count': 19, 'renderer_near_plane_m': .01, 'solver_success': True,
            'minimum_constrained_forward_depth_m': .01 + 1e-13,
            'numerical_constraint_correction': correction, 'original_candidate_constraint_evidence': before}}}
        self.write('target/vehicle_roi_receipt.json', value)
        return value

    def test_numerical_focus_correction_is_recorded_and_not_called_zero_failure(self):
        self.numerical_focus_fixture()
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_focus_numerical_correction_verified'])
        self.assertEqual(len(result['offline_focus_numerical_corrections']), 1)
        self.assertIn('数值位移', (self.side / 'RUN_REPORT.md').read_text(encoding='utf-8'))

    def test_relaxed_or_unbound_focus_correction_blocks_unassisted_completion(self):
        value = self.numerical_focus_fixture()
        correction = value['roi_derivation']['forward_focus_fit']['numerical_constraint_correction']
        correction['original_guard_tolerance_relaxed'] = True
        self.write('target/vehicle_roi_receipt.json', value)
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])
        correction['original_guard_tolerance_relaxed'] = False
        correction['actual_maximum_focus_component_shift_m'] = 1.
        self.write('target/vehicle_roi_receipt.json', value)
        self.assertFalse(focus_numerical_recoveries(self.side)[0]['predefined_numerical_focus_correction_verified'])


    def destination_admission_fixture(self):
        _, current, _, _ = self.reboot_delivery_fixture()
        atomic_json(current, {'status': 'released'})
        camera = {'minimum_retained_cameras': 32, 'minimum_retained_fraction_of_sim3_inliers': .9,
                  'minimum_triangulated_points': 30, 'minimum_occupied_grid_cells': 4}
        points = {'minimum_retained_sparse_points': 10000, 'minimum_point_track_length': 3}
        policy_path = self.side / 'original_destination_policy.json'
        atomic_json(policy_path, {'camera_geometry_admission': camera, 'point_filter': points})
        rejection = {'failed': {'source': {'passed': True}, 'destination': {'passed': False}}}
        value = {'status': 'source_and_destination_camera_point_admission_stable',
            'protocol': 'camera_point_destination_admission_v3', 'used': True,
            'full_registered_denominator': 40, 'retained_cameras': 39, 'retained_points': 10001,
            'retained_observations': 40000, 'retained_fraction_of_full_map': 39 / 40,
            'minimum_full_map_retained_fraction': .9, 'original_camera_policy': camera,
            'original_point_policy': points, 'original_final_support_guard_passed': False,
            'original_support_failure_not_relabelled': True, 'original_candidate_rejection': rejection,
            'new_excluded_names': ['failed'], 'thresholds_unchanged': True,
            'complete_registered_denominator_unchanged': True, 'source_and_destination_support_passed': True,
            'paired_observation_identity_verified': True, 'full_metric_model_modified': False,
            'camera_poses_or_point_coordinates_modified': False, 'source_or_delivered_calibration_modified': False,
            'new_observations_invented': False, 'training_admission_granted': False,
            'runtime_language_model_used': False}
        prefix = 'case/optimized/export/'
        self.write(prefix + 'DESTINATION_JOINT_ADMISSION.json', value)
        self.write(prefix + 'ORIGINAL_DESTINATION_SUPPORT_REJECTION.json', {
            'status': 'original_final_image_support_rejected_not_relabelled', 'full_registered_denominator': 40,
            'original_camera_policy': camera, 'original_point_policy': points,
            'rejected_support': rejection, 'thresholds_relaxed': False})
        admission = {'status': 'export_and_admission_verified_training_pending',
            'initial_not_registered': [], 'full_registered_denominator': 40, 'training_count': 39,
            'training_points': 10001, 'training_observations': 40000, 'excluded_names': ['failed'],
            'destination_joint_admission': value, 'policy': {'path': str(policy_path)},
            'rows': [{'image_name': 'kept_' + str(index), 'training_role': 'admitted'} for index in range(39)]
                    + [{'image_name': 'failed', 'training_role': 'excluded_insufficient_support'}]}
        self.write(prefix + 'TRAINING_ADMISSION.json', admission)
        return value, admission

    def test_destination_rejection_is_preserved_and_cannot_be_called_zero_failure(self):
        self.destination_admission_fixture()
        result = review(self.side)
        self.assertTrue(result['unassisted_single_raw_to_delivery_verified'])
        self.assertFalse(result['zero_failure_claim_allowed'])
        self.assertTrue(result['predefined_destination_camera_admission_verified'])
        self.assertEqual(len(result['offline_destination_camera_admission']), 1)
        self.assertIn('最终针孔表示原支撑拒绝', (self.side / 'RUN_REPORT.md').read_text(encoding='utf-8'))

    def test_destination_threshold_change_or_mutated_coordinates_block_completion(self):
        value, admission = self.destination_admission_fixture()
        value['original_camera_policy']['minimum_occupied_grid_cells'] = 3
        self.write('case/optimized/export/DESTINATION_JOINT_ADMISSION.json', value)
        admission['destination_joint_admission'] = value
        self.write('case/optimized/export/TRAINING_ADMISSION.json', admission)
        self.assertFalse(review(self.side)['unassisted_single_raw_to_delivery_verified'])
        value['original_camera_policy']['minimum_occupied_grid_cells'] = 4
        value['camera_poses_or_point_coordinates_modified'] = True
        self.write('case/optimized/export/DESTINATION_JOINT_ADMISSION.json', value)
        admission['destination_joint_admission'] = value
        self.write('case/optimized/export/TRAINING_ADMISSION.json', admission)
        self.assertFalse(destination_admission_recoveries(self.side / 'case/optimized')[0]
                         ['original_policy_and_rejection_binding_verified'])


if __name__ == '__main__':
    unittest.main()
