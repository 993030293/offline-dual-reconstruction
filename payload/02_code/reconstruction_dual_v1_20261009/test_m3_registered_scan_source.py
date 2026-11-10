"""Regress ground support sources, independent scan gates, and cold-run binding."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from target_prepare import PACKAGE

sys.path.insert(0, str(PACKAGE / 'ground/code'))
import registered_lidar_ground_support as support
from run_target_pipeline import m3_registered_scan_arguments
from audit_target_run import ground_support_sampling_recovery


class RegisteredGroundSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'attempt'
        self.scans = self.root / 'raw_sensor_backend/scans'
        self.scans.mkdir(parents=True)
        self.raw = Path(self.temporary.name) / 'original.mcap'
        self.raw.write_bytes(b'synthetic original recording identity')
        identity = {'raw': str(self.raw.resolve()), 'raw_bytes': self.raw.stat().st_size,
                    'raw_mtime_ns': self.raw.stat().st_mtime_ns}
        self.config = dict(identity, GT_used=False, historical_derived_inputs_used=False)
        self.binding = dict(identity, status='raw_sensor_backend_selected',
                            backend='fresh_raw_gyro_translation_local_map',
                            runtime_language_model_used=False,
                            historical_motion_or_geometry_used=False)
        axis = np.linspace(-.49, .49, 21)
        x, y = np.meshgrid(axis, axis)
        self.points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
        self.rows = []
        for index in range(300):
            stamp = 1_000_000_000 + index * 100_000_000
            relative = 'scans/scan_' + str(stamp) + '.npz'
            np.savez_compressed(self.scans.parent / relative,
                                points_world_lio_m=self.points, lidar_timestamp_ns=stamp)
            self.rows.append({'index': index, 'scan_end_ns': stamp,
                              'eligible_registered_evidence': True,
                              'valid_raw_points': len(self.points), 'path': relative})
        self.receipt = dict(identity, status='fresh_raw_registered_scans_complete',
                            run_root=str(self.root.resolve()), topic='/sensing/lidar/points_raw',
                            source_frame='fresh_local_gyro_translation_lio_world',
                            raw_scans_read=300, eligible_scans=300, rejected_evidence_scans=0,
                            runtime_language_model_used=False, historical_derived_inputs_used=False,
                            SfM_used_for_scan_registration=False, GT_used=False, scans=self.rows)
        self.path = self.scans.parent / 'RESULT.json'
        self.publish()

    def publish(self):
        for path, value in [(self.root / 'CONFIG.json', self.config),
                            (self.root / 'SENSOR_BACKEND.json', self.binding),
                            (self.path, self.receipt)]:
            path.write_text(json.dumps(value), encoding='utf-8')

    def validate(self):
        return support.validate_generated_scan_source(self.raw, self.path, self.root)

    def ground(self):
        return support.validate_registered_lidar_support(
            self.raw, np.array([0., 0., 1.]), 0.,
            np.array([-.5, -.5, -.1]), np.array([.5, .5, .2]),
            0., .5, .6, .05, self.path, self.root)

    def test_full_measured_scans_pass_unchanged_independent_support_gates(self):
        value = self.ground()
        self.assertEqual(value['status'], 'accepted')
        self.assertEqual(value['frame_count'], 300)
        self.assertEqual(value['supported_frame_count'], 30)
        self.assertEqual(value['near_plane_point_count'], 30 * len(self.points))
        self.assertEqual(value['source_run_root'], str(self.root.resolve()))
        self.assertEqual(value['coordinate_frame'], 'fresh_local_gyro_translation_lio_world')
        self.assertFalse(value['independent_geometry_accuracy_verified'])

    def test_dense_points_in_only_29_independent_scans_remain_rejected(self):
        self.rows[9]['eligible_registered_evidence'] = False
        self.rows[9]['path'] = None
        self.receipt.update(eligible_scans=299, rejected_evidence_scans=1)
        self.publish()
        with self.assertRaisesRegex(RuntimeError, 'insufficient'):
            self.ground()

    def test_dense_narrow_floor_remains_rejected_by_spatial_gates(self):
        for row in self.rows[9::10]:
            np.savez_compressed(self.scans.parent / row['path'],
                                points_world_lio_m=self.points * .2,
                                lidar_timestamp_ns=row['scan_end_ns'])
        with self.assertRaisesRegex(RuntimeError, 'insufficient'):
            self.ground()

    def test_cross_attempt_receipt_is_rejected(self):
        other = Path(self.temporary.name) / 'other'
        other.mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            support.validate_generated_scan_source(self.raw, self.path, other)

    def test_cross_recording_identity_is_rejected(self):
        other = Path(self.temporary.name) / 'other.mcap'
        other.write_bytes(b'other recording')
        with self.assertRaisesRegex(ValueError, 'different original'):
            support.validate_generated_scan_source(other, self.path, self.root)

    def test_missing_machine_only_flag_is_rejected(self):
        del self.receipt['runtime_language_model_used']
        self.publish()
        with self.assertRaisesRegex(ValueError, 'machine-only'):
            self.validate()

    def test_non_boolean_machine_only_flag_is_rejected(self):
        self.receipt['historical_derived_inputs_used'] = 0
        self.publish()
        with self.assertRaisesRegex(ValueError, 'machine-only'):
            self.validate()

    def test_duplicate_timestamps_cannot_count_as_independent_scans(self):
        self.rows[1]['scan_end_ns'] = self.rows[0]['scan_end_ns']
        self.publish()
        with self.assertRaisesRegex(ValueError, 'unique ordered'):
            self.validate()

    def test_scan_path_cannot_escape_current_raw_backend(self):
        outside = self.root / 'foreign.npz'
        np.savez_compressed(outside, points_world_lio_m=self.points, lidar_timestamp_ns=1)
        self.rows[9]['path'] = '../foreign.npz'
        self.publish()
        with self.assertRaises(ValueError):
            self.validate()

    def test_sampled_scan_timestamp_is_checked_against_actual_file(self):
        row = self.rows[9]
        np.savez_compressed(self.scans.parent / row['path'],
                            points_world_lio_m=self.points, lidar_timestamp_ns=row['scan_end_ns'] + 1)
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.ground()

    def test_sampled_point_count_is_checked_against_actual_file(self):
        self.rows[9]['valid_raw_points'] += 1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'point data'):
            self.ground()

    def test_generated_pipeline_route_passes_bound_full_scan_receipt(self):
        self.assertEqual(m3_registered_scan_arguments(self.root, self.raw),
                         ['-RegisteredScanReceipt', self.path])

    def test_recorded_pipeline_route_keeps_original_MCAP_decoder(self):
        self.binding['backend'] = 'recorded_registered_lidar_and_odometry'
        self.publish()
        self.assertEqual(m3_registered_scan_arguments(self.root, self.raw), [])

    def test_unknown_pipeline_route_has_no_silent_recorded_fallback(self):
        self.binding['backend'] = 'unknown'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'compatible'):
            m3_registered_scan_arguments(self.root, self.raw)

    def recorded_ground(self, frame='odom', topic=support.TOPIC):
        points = self.points

        class Reader:
            def __init__(self, paths):
                self.connections = [SimpleNamespace(topic=topic, msgtype='PointCloud2')]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def messages(self, connections):
                for index in range(300):
                    self.current_index = index
                    yield self.connections[0], index, b'synthetic encoded scan'

            def deserialize(self, payload, message_type):
                return SimpleNamespace(header=SimpleNamespace(frame_id=frame,
                    stamp=SimpleNamespace(sec=self.current_index, nanosec=0)), points=points)

        highlevel = ModuleType('rosbags.highlevel')
        highlevel.AnyReader = Reader
        decoder = ModuleType('extract_stable_registered_lidar')
        decoder.decode_xyz = lambda message: message.points
        decoder.timestamp_ns = lambda stamp: stamp.sec * 1_000_000_000 + stamp.nanosec
        with patch.dict(sys.modules, {'rosbags.highlevel': highlevel,
                                     'extract_stable_registered_lidar': decoder}):
            return support.validate_registered_lidar_support(
                self.raw, np.array([0., 0., 1.]), 0.,
                np.array([-.5, -.5, -.1]), np.array([.5, .5, .2]), 0., .5, .6, .05)

    def test_recorded_support_retains_original_spatial_and_temporal_gates(self):
        value = self.recorded_ground()
        self.assertEqual(value['supported_frame_count'], 30)
        self.assertEqual(value['near_plane_point_count'], 30 * len(self.points))
        self.assertEqual(value['coordinate_frame'], 'odom')
        self.assertNotIn('source_registered_scan_receipt', value)

    def test_recorded_support_still_rejects_wrong_coordinate_frame(self):
        with self.assertRaisesRegex(RuntimeError, 'frame does not match'):
            self.recorded_ground(frame='camera')

    def test_recorded_support_still_rejects_absent_registered_topic(self):
        with self.assertRaisesRegex(RuntimeError, 'found 0'):
            self.recorded_ground(topic='/sensing/lidar/points_raw')

    def expanded_ground(self):
        return support.validate_registered_lidar_support(
            self.raw, np.array([0., 0., 1.]), 0.,
            np.array([-.5, -.5, -.1]), np.array([.5, .5, .2]),
            0., .5, .6, .05, self.path, self.root, expand_evidence=True)

    def test_sample_failure_expands_same_source_without_double_counting(self):
        for row in self.rows[199::10]:
            row['eligible_registered_evidence'] = False
            row['path'] = None
        self.receipt.update(eligible_scans=289, rejected_evidence_scans=11)
        self.publish()
        value = self.expanded_ground()
        self.assertEqual(value['status'], 'accepted')
        self.assertEqual(value['sample_every_nth_frame'], 1)
        self.assertEqual(value['supported_frame_count'], 289)
        original, full = value['support_sampling_attempts']
        self.assertEqual(original['status'], 'insufficient_support')
        self.assertEqual(original['supported_frame_count'], 19)
        self.assertEqual(full['source_registered_scan_receipt'], original['source_registered_scan_receipt'])
        self.assertEqual(full['minimum_supported_frames'], 30)

    def test_full_evidence_still_rejects_only_29_genuine_independent_scans(self):
        for row in self.rows[29:]:
            row['eligible_registered_evidence'] = False
            row['path'] = None
        self.receipt.update(eligible_scans=29, rejected_evidence_scans=271)
        self.publish()
        with self.assertRaisesRegex(RuntimeError, 'sampled and full-source'):
            self.expanded_ground()

    def test_full_evidence_still_rejects_genuinely_narrow_floor(self):
        for row in self.rows:
            np.savez_compressed(self.scans.parent / row['path'],
                                points_world_lio_m=self.points * .2,
                                lidar_timestamp_ns=row['scan_end_ns'])
        with self.assertRaisesRegex(RuntimeError, 'sampled and full-source'):
            self.expanded_ground()

    def write_accounting_recovery(self):
        for row in self.rows[199::10]:
            row['eligible_registered_evidence'] = False
            row['path'] = None
        self.receipt.update(eligible_scans=289, rejected_evidence_scans=11)
        self.publish()
        value = self.expanded_ground()
        target = self.root / 'target'
        path = target / '09_ground_target_m3/ground_plane_receipt.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'fit': {'registered_lidar_spatial_support': value}}), encoding='utf-8')
        return target, path, value

    def test_accounting_preserves_original_sampling_rejection(self):
        target, _, value = self.write_accounting_recovery()
        report, valid = ground_support_sampling_recovery(target, self.raw, self.root)
        self.assertTrue(valid)
        self.assertTrue(report['original_sampled_support_insufficient'])
        self.assertEqual(report['actual_sampling_attempts'][0]['supported_frame_count'], 19)
        self.assertEqual(report['actual_sampling_attempts'][1]['supported_frame_count'], 289)

    def test_accounting_rejects_lowered_recovery_threshold(self):
        target, path, value = self.write_accounting_recovery()
        value['support_sampling_attempts'][1]['minimum_supported_frames'] = 1
        path.write_text(json.dumps({'fit': {'registered_lidar_spatial_support': value}}), encoding='utf-8')
        _, valid = ground_support_sampling_recovery(target, self.raw, self.root)
        self.assertFalse(valid)

    def test_accounting_rejects_foreign_recovery_source_root(self):
        target, _, _ = self.write_accounting_recovery()
        _, valid = ground_support_sampling_recovery(target, self.raw, self.root.parent / 'other')
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
