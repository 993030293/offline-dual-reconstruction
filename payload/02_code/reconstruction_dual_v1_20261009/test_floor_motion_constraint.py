"""Regress actual plane constraints, deskew algebra and saved motion binding."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import register_raw_lidar as dense
import floor_motion_constraint as floor

TOOLS = dense.TOOLS
sys.path.insert(0, str(TOOLS))
import lio_local_map as native
import lio_baseline
import mcap_audit


def broad_ground():
    x, y = np.meshgrid(np.linspace(-1.7, 1.7, 45), np.linspace(-1.7, 1.7, 45))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.4)))


class MeasuredPlaneTests(unittest.TestCase):
    def test_minimal_normal_rotation_is_proper_and_preserves_plane_yaw_gauge(self):
        source = Rotation.from_euler('xy', [6., -4.], degrees=True).apply([0., 0., 1.])
        delta = floor.minimum_normal_rotation(source, [0., 0., 1.])
        np.testing.assert_allclose(delta.apply(source), [0., 0., 1.], atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(delta.as_matrix()), 1., places=14)
        self.assertAlmostEqual(delta.as_rotvec()[2], 0., places=14)

    def test_invalid_and_opposite_normals_are_rejected(self):
        for value in ([0., 0., -1.], [np.nan, 0., 1.], [0., 0., 0.]):
            with self.assertRaises(ValueError):
                floor.minimum_normal_rotation(value, [0., 0., 1.])

    def test_plane_equation_preserves_metric_translation_sign(self):
        n, target = floor.plane_translation_equation([0., 0., 1.], 1.65, 1.4)
        self.assertAlmostEqual(target, .25)
        self.assertAlmostEqual(np.dot(n, [1., 2., .25]) - target, 0.)
        with self.assertRaises(ValueError):
            floor.plane_translation_equation([0., 0., 2.], 1.65, 1.4)

    def test_actual_original_estimator_drives_three_scan_reference_and_new_motion(self):
        tracker = floor.FloorMotionConstraint()
        ground = broad_ground()
        dt = np.zeros(len(ground))
        for stamp in (1, 2):
            _, constraint = tracker.propose(ground, dt, np.zeros(3), Rotation.identity(), np.zeros(3), stamp)
            self.assertIsNone(constraint)
            self.assertFalse(tracker.accepted_translation(constraint, np.zeros(3)))
        _, constraint = tracker.propose(ground, dt, np.zeros(3), Rotation.identity(), np.zeros(3), 3)
        self.assertIsNotNone(constraint)
        self.assertTrue(tracker.accepted_translation(constraint, np.zeros(3)))
        for stamp, angles in ((4, [4., -2.]), (5, [-3., 5.])):
            motion = np.asarray([.3, -.1, .25])
            error = Rotation.from_euler('xy', angles, degrees=True)
            predicted = error.inv().apply(ground - motion)
            corrected_end, constraint = tracker.propose(predicted, dt, np.zeros(3),
                Rotation.identity(), np.zeros(3), stamp)
            self.assertIsNotNone(constraint)
            self.assertTrue(tracker.accepted_translation(constraint, motion))
            self.assertLess(abs(constraint['target_m'] - motion[2]), floor.MAXIMUM_INLIER_RESIDUAL_M)
            self.assertLess(np.ptp(corrected_end.apply(predicted)[:, 2]), floor.MAXIMUM_INLIER_RESIDUAL_M)
        receipt = tracker.receipt()
        self.assertEqual(receipt['actual_accepted_plane_motion_count'], 3)
        self.assertFalse(receipt['future_sequence_measurements_or_GT_used'])

    def test_support_absence_abstains_without_weakening_original_threshold(self):
        tracker = floor.FloorMotionConstraint()
        _, constraint = tracker.propose(np.zeros((20, 3)), np.zeros(20), np.zeros(3),
            Rotation.identity(), np.zeros(3), 1)
        self.assertIsNone(constraint)
        self.assertIn('Too few metric sparse floor candidates', tracker.rows[-1]['reason'])
        self.assertFalse(tracker.accepted_translation(constraint, np.zeros(3)))
        constraint = {'normal': np.asarray([0., 0., 1.]), 'target_m': 0.}
        self.assertTrue(tracker.accepted_translation(constraint, [0., 0., .035]))
        self.assertFalse(tracker.accepted_translation(constraint, [0., 0., np.nextafter(.035, np.inf)]))

    def test_duplicate_clock_and_unexpected_estimator_error_fail_closed(self):
        tracker = floor.FloorMotionConstraint()
        values = (np.zeros((20, 3)), np.zeros(20), np.zeros(3), Rotation.identity(), np.zeros(3), 1)
        tracker.propose(*values)
        with self.assertRaisesRegex(ValueError, 'unique increasing'):
            tracker.propose(*values)
        with patch.object(floor, 'estimate_sparse_floor', side_effect=RuntimeError('unexpected numerical failure')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected numerical'):
                floor.measured_plane(np.zeros((20, 3)))

    def test_real_icp_uses_measured_plane_residual(self):
        source = broad_ground()[::2]
        target = source + [0., 0., .12]
        arguments = (source, target, Rotation.identity(), np.zeros(3), .45, 8)
        original, _ = native.point_to_plane_translation_icp(*arguments)
        measured, _ = native.point_to_plane_translation_icp(*arguments,
            plane_constraint={'normal': [0., 0., 1.], 'target_m': .25, 'independent_inlier_cells': 1000000})
        self.assertAlmostEqual(original[2], .12, places=10)
        self.assertGreater(measured[2], .249)
        with self.assertRaisesRegex(ValueError, 'measured plane ICP'):
            native.point_to_plane_translation_icp(*arguments,
                plane_constraint={'normal': [0., 0., 2.], 'target_m': .25, 'independent_inlier_cells': 10})

    def test_corrected_coarse_deskew_equals_all_dense_point_transformation(self):
        points = np.asarray([[1., .2, -.7], [-.4, 2., -1.3], [.3, -.7, .2]])
        stamps = np.asarray([1010000000, 1040000000, 1080000000], dtype=np.int64)
        end = 1100000000
        slerp = Slerp([1., 1.1], Rotation.from_euler('xyz', [[1., -2., 5.], [3., 4., 15.]], degrees=True))
        correction = Rotation.from_euler('xy', [4., -3.], degrees=True)
        translation, velocity = np.asarray([2., -1., .3]), np.asarray([.7, -.2, .4])
        deskewed, gyro_end = lio_baseline.deskew_to_scan_end(points, stamps, end, slerp)
        corrected_end = correction * gyro_end
        dt = (stamps - end) / 1e9
        deskewed += corrected_end.inv().apply(dt[:, None] * velocity)
        coarse_world = corrected_end.apply(deskewed) + translation
        dense_world = dense.world_points(points, correction * slerp(stamps / 1e9), stamps,
            end, translation, velocity)
        np.testing.assert_allclose(dense_world, coarse_world, atol=1e-14, rtol=0)

    def test_legacy_unconstrained_icp_is_equal_to_retained_original(self):
        backup = Path(__file__).resolve().parents[2] / '04_outputs/dual_four_cold_stability_v1_20261010/floor_repair_original_sources_a0014/lio_local_map.py'
        spec = importlib.util.spec_from_file_location('original_floor_repair_native', backup)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        rng = np.random.default_rng(182)
        points = rng.normal(size=(350, 3))
        arguments = (points, points + [.06, -.04, .08], Rotation.identity(), np.zeros(3), .45, 8)
        before, before_metrics = original.point_to_plane_translation_icp(*arguments)
        after, after_metrics = native.point_to_plane_translation_icp(*arguments)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before_metrics, after_metrics)

    def test_actual_native_constrained_end_clock_uses_full_scan_offsets(self):
        points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        offsets = np.asarray([0, 50000000, 100000000], dtype=np.int64)
        messages = [{'payload': (1000000000 + i * 100000000, points.copy(), offsets)} for i in range(2)]
        index = SimpleNamespace(path=Path('clock_fixture.mcap'), iter_messages=lambda topic: iter(messages))
        slerp = Slerp([0., 2.], Rotation.identity(2))
        with patch.object(native, 'load_imu', return_value=(np.asarray([0, 2000000000]), None, None)), \
                patch.object(native, 'build_orientation_slerp', return_value=(slerp, np.zeros(3))), \
                patch.object(native, 'decode_points', side_effect=lambda payload: payload), \
                patch.object(native, 'voxel_select', side_effect=lambda p, o, *args: (p[:2], o[:2])), \
                patch.object(native, 'point_to_plane_translation_icp', return_value=(np.zeros(3),
                    {'correspondences': 100, 'rmse_m': .01, 'plane_rmse_m': .01})):
            _, legacy = native.run(index, .15, 2500, 10000, 8, .45, 8, 0., None, None, True)
            tracker = floor.FloorMotionConstraint()
            _, constrained = native.run(index, .15, 2500, 10000, 8, .45, 8, 0., None, None, True,
                motion_constraint=tracker)
        np.testing.assert_array_equal(legacy['timestamps_ns'], [1050000000, 1150000000])
        np.testing.assert_array_equal(constrained['timestamps_ns'], [1100000000, 1200000000])
        self.assertFalse(np.any(constrained['accepted']))


class SavedMotionBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.raw = self.root / 'recording.mcap'
        self.raw.write_bytes(b'raw-fixture')
        self.coarse = self.root / 'raw_sensor_backend/coarse'
        self.coarse.mkdir(parents=True)
        self.stamps = np.asarray([1000000000, 2000000000, 3000000000, 4000000000], dtype=np.int64)
        self.correction = Rotation.from_euler('xy', [4., -3.], degrees=True)
        self.positions = np.asarray([[0., 0., 0.], [.1, .1, .02], [.2, .15, .03], [.3, .2, .04]])
        self.fields = {'timestamp_ns': self.stamps,
            'pose_xyz_xyzw': np.column_stack((self.positions, np.tile(self.correction.as_quat(), (4, 1)))),
            'accepted': np.ones(4, dtype=bool),
            'gyro_world_correction_xyzw': np.tile(self.correction.as_quat(), (4, 1)),
            'measured_floor_supported': np.ones(4, dtype=bool),
            'measured_floor_motion_accepted': np.ones(4, dtype=bool)}
        self.write_json('CONFIG.json', {'raw': str(self.raw), 'raw_bytes': self.raw.stat().st_size,
            'raw_mtime_ns': self.raw.stat().st_mtime_ns})
        self.write_json('SENSOR_BACKEND.json', {'backend': 'fresh_raw_gyro_translation_local_map', 'raw': str(self.raw)})
        self.write_json('raw_sensor_backend/coarse/COARSE_RECEIPT.json', {'raw_file': str(self.raw),
            'SfM_used': False, 'GT_used': False, 'gyro_bias_rad_s': [0., 0., 0.],
            'measured_floor_constraint_used': True, 'measured_floor_constraint_protocol': floor.PROTOCOL})
        self.receipt = {'protocol': floor.PROTOCOL, 'actual_scan_count': 4,
            'per_scan': [{'timestamp_ns': int(t), 'motion_accepted': True} for t in self.stamps]}
        self.points = np.asarray([[1., .2, -.7], [-.4, 2., -1.3], [.3, -.7, .2]])
        self.offsets = np.asarray([0, 25000000, 50000000], dtype=np.int64)
        self.messages = [{'payload': (int(t - 50000000), self.points, self.offsets)} for t in self.stamps]
        self.index = SimpleNamespace(message_count=lambda topic: 4, iter_messages=lambda topic: iter(self.messages))

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding='utf-8')

    def execute(self):
        np.savez(self.coarse / 'coarse_body_odometry.npz', **self.fields)
        np.savez(self.coarse / 'coarse_sensor_diagnostics.npz', rmse=np.full(4, .01), correspondences=np.full(4, 400))
        self.write_json('raw_sensor_backend/coarse/MEASURED_FLOOR_RECEIPT.json', self.receipt)
        slerp = Slerp([0., 5.], Rotation.identity(2))
        with patch.object(mcap_audit, 'McapIndex', return_value=self.index), \
                patch.object(lio_baseline, 'load_imu', return_value=(np.asarray([0, 5000000000]), None, None)), \
                patch.object(lio_baseline, 'build_orientation_slerp', return_value=(slerp, np.zeros(3))), \
                patch.object(lio_baseline, 'decode_points', side_effect=lambda payload: payload):
            dense.run(self.root)

    def test_actual_dense_entry_applies_saved_correction_to_every_raw_point(self):
        self.execute()
        result = json.loads((self.root / 'raw_sensor_backend/RESULT.json').read_text(encoding='utf-8'))
        self.assertEqual(result['eligible_scans'], 4)
        velocities = dense.prior_velocities(self.stamps, self.positions)
        for i, row in enumerate(result['scans']):
            with np.load(self.root / 'raw_sensor_backend' / row['path'], allow_pickle=False) as data:
                expected = self.correction.apply(self.points) + self.positions[i] \
                    + ((self.offsets - 50000000) / 1e9)[:, None] * velocities[i]
                self.assertEqual(data['points_world_lio_m'].dtype, np.float32)
                np.testing.assert_array_equal(data['points_world_lio_m'], expected.astype(np.float32))

    def test_incomplete_fields_and_wrong_acceptance_fail_before_scan_output(self):
        del self.fields['measured_floor_supported']
        with self.assertRaisesRegex(ValueError, 'fields are incomplete'):
            self.execute()
        self.assertFalse((self.root / 'raw_sensor_backend/scans').exists())

    def test_nonunit_saved_correction_is_rejected(self):
        self.fields['gyro_world_correction_xyzw'][1] *= 2.
        with self.assertRaisesRegex(ValueError, 'corrections or acceptance'):
            self.execute()

    def test_saved_motion_acceptance_mismatch_is_rejected(self):
        self.fields['measured_floor_motion_accepted'][1] = False
        with self.assertRaisesRegex(ValueError, 'corrections or acceptance'):
            self.execute()

    def test_accepted_scan_requires_actual_plane_support(self):
        self.fields['measured_floor_supported'][2] = False
        with self.assertRaisesRegex(ValueError, 'corrections or acceptance'):
            self.execute()

    def test_receipt_clock_mismatch_is_rejected(self):
        self.receipt['per_scan'][2]['timestamp_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'raw-motion clocks'):
            self.execute()

    def test_dense_orientation_must_equal_saved_coarse_orientation(self):
        self.fields['pose_xyz_xyzw'][0, 3:] = Rotation.identity().as_quat()
        with self.assertRaisesRegex(ValueError, 'different orientation frame'):
            self.execute()


if __name__ == '__main__':
    unittest.main()
