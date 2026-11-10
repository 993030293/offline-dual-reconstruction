"""Check indexed native pair votes and exact-input rejection retry behavior."""

import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap

from camera_support_pnp import PAIR_BASE, correspondences, estimate_camera


class IndexedPairTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(':memory:')
        self.addCleanup(self.connection.close)
        self.connection.execute('CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, data BLOB)')
        self.trusted = {5: np.array([101, 102, 103]), 9: np.array([101, 102, 104]),
                        14: np.array([101, 999, 104]), 20: np.array([101])}
        for first, second, pairs in ((5, 11, [[0, 2], [1, 3], [2, 4]]),
                (9, 11, [[0, 2], [1, 3], [2, 5]]),
                (11, 14, [[2, 0], [3, 1], [5, 2]]), (11, 20, [[8, 0]])):
            values = np.asarray(pairs, np.uint32)
            self.connection.execute('INSERT INTO two_view_geometries VALUES(?,?,?)',
                (first*PAIR_BASE + second, len(values), values.tobytes()))
        self.connection.execute('INSERT INTO two_view_geometries VALUES(?,?,?)', (11*PAIR_BASE+16, 1, b''))
        self.expected = [(2, 101, 3), (5, 104, 2), (3, 102, 2), (4, 103, 1)]

    def test_real_left_right_pair_votes_ambiguities_and_unique_points(self):
        self.assertEqual(correspondences(self.connection, 11, self.trusted), self.expected)
        self.assertEqual(correspondences(self.connection, 11, {}), [])

    def test_sqlite_variable_limit_does_not_drop_trusted_neighbors(self):
        self.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 2)
        self.assertEqual(correspondences(self.connection, 11, self.trusted), self.expected)

    def test_actual_query_uses_primary_key_lookup_instead_of_full_geometry_scan(self):
        statements = []
        self.connection.set_trace_callback(statements.append)
        correspondences(self.connection, 11, self.trusted)
        query = next(value for value in statements if value.startswith('SELECT'))
        self.connection.set_trace_callback(None)
        plan = self.connection.execute('EXPLAIN QUERY PLAN ' + query).fetchall()
        self.assertTrue(any('SEARCH' in row[3] and 'PRIMARY KEY' in row[3] for row in plan), plan)
        self.assertFalse(any('SCAN ' in row[3] for row in plan), plan)


class FixedInputRetryTests(unittest.TestCase):
    def setUp(self):
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
        options.num_points3D = 50
        self.model = pycolmap.synthesize_dataset(options)
        for image in self.model.images.values():
            image.name = f'frame_{image.image_id:05d}_{1_000_000_000 + image.image_id*40_000_000}.jpg'
        self.image = next(iter(self.model.images.values()))
        self.choices = [(index, int(point.point3D_id), 1) for index, point in enumerate(self.image.points2D)
                        if point.has_point3D()]
        self.trusted = {image.image_id: np.array([], np.int64) for image in self.model.images.values()
                        if image.image_id != self.image.image_id}
        self.lookup = {self.image.name: {'C_W_m': self.image.projection_center().copy()}}
        self.mapper, self.alignment = pycolmap.IncrementalMapperOptions(), pycolmap.Sim3d()
        self.cache = {}

    def invoke(self):
        return estimate_camera(None, self.image, self.model, self.trusted, self.mapper,
            self.lookup, self.alignment, rejection_cache=self.cache)

    def test_identical_rejected_numeric_inputs_do_not_repeat_native_estimation(self):
        with patch('camera_support_pnp.correspondences', return_value=self.choices), \
                patch('camera_support_pnp.pycolmap.estimate_and_refine_absolute_pose', return_value=None) as native:
            first = self.invoke()
            second = self.invoke()
        self.assertEqual(native.call_count, 2)
        self.assertIsNone(first[2])
        self.assertIsNone(second[2])
        self.assertEqual(second[0]['status'], 'abstain_no_genuine_PnP')
        self.assertTrue(all(row['unchanged_effective_PnP_input_cached'] for row in second[1]))
        self.assertTrue(all(not row['native_pose_estimation_executed'] for row in second[1]))

    def test_changed_actual_point_calibration_or_prior_invalidates_rejection(self):
        for changed in ('point', 'calibration', 'prior'):
            with self.subTest(changed=changed):
                self.cache.clear()
                with patch('camera_support_pnp.correspondences', return_value=self.choices), \
                        patch('camera_support_pnp.pycolmap.estimate_and_refine_absolute_pose', return_value=None) as native:
                    self.invoke()
                    if changed == 'point':
                        point = self.model.points3D[self.choices[0][1]]
                        point.xyz = point.xyz + np.array([1e-6, 0., 0.])
                    elif changed == 'calibration':
                        camera = self.model.cameras[self.image.camera_id]
                        parameters = camera.params.copy()
                        parameters[0] += .1
                        camera.params = parameters
                    else:
                        self.lookup[self.image.name]['C_W_m'] += np.array([1e-6, 0., 0.])
                    self.invoke()
                self.assertEqual(native.call_count, 4)

    def test_actual_native_success_is_reestimated_and_never_cached_as_registration(self):
        native_function = pycolmap.estimate_and_refine_absolute_pose
        with patch('camera_support_pnp.correspondences', return_value=self.choices), \
                patch('camera_support_pnp.pycolmap.estimate_and_refine_absolute_pose', wraps=native_function) as native:
            first = self.invoke()
            second = self.invoke()
        self.assertEqual(first[0]['status'], 'genuine_PnP_accepted')
        self.assertEqual(second[0]['status'], 'genuine_PnP_accepted')
        self.assertIsNotNone(first[2])
        self.assertIsNotNone(second[2])
        self.assertEqual(native.call_count, 2)
        self.assertEqual(self.cache, {})


if __name__ == '__main__':
    unittest.main()
