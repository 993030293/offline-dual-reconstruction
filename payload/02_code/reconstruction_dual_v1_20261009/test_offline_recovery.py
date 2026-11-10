"""Regression checks for unattended recovery and the raw sensor adapter."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_camera_backend import prepare, require_automatic_geometry
from register_raw_lidar import prior_velocities, world_points
from sensor_backend import select_backend, REGISTERED, ODOMETRY, RAW_LIDAR, RAW_IMU


class OfflineRecoveryTests(unittest.TestCase):
    def test_developer_adoption_rejected_before_input_or_cache_reads(self):
        for kwargs in ({'adopted': 'missing'}, {'verified_metric': 'missing'}):
            with self.assertRaisesRegex(ValueError, 'manually adopted'):
                prepare('missing', 'missing', 'missing', 'missing', **kwargs)

    def test_developer_cache_cannot_masquerade_as_automatic(self):
        with self.assertRaisesRegex(ValueError, 'developer-adopted'):
            require_automatic_geometry({'method': 'same_run_adopted_geometry', 'stages': []})
        with self.assertRaisesRegex(ValueError, 'developer-adopted'):
            require_automatic_geometry({'method': 'automatic', 'stages': [{'assistant_intervention': True}]})
        require_automatic_geometry({'method': 'genuine_fixed_map_PnP', 'stages': []})

    def test_actual_topics_determine_the_sensor_backend(self):
        self.assertEqual(select_backend({REGISTERED: 3, ODOMETRY: 3}), 'recorded_registered_lidar_and_odometry')
        self.assertEqual(select_backend({RAW_LIDAR: 3, RAW_IMU: 100}), 'fresh_raw_gyro_translation_local_map')
        with self.assertRaisesRegex(ValueError, 'Partially recorded'):
            select_backend({REGISTERED: 3, RAW_LIDAR: 3, RAW_IMU: 100})

    def test_previous_velocity_does_not_use_the_current_or_future_step(self):
        times = np.arange(4, dtype=np.int64) * 1_000_000_000
        positions = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0], [100, 0, 0]], dtype=float)
        np.testing.assert_array_equal(prior_velocities(times, positions),
            [[0, 0, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0]])

    def test_dense_world_deskew_matches_native_frame_algebra(self):
        points = np.array([[1., 2, 3], [-2, 1, .5]])
        rotations = Rotation.from_rotvec([[0., 0, .1], [0., 0, .2]])
        end_rotation = Rotation.from_rotvec([0., 0, .3])
        stamps, end = np.array([900_000_000, 950_000_000], np.int64), 1_000_000_000
        translation, velocity = np.array([1., 2, 3]), np.array([4., -1, .5])
        local = (end_rotation.inv() * rotations).apply(points)
        local += end_rotation.inv().apply((stamps - end)[:, None] / 1e9 * velocity)
        expected = end_rotation.apply(local) + translation
        np.testing.assert_allclose(world_points(points, rotations, stamps, end, translation, velocity),
            expected, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
