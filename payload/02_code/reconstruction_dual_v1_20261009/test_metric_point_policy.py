"""Exercise physical seed counts against actual point and observation mutations."""

import unittest

import numpy as np
import pycolmap

from metric_point_policy import native_point_policy, point_policy_removals, qualified_seed_observations


class MetricPointPolicyTests(unittest.TestCase):
    def setUp(self):
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
        options.num_points3D = 12
        self.model = pycolmap.synthesize_dataset(options)

    def test_predictions_equal_actual_point_deletions_without_changing_seed(self):
        model = self.model
        first = next(iter(model.images.values()))
        point_id = next(iter(model.points3D))
        pose = first.cam_from_world()
        model.points3D[point_id].xyz = pose.rotation.matrix().T @ (np.array([0., 0., .005]) - pose.translation)
        short_id = next(key for key in model.points3D if key != point_id)
        for element in list(model.points3D[short_id].track.elements)[:2]:
            model.delete_observation(element.image_id, element.point2D_idx)
        before = {image.name: image.cam_from_world().matrix().copy() for image in model.images.values()}
        removals = point_policy_removals(model, 1., 3, .01)
        self.assertEqual(set(removals), {point_id, short_id})
        predicted = qualified_seed_observations(model, removals)
        self.assertEqual(model.num_points3D(), 12)
        copied = pycolmap.Reconstruction(model)
        for key in removals:
            copied.delete_point3D(key)
        self.assertEqual(predicted, {image.name: image.num_points3D for image in copied.images.values()})
        for image in model.images.values():
            self.assertTrue(np.array_equal(before[image.name], image.cam_from_world().matrix()))

    def test_invalid_active_depths_remain_a_failure(self):
        first = next(iter(self.model.images.values()))
        key = next(iter(self.model.points3D))
        pose = first.cam_from_world()
        self.model.points3D[key].xyz = pose.rotation.matrix().T @ (np.array([0., 0., -1.]) - pose.translation)
        with self.assertRaisesRegex(ValueError, 'invalid observed depths'):
            point_policy_removals(self.model, 1., 3, .01)

    def test_native_policy_reads_actual_installed_thresholds(self):
        _, policy, source, near = native_point_policy()
        self.assertEqual(policy['minimum_point_track_length'], 3)
        self.assertEqual(policy['minimum_retained_sparse_points'], 10000)
        self.assertEqual(near, .01)
        self.assertTrue(source.is_file())

    def test_nonfinite_or_nonpositive_scales_are_not_silent_fallbacks(self):
        for scale in (0., -1., float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                point_policy_removals(self.model, scale, 3, .01)


if __name__ == '__main__':
    unittest.main()
