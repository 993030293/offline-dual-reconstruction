"""Check real native camera and observation state across a partial PnP failure."""

import unittest

import numpy as np
import pycolmap

from camera_pnp_recovery_input import preserve_verified_updates


def fixture():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
    options.num_points3D = 50
    original = pycolmap.synthesize_dataset(options)
    partial = pycolmap.Reconstruction(original)
    images = sorted(original.images.values(), key=lambda image: image.image_id)
    accepted, failed = images[:2]
    observations = [(index, point.point3D_id) for index, point in enumerate(accepted.points2D)
                    if point.has_point3D()]
    for image in (accepted, failed):
        partial.deregister_frame(image.frame_id)
    partial.frames[accepted.frame_id].set_cam_from_world(accepted.camera_id, accepted.cam_from_world())
    partial.register_frame(accepted.frame_id)
    for index, key in observations:
        partial.add_observation(key, pycolmap.TrackElement(accepted.image_id, index))
    changed = original.frames[accepted.frame_id].rig_from_world
    changed.translation = changed.translation + np.array([1000., 0., 0.])
    original.frames[accepted.frame_id].rig_from_world = changed
    rows = [{'image_name': accepted.name, 'status': 'genuine_PnP_accepted'},
            {'image_name': failed.name, 'status': 'abstain_PnP_geometry_or_coarse_conflict'}]
    return original, partial, accepted, failed, rows


class PartialPnpRecoveryInputTests(unittest.TestCase):
    def test_real_verified_pose_and_correspondences_survive_other_camera_failure(self):
        original, partial, accepted, failed, rows = fixture()
        pose = partial.images[accepted.image_id].cam_from_world().matrix().copy()
        observations = partial.images[accepted.image_id].num_points3D
        trusted = {image.image_id: image.cam_from_world().matrix().copy()
                   for image in partial.images.values() if image.has_pose and image.name != accepted.name}
        points = {key: point.xyz.copy() for key, point in partial.points3D.items()}
        value = preserve_verified_updates(original, partial, rows)
        np.testing.assert_array_equal(pose, partial.images[accepted.image_id].cam_from_world().matrix())
        self.assertFalse(np.array_equal(pose, original.images[accepted.image_id].cam_from_world().matrix()))
        self.assertEqual(partial.images[accepted.image_id].num_points3D, observations)
        self.assertEqual(partial.num_reg_images(), original.num_reg_images())
        self.assertGreater(partial.images[failed.image_id].num_points3D, 0)
        for key, matrix in trusted.items():
            np.testing.assert_array_equal(partial.images[key].cam_from_world().matrix(), matrix)
        for key, xyz in points.items():
            np.testing.assert_array_equal(partial.points3D[key].xyz, xyz)
        self.assertEqual(value['verified_PnP_names'], [accepted.name])
        self.assertEqual(value['unverified_registered_names'], [failed.name])
        self.assertFalse(value['training_admission_granted'])
        self.assertFalse(value['runtime_language_model_used'])

    def test_a_false_success_without_actual_native_pose_is_rejected(self):
        original, partial, accepted, _, rows = fixture()
        partial.deregister_frame(accepted.frame_id)
        with self.assertRaisesRegex(ValueError, 'actual pose or native visual support'):
            preserve_verified_updates(original, partial, rows)

    def test_duplicate_or_foreign_decision_is_rejected(self):
        for invalid in ('duplicate', 'foreign'):
            original, partial, _, _, rows = fixture()
            if invalid == 'duplicate':
                rows.append(dict(rows[0]))
            else:
                rows[1]['image_name'] = 'absent'
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                preserve_verified_updates(original, partial, rows)

    def test_complete_failure_cannot_masquerade_as_verified_partial_recovery(self):
        original, partial, _, _, rows = fixture()
        rows[0]['status'] = 'abstain_no_genuine_PnP'
        with self.assertRaisesRegex(ValueError, 'genuine successes'):
            preserve_verified_updates(original, partial, rows)


if __name__ == '__main__':
    unittest.main()
