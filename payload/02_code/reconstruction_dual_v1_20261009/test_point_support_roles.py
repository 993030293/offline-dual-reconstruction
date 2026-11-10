"""Exercise real sparse filtering without dropping required cameras or lowering policies."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pycolmap

from camera_initialization_roles import point_support_roles
from camera_support_recovery import model_identity
from common import atomic_json, read_json
from prepare_metric_support import prepare


class PointSupportRoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
        options.num_points3D = 10001
        model = pycolmap.synthesize_dataset(options)
        self.names = sorted(image.name for image in model.images.values())
        self.optional = self.names[-1]
        by_name = {image.name: image for image in model.images.values()}
        point_key = next(iter(model.points3D))
        keep = {by_name[self.names[0]].image_id, by_name[self.optional].image_id}
        first_track = [(int(element.image_id), int(element.point2D_idx))
            for element in model.points3D[point_key].track.elements]
        self.assertEqual(len(first_track), 4)
        for image_id, point_index in first_track:
            if image_id not in keep:
                model.delete_observation(image_id, point_index)
        self.assertEqual(model.points3D[point_key].track.length(), 2)
        for element in list(by_name[self.optional].points2D):
            if element.has_point3D() and element.point3D_id != point_key:
                index = next(entry.point2D_idx for entry in model.points3D[element.point3D_id].track.elements
                    if entry.image_id == by_name[self.optional].image_id)
                model.delete_observation(by_name[self.optional].image_id, index)
        self.assertEqual(by_name[self.optional].num_points3D, 1)
        self.source = self.root / 'coarse_initialization/supported_models_BA_REQUIRED'
        self.source.mkdir(parents=True)
        model.write(self.source)
        self.identity = model_identity(self.source)
        self.initial = self.root / 'initial'
        self.initial.mkdir()
        self.required = set(self.names[:-1])
        atomic_json(self.initial / 'INITIAL_FRAME_MANIFEST.json', {'selected_frames':
            [{'image_name': name} for name in sorted(self.required)]})
        rows = [{'image_name': image.name, 'coarse_pose_valid': True, 'excluded_role': False,
            'decision': 'selected_initial_sfm_pending' if image.name in self.required else 'eligible_not_selected',
            'C_W_m': image.projection_center().tolist()} for image in model.images.values()]
        (self.initial / 'frame_decisions.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.role_file = self.source.parent / 'RESULT.json'
        self.seeded = {'status': 'coarse_initialized_visual_support_ready_BA_REQUIRED', 'model': str(self.source),
            'output_identity': self.identity, 'required_initial_names_all_registered': True,
            'initialization_roles': {'original_registered_names': sorted(self.required),
                'required_names': sorted(self.required), 'optional_bridge_names': [self.optional]}}
        atomic_json(self.role_file, self.seeded)
        self.poses = {image.name: image.cam_from_world().matrix().copy() for image in model.images.values()}

    def test_zero_filtered_optional_witness_abstains_with_real_binary_readback(self):
        result = prepare(self.source, self.initial, self.root / 'prepared', 'full-training-policy', self.role_file)
        actual = pycolmap.Reconstruction(result['model'])
        names = {image.name for image in actual.images.values() if image.has_pose}
        self.assertEqual(names, self.required)
        self.assertEqual(actual.num_points3D(), 10000)
        self.assertEqual(result['minimum_track'], 3)
        self.assertEqual(result['rejected_optional_bridge_names'], [self.optional])
        self.assertTrue(result['all_required_cameras_and_poses_retained'])
        self.assertFalse(result['all_original_cameras_and_poses_retained'])
        self.assertTrue(all(point.track.length() >= 3 for point in actual.points3D.values()))
        for image in actual.images.values():
            self.assertTrue(np.array_equal(image.cam_from_world().matrix(), self.poses[image.name]))
        self.assertEqual(model_identity(self.source), self.identity)

    def test_default_strict_scope_does_not_silently_drop_a_registered_camera(self):
        with self.assertRaisesRegex(ValueError, 'required cameras without observations'):
            prepare(self.source, self.initial, self.root / 'strict', 'full-training-policy')
        decisions = read_json(self.root / 'strict/POINT_SUPPORT_DECISIONS.json')
        self.assertEqual(decisions['unsupported_required_names'], [self.optional])
        self.assertFalse((self.root / 'strict/RESULT.json').exists())
        self.assertEqual(model_identity(self.source), self.identity)

    def test_initial_camera_without_three_view_points_still_fails(self):
        atomic_json(self.initial / 'INITIAL_FRAME_MANIFEST.json', {'selected_frames':
            [{'image_name': name} for name in self.names]})
        value = {**self.seeded, 'initialization_roles': {**self.seeded['initialization_roles'],
            'required_names': self.names, 'optional_bridge_names': []}}
        atomic_json(self.role_file, value)
        with self.assertRaisesRegex(ValueError, 'required cameras without observations'):
            prepare(self.source, self.initial, self.root / 'required_failure', 'full-training-policy', self.role_file)
        self.assertFalse((self.root / 'required_failure/RESULT.json').exists())
        self.assertEqual(model_identity(self.source), self.identity)

    def test_roles_cannot_relabel_an_initial_or_original_camera_as_optional(self):
        value = {**self.seeded, 'initialization_roles': {**self.seeded['initialization_roles'],
            'required_names': self.names[1:-1], 'optional_bridge_names': [self.names[0], self.optional]}}
        with self.assertRaisesRegex(ValueError, 'required initial or original camera denominator'):
            point_support_roles(self.names, self.required, value)

    def test_point_roles_reject_changed_actual_model_identity(self):
        atomic_json(self.role_file, {**self.seeded, 'output_identity': {'images.bin': 'changed'}})
        with self.assertRaisesRegex(ValueError, 'typed support receipt changed'):
            prepare(self.source, self.initial, self.root / 'changed', 'full-training-policy', self.role_file)
        self.assertFalse((self.root / 'changed').exists())


if __name__ == '__main__':
    unittest.main()
