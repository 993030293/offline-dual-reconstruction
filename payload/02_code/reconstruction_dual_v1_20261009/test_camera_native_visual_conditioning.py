"""Check real native recovery, point-only rebuilding, and variable joint pose blocks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pycolmap

from camera_initialization_roles import NATIVE_POINT_REBUILD_INPUT_STATUS, point_support_roles
from camera_joint_conditioning import (NATIVE_JOINT_BA_SOFT_SECONDS,
    NATIVE_JOINT_STAGE_HARD_SECONDS, joint_problem, load_helpers, run as joint_run)
from camera_point_rebuild import run as rebuild_run
import camera_point_rebuild
from camera_required_relocalization import low_support_targets, run as relocalization_run
from camera_support_recovery import model_identity
from camera_visual_recovery import (reestimate_unbound_anchors, required_native_support,
    required_registered_support, run as visual_run)
from common import atomic_json, read_json


class NativeVisualConditioningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / 'database.db'
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
        options.num_points3D = 120
        with pycolmap.Database.open(self.database) as database:
            self.model = pycolmap.synthesize_dataset(options, database)
        self.names = {image.name for image in self.model.images.values() if image.has_pose}
        self.initial = self.root / 'initial'
        self.initial.mkdir()
        atomic_json(self.initial / 'INITIAL_FRAME_MANIFEST.json', {
            'selected_frames': [{'image_name': name} for name in sorted(self.names)]})
        rows = [{'image_name': image.name, 'coarse_pose_valid': True, 'excluded_role': False,
            'decision': 'selected_initial_sfm_pending', 'C_W_m': image.projection_center().tolist()}
            for image in self.model.images.values()]
        (self.initial / 'frame_decisions.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
        self.images = self.root / 'images'
        self.images.mkdir()
        for name in self.names:
            (self.images / name).write_bytes(b'synthetic_native_database_fixture')
        self.source = self.root / 'source/models'
        self.source.mkdir(parents=True)

    def source_receipt(self, model=None):
        (model or self.model).write(self.source)
        value = {'status': 'native_visual_registered_support_ready_BA_REQUIRED', 'model': str(self.source),
            'output_identity': model_identity(self.source), 'required_initial_names_all_registered': True,
            'initialization_roles': {'original_registered_names': sorted(self.names),
                'required_names': sorted(self.names), 'optional_bridge_names': []}}
        path = self.source.parent / 'RESULT.json'
        atomic_json(path, value)
        return path

    def test_joint_hard_watchdog_preserves_bounded_post_solver_window(self):
        self.assertEqual(NATIVE_JOINT_STAGE_HARD_SECONDS, 4 * NATIVE_JOINT_BA_SOFT_SECONDS)
        self.assertGreater(NATIVE_JOINT_STAGE_HARD_SECONDS - NATIVE_JOINT_BA_SOFT_SECONDS,
            NATIVE_JOINT_BA_SOFT_SECONDS)

    def test_actual_native_missing_camera_registration_and_fixed_pose_point_rebuild(self):
        missing = max(self.model.images.values(), key=lambda image: image.image_id)
        self.model.deregister_frame(missing.frame_id)
        self.model.write(self.source)
        identity = model_identity(self.source)
        recovered_path = self.root / 'recovered'
        recovered = visual_run(self.source, self.initial, self.database, self.images, recovered_path)
        self.assertEqual(recovered['registered_frames'], len(self.names))
        self.assertLessEqual(recovered['trusted_pose_maximum_component_delta'], 1e-9)
        self.assertFalse(recovered['training_admission_granted'])
        self.assertEqual(recovered['status'], NATIVE_POINT_REBUILD_INPUT_STATUS)
        self.assertTrue(recovered['full_native_point_support_pending'])
        rebuilt = rebuild_run(recovered['model'], self.initial, recovered['database'], self.images,
            recovered_path / 'RESULT.json', self.root / 'rebuilt')
        self.assertEqual(rebuilt['registered_frames'], len(self.names))
        self.assertLessEqual(rebuilt['fixed_pose_maximum_component_delta'], 1e-9)
        self.assertTrue(rebuilt['explicit_joint_camera_BA_still_required'])
        self.assertTrue(rebuilt['full_required_native_point_support_verified'])
        self.assertEqual(model_identity(self.source), identity)
        renamed = self.database.with_name('released_database.db')
        self.database.rename(renamed)
        renamed.rename(self.database)

    def test_actual_joint_optimizer_has_free_pose_blocks_and_all_visual_factors(self):
        role_path = self.source_receipt()
        model = pycolmap.Reconstruction(self.source)
        adjuster, variables, residuals = joint_problem(model, load_helpers(), iterations=5, solver_seconds=10)
        self.assertEqual(len(variables), len(self.names))
        self.assertEqual(residuals, 2*model.compute_num_observations())
        summary = adjuster.solve()
        self.assertTrue(summary.is_solution_usable())
        self.assertEqual(summary.ceres_summary.num_residuals, residuals)
        self.assertEqual(read_json(role_path)['output_identity'], model_identity(self.source))

    def test_joint_conditioning_preserves_actual_names_intrinsics_and_unchanged_source(self):
        role_path = self.source_receipt()
        identity = model_identity(self.source)
        result = joint_run(self.source, self.initial, role_path, self.root / 'joint')
        actual = pycolmap.Reconstruction(result['model'])
        self.assertEqual({image.name for image in actual.images.values() if image.has_pose}, self.names)
        self.assertEqual(result['actual_variable_camera_blocks'], len(self.names))
        self.assertTrue(result['full_metric_evaluation_still_required'])
        self.assertFalse(result['training_admission_granted'])
        self.assertTrue(np.isfinite(result['actual_pose_maximum_component_delta']))
        contract = read_json(self.root / 'joint/CONTRACT.json')
        self.assertEqual(contract['native_joint_BA_iterations'], 60)
        self.assertEqual(contract['native_joint_BA_soft_seconds'], NATIVE_JOINT_BA_SOFT_SECONDS)
        self.assertEqual(contract['native_joint_stage_hard_seconds'], NATIVE_JOINT_STAGE_HARD_SECONDS)
        for key, camera in self.model.cameras.items():
            self.assertTrue(np.array_equal(camera.params, actual.cameras[key].params))
        self.assertEqual(model_identity(self.source), identity)

    def test_changed_actual_role_identity_fails_before_joint_solver(self):
        role_path = self.source_receipt()
        value = read_json(role_path)
        value['output_identity'] = {'images.bin': 'changed'}
        atomic_json(role_path, value)
        with self.assertRaisesRegex(ValueError, 'typed role receipt changed'):
            joint_run(self.source, self.initial, role_path, self.root / 'rejected_joint')
        self.assertFalse((self.root / 'rejected_joint/NATIVE_SOLVER.json').exists())
        self.assertTrue((self.root / 'rejected_joint/FAILURE.json').is_file())

    def test_pose_gap_anchors_are_reregistered_without_inventing_prior_or_dropping_required_role(self):
        gap = max(self.model.images.values(), key=lambda image: image.image_id).name
        names = reestimate_unbound_anchors(self.model, self.names-{gap})
        self.assertEqual(names, [gap])
        self.assertEqual(self.model.num_reg_images(), len(self.names)-1)
        self.assertEqual(set(read_json(self.initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames'][i]['image_name']
            for i in range(len(self.names))), self.names)

    def test_native_support_roles_cannot_turn_required_camera_into_optional_witness(self):
        role_path = self.source_receipt()
        value = read_json(role_path)
        required = sorted(self.names)
        value['initialization_roles']['required_names'] = required[1:]
        value['initialization_roles']['optional_bridge_names'] = required[:1]
        with self.assertRaisesRegex(ValueError, 'required initial or original camera denominator'):
            point_support_roles(self.names, self.names, value)


    def weaken_actual_image(self, target_count):
        image = max(self.model.images.values(), key=lambda value: value.num_points3D)
        indices = [index for index, point in enumerate(image.points2D) if point.has_point3D()]
        for index in indices[target_count:]:
            self.model.delete_observation(image.image_id, index)
        self.assertEqual(self.model.images[image.image_id].num_points3D, target_count)
        return image.name

    def pending_source_receipt(self):
        path = self.source_receipt()
        value = read_json(path)
        value.update(status=NATIVE_POINT_REBUILD_INPUT_STATUS, full_native_point_support_pending=True,
            training_admission_granted=False)
        atomic_json(path, value)
        return path

    def test_filtered_registration_boundary_is_provisional_and_full_minimum_remains_strict(self):
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        name = self.weaken_actual_image(minimum-1)
        observations = required_registered_support(self.model, self.names)
        self.assertEqual(observations[name], minimum-1)
        with self.assertRaisesRegex(ValueError, 'below_native_minimum'):
            required_native_support(self.model, self.names)
        receipt = read_json(self.pending_source_receipt())
        with self.assertRaisesRegex(ValueError, 'requires actual all-point rebuilding'):
            point_support_roles(self.names, self.names, receipt)
        roles = point_support_roles(self.names, self.names, receipt, allow_pending_point_rebuild=True)
        self.assertEqual(set(roles['required_names']), self.names)

    def test_actual_all_point_rebuild_restores_filtered_support_before_unchanged_full_gate(self):
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        name = self.weaken_actual_image(minimum-1)
        path = self.pending_source_receipt()
        identity = model_identity(self.source)
        result = rebuild_run(self.source, self.initial, self.database, self.images, path,
            self.root / 'boundary_rebuild')
        actual = pycolmap.Reconstruction(result['model'])
        observations = required_native_support(actual, self.names)
        self.assertGreaterEqual(observations[name], minimum)
        self.assertEqual(result['required_native_minimum_unchanged'], minimum)
        self.assertTrue(result['full_required_native_point_support_verified'])
        self.assertFalse(result['training_admission_granted'])
        self.assertEqual(model_identity(self.source), identity)

    def test_missing_or_zero_support_cannot_be_provisional_registration(self):
        self.weaken_actual_image(0)
        with self.assertRaisesRegex(ValueError, 'zero_observations'):
            required_registered_support(self.model, self.names)
        path = self.pending_source_receipt()
        with self.assertRaisesRegex(ValueError, 'zero_observations'):
            rebuild_run(self.source, self.initial, self.database, self.images, path,
                self.root / 'zero_support_rejected')
        self.assertFalse((self.root / 'zero_support_rejected/triangulation.log').exists())
        with self.assertRaisesRegex(ValueError, 'missing'):
            required_registered_support(self.model, self.names | {'missing_camera_fixture.jpg'})

    def test_pending_registered_model_cannot_bypass_rebuilding_into_joint_BA(self):
        role_path = self.pending_source_receipt()
        with self.assertRaisesRegex(ValueError, 'requires actual all-point rebuilding'):
            joint_run(self.source, self.initial, role_path, self.root / 'pending_joint_rejected')
        self.assertFalse((self.root / 'pending_joint_rejected/NATIVE_SOLVER.json').exists())
        receipt = read_json(role_path)
        receipt['training_admission_granted'] = True
        with self.assertRaisesRegex(ValueError, 'requires actual all-point rebuilding'):
            point_support_roles(self.names, self.names, receipt, allow_pending_point_rebuild=True)

    def test_actual_stale_fixed_pose_is_relocalized_before_strict_rebuilt_support(self):
        image = max(self.model.images.values(), key=lambda value: value.image_id)
        name = image.name
        frame = self.model.frames[image.frame_id]
        before = pycolmap.Reconstruction(self.model)
        pose = frame.rig_from_world
        frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, pose.translation + np.array([.5, 0, 0]))
        path = self.pending_source_receipt()
        identity = model_identity(self.source)
        result = rebuild_run(self.source, self.initial, self.database, self.images, path,
            self.root / 'stale_pose_rebuild')
        self.assertTrue(result['required_low_support_relocalization_completed'])
        relocalization = result['required_support_relocalization']
        self.assertIn(name, relocalization['targets_before'])
        self.assertLess(relocalization['targets_before'][name], result['required_native_minimum_unchanged'])
        self.assertLessEqual(relocalization['remaining_anchor_maximum_component_delta'], 1e-9)
        self.assertGreater(relocalization['actual_pose_maximum_component_delta'], .1)
        self.assertGreater(result['overall_source_to_output_pose_maximum_component_delta'], .1)
        self.assertEqual(result['fixed_pose_check_scope'], 'first_and_last_point_triangulation_inputs')
        actual = pycolmap.Reconstruction(result['model'])
        support = required_native_support(actual, self.names)
        self.assertGreaterEqual(support[name], result['required_native_minimum_unchanged'])
        self.assertFalse(result['training_admission_granted'])
        self.assertEqual(set(support), self.names)
        self.assertTrue(result['full_required_native_point_support_verified'])
        self.assertLess(np.linalg.norm(actual.images[image.image_id].projection_center()
            - before.images[image.image_id].projection_center()), .1)
        self.assertEqual(model_identity(self.source), identity)
        self.assertFalse(low_support_targets(actual, self.names))

    def test_relocalization_targets_follow_required_roles_without_promoting_optional_witness(self):
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        name = self.weaken_actual_image(minimum-1)
        self.assertEqual(low_support_targets(self.model, self.names), {name: minimum-1})
        self.assertEqual(low_support_targets(self.model, self.names-{name}), {})

    def test_relocalization_cannot_hide_a_missing_required_camera(self):
        with self.assertRaisesRegex(ValueError, 'registered camera denominator'):
            low_support_targets(self.model, self.names | {'missing_required_fixture.jpg'})

    def test_relocalization_rejects_absent_fixed_visual_anchors_before_native_execution(self):
        for image in list(self.model.images.values()):
            indices = [index for index, point in enumerate(image.points2D) if point.has_point3D()]
            for index in indices[1:]:
                self.model.delete_observation(image.image_id, index)
        self.model.write(self.source)
        identity = model_identity(self.source)
        rejected = self.root / 'no_fixed_anchors'
        with self.assertRaisesRegex(ValueError, 'sufficient actual fixed visual anchors'):
            relocalization_run(self.source, self.names, self.database, self.images, rejected, 60)
        self.assertTrue((rejected / 'FAILURE.json').is_file())
        self.assertFalse((rejected / 'mapping.log').exists())
        self.assertFalse((rejected / 'RESULT.json').exists())
        self.assertEqual(model_identity(self.source), identity)

    def test_relocalization_cannot_exceed_exhausted_parent_budget(self):
        self.model.write(self.source)
        rejected = self.root / 'no_budget'
        with self.assertRaisesRegex(TimeoutError, 'relocalization budget remains'):
            relocalization_run(self.source, self.names, self.database, self.images, rejected, 10)
        self.assertFalse(rejected.exists())

    def test_first_point_interface_cannot_move_a_fixed_anchor_before_relocalization(self):
        role_path = self.pending_source_receipt()
        identity = model_identity(self.source)
        original_run = camera_point_rebuild.run_native

        def unexpected_pose_change(command, output, label, seconds):
            elapsed = original_run(command, output, label, seconds)
            models = Path(command[command.index('--output')+1])
            actual = pycolmap.Reconstruction(models)
            image = min(actual.images.values(), key=lambda value: value.image_id)
            frame = actual.frames[image.frame_id]
            pose = frame.rig_from_world
            frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, pose.translation + np.array([.01, 0, 0]))
            actual.write(models)
            return elapsed

        rejected = self.root / 'unexpected_fixed_pose_change'
        with mock.patch.object(camera_point_rebuild, 'run_native', side_effect=unexpected_pose_change):
            with self.assertRaisesRegex(ValueError, 'moved fixed trusted camera anchors'):
                rebuild_run(self.source, self.initial, self.database, self.images, role_path, rejected)
        self.assertTrue((rejected / 'FAILURE.json').is_file())
        self.assertFalse((rejected / 'RESULT.json').exists())
        self.assertFalse((rejected / 'required_support_relocalization').exists())
        self.assertEqual(model_identity(self.source), identity)


if __name__ == '__main__':
    unittest.main()
