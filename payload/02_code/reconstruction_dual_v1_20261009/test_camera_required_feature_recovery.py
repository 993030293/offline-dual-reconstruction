"""Check native track preservation, replaced arrays, and bounded recovery routing."""

from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pycolmap

import camera_point_rebuild
import camera_required_relocalization
from camera_required_feature_recovery import run as feature_run, verify_feature_binding
from camera_support_recovery import model_identity
from camera_visual_recovery import fixed_camera_invariants, required_native_support
from common import atomic_json, read_json
from native_command import run_native


class NativeFeatureRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / 'database.db'
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 4
        options.num_points3D = 120
        with pycolmap.Database.open(self.database) as db:
            self.model = pycolmap.synthesize_dataset(options, db)
        # Actual feature extractors persist float32 coordinates in COLMAP databases.
        for image in self.model.images.values():
            for point in image.points2D:
                point.xy = point.xy.astype(np.float32).astype(np.float64)
        self.names = {image.name for image in self.model.images.values() if image.has_pose}
        self.source = self.root / 'source'
        self.source.mkdir()
        self.images = self.root / 'images'
        self.images.mkdir()
        for name in self.names:
            (self.images / name).write_bytes(b'native_database_geometry_fixture')
        self.model.write(self.source)

    def weaken(self):
        image = min(self.model.images.values(), key=lambda value: value.image_id)
        indices = [index for index, point in enumerate(image.points2D) if point.has_point3D()]
        for index in indices[29:]:
            self.model.delete_observation(image.image_id, index)
        self.model.write(self.source)
        return image.name

    def test_actual_native_incremental_worker_keeps_fixed_geometry_and_strict_support(self):
        identity = model_identity(self.source)
        destination = self.root / 'incremental_models'
        seconds = run_native([sys.executable, str(Path(camera_point_rebuild.__file__)),
            '--triangulate-only', '--preserve-native-tracks', '--source-model', self.source,
            '--database', self.database, '--images', self.images, '--output', destination],
            self.root, 'incremental', 60)
        actual = pycolmap.Reconstruction(destination)
        observations = required_native_support(actual, self.names)
        self.assertGreater(seconds, 0)
        self.assertEqual(set(observations), self.names)
        self.assertGreaterEqual(min(observations.values()), 30)
        self.assertLessEqual(fixed_camera_invariants(self.model, actual), 1e-9)
        self.assertEqual(verify_feature_binding(actual, self.database), sum(len(image.points2D) for image in actual.images.values()))
        self.assertEqual(model_identity(self.source), identity)

    def test_actual_changed_keypoint_coordinate_is_rejected_before_triangulation(self):
        model = pycolmap.Reconstruction(self.source)
        image = min(model.images.values(), key=lambda value: value.image_id)
        image.points2D[0].xy = image.points2D[0].xy + np.array([1., 0.])
        with self.assertRaisesRegex(ValueError, 'coordinates or indices differ'):
            verify_feature_binding(model, self.database)

    def test_old_in_memory_keypoint_arrays_cannot_match_replaced_database_features(self):
        image = min(self.model.images.values(), key=lambda value: value.image_id)
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image.image_id,)).fetchone()
            values = np.frombuffer(row[2], dtype=np.float32).reshape(row[0], row[1])
            replacement = np.concatenate([values, values[-1:]], axis=0)
            with db:
                db.execute('UPDATE keypoints SET rows=?,data=? WHERE image_id=?',
                    (len(replacement), replacement.tobytes(), image.image_id))
        self.model.deregister_frame(image.frame_id)
        self.assertEqual(len(self.model.images[image.image_id].points2D), row[0])
        with self.assertRaisesRegex(ValueError, 'coordinates or indices differ'):
            restored = pycolmap.Reconstruction(self.source)
            verify_feature_binding(restored, self.database)

    def test_feature_recovery_does_not_reset_an_exhausted_parent_budget(self):
        output = self.root / 'no_budget'
        with self.assertRaisesRegex(TimeoutError, 'original final triangulation reserve'):
            feature_run(self.source, self.names, self.database, self.images, output, 60, {})
        self.assertFalse(output.exists())

    def test_nonweak_or_fabricated_target_is_rejected_without_native_GPU_command(self):
        name = min(self.names)
        output = self.root / 'wrong_targets'
        with mock.patch('camera_required_feature_recovery.run_native') as command:
            with self.assertRaisesRegex(ValueError, 'actual same-run weak required camera support'):
                feature_run(self.source, self.names, self.database, self.images, output, 120, {name: 1})
        command.assert_not_called()
        self.assertTrue((output / 'FAILURE.json').is_file())
        self.assertFalse((output / 'RESULT.json').exists())

    def test_successful_native_command_with_missing_camera_routes_using_original_same_run_source(self):
        name = self.weaken()
        identity = model_identity(self.source)

        def incomplete_mapper(command, output, label, budget):
            seed = Path(command[command.index('--input_path')+1])
            destination = Path(command[command.index('--output_path')+1])
            pycolmap.Reconstruction(seed).write(destination)
            return .1

        output = self.root / 'incomplete'
        payload = {'status': 'native_required_feature_relocalized_incremental_triangulation_REQUIRED',
            'model': str(self.source), 'database': str(self.database), 'training_admission_granted': False}
        # Mocked work must advance time independently of Windows clock quantization.
        clock = mock.Mock()
        clock.monotonic.side_effect = [10., 10.02, 10.12]
        with mock.patch.object(camera_required_relocalization, 'time', clock):
            with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=incomplete_mapper):
                with mock.patch.object(camera_required_relocalization, 'recover_required_features', return_value=dict(payload)) as recovery:
                    result = camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        arguments = recovery.call_args.args
        self.assertEqual(arguments[0], self.source.resolve())
        self.assertEqual(arguments[2], self.database.resolve())
        self.assertEqual(arguments[6], {name: 29})
        self.assertGreater(arguments[5], 60)
        self.assertLess(arguments[5], 180)
        self.assertEqual(result['database'], str(self.database))
        self.assertFalse(result['training_admission_granted'])
        self.assertEqual(read_json(output / 'MAPPING_READBACK.json')['missing_original_names'], [name])
        self.assertEqual(model_identity(self.source), identity)

    def test_rebuild_result_returns_replaced_database_and_workers_receive_that_binding(self):
        from camera_initialization_roles import NATIVE_POINT_REBUILD_INPUT_STATUS
        initial = self.root / 'initial'
        initial.mkdir()
        atomic_json(initial / 'INITIAL_FRAME_MANIFEST.json', {
            'selected_frames': [{'image_name': name} for name in sorted(self.names)]})
        role_path = self.source.parent / 'RESULT.json'
        atomic_json(role_path, {'status': NATIVE_POINT_REBUILD_INPUT_STATUS,
            'model': str(self.source), 'output_identity': model_identity(self.source),
            'required_initial_names_all_registered': True, 'full_native_point_support_pending': True,
            'training_admission_granted': False, 'initialization_roles': {
                'original_registered_names': sorted(self.names), 'required_names': sorted(self.names),
                'optional_bridge_names': []}})
        original_run, commands = camera_point_rebuild.run_native, []
        replaced_database = self.root / 'actual_refined_database.db'

        def expose_weak_support(command, output, label, budget):
            commands.append(list(map(str, command)))
            seconds = original_run(command, output, label, budget)
            if label == 'triangulation':
                destination = Path(command[command.index('--output')+1])
                actual = pycolmap.Reconstruction(destination)
                image = min(actual.images.values(), key=lambda value: value.image_id)
                indices = [index for index, point in enumerate(image.points2D) if point.has_point3D()]
                for index in indices[29:]:
                    actual.delete_observation(image.image_id, index)
                actual.write(destination)
            return seconds

        def replacement_result(source, required, database, images, output, budget):
            output.mkdir(parents=True, exist_ok=False)
            models = output / 'models'
            models.mkdir()
            self.model.write(models)
            with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as incoming:
                with closing(sqlite3.connect(replaced_database)) as destination:
                    incoming.backup(destination)
            return {'model': str(models), 'database': str(replaced_database),
                'point_rebuilding_mode': 'native_incremental_preserve_geometry_filtered_tracks'}

        with mock.patch.object(camera_point_rebuild, 'run_native', side_effect=expose_weak_support):
            with mock.patch.object(camera_point_rebuild, 'relocalize_required', side_effect=replacement_result):
                result = camera_point_rebuild.run(self.source, initial, self.database, self.images,
                    role_path, self.root / 'rebuild')
        self.assertEqual(Path(result['database']), replaced_database)
        retry = commands[-1]
        self.assertIn('--preserve-native-tracks', retry)
        self.assertEqual(Path(retry[retry.index('--database')+1]), replaced_database)
        self.assertGreaterEqual(result['minimum_required_support_after_rebuilding'], 30)
        self.assertFalse(result['training_admission_granted'])

    def complete_mapper(self, model):
        def mapper(command, output, label, budget):
            destination = Path(command[command.index('--output_path')+1])
            model.write(destination)
            return .1
        return mapper

    def test_registered_but_newly_weak_camera_uses_current_geometry_features_and_remaining_budget(self):
        complete = pycolmap.Reconstruction(self.model)
        original_target = self.weaken()
        other = next(image for image in complete.images.values() if image.name != original_target)
        indices = [index for index, point in enumerate(other.points2D) if point.has_point3D()]
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        for index in indices[minimum-2:]:
            complete.delete_observation(other.image_id, index)
        identity = model_identity(self.source)
        output = self.root / 'registered_but_weak'
        payload = {'status': 'native_required_feature_relocalized_incremental_triangulation_REQUIRED',
            'model': str(self.source), 'database': str(self.database),
            'point_rebuilding_mode': 'native_incremental_preserve_geometry_filtered_tracks',
            'training_admission_granted': False}
        clock = mock.Mock()
        clock.monotonic.side_effect = [10., 10.02, 10.12]
        with mock.patch.object(camera_required_relocalization, 'time', clock):
            with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
                with mock.patch.object(camera_required_relocalization, 'recover_required_features', return_value=dict(payload)) as recovery:
                    result = camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        arguments = recovery.call_args.args
        self.assertEqual(arguments[0], output / 'models')
        self.assertEqual(arguments[2], output / 'database.db')
        self.assertEqual(arguments[6], {other.name: minimum-2})
        self.assertNotIn(original_target, arguments[6])
        self.assertGreater(arguments[5], 60)
        self.assertLess(arguments[5], 180)
        readback = read_json(output / 'MAPPING_READBACK.json')
        self.assertEqual(readback['missing_original_names'], [])
        self.assertEqual(readback['below_native_minimum'], {other.name: minimum-2})
        self.assertEqual(readback['original_native_minimum'], minimum)
        self.assertFalse(readback['training_admission_granted'])
        self.assertFalse(result['training_admission_granted'])
        self.assertEqual(result['point_rebuilding_mode'], 'native_incremental_preserve_geometry_filtered_tracks')
        self.assertEqual(model_identity(self.source), identity)

    def test_registered_camera_with_zero_support_routes_before_nonzero_provisional_gate(self):
        name = self.weaken()
        complete = pycolmap.Reconstruction(self.source)
        image = next(image for image in complete.images.values() if image.name == name)
        for index, point in enumerate(list(image.points2D)):
            if point.has_point3D():
                complete.delete_observation(image.image_id, index)
        output = self.root / 'registered_zero'
        with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
            with mock.patch.object(camera_required_relocalization, 'recover_required_features', return_value={
                    'model': str(self.source), 'database': str(self.database),
                    'training_admission_granted': False}) as recovery:
                result = camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        self.assertEqual(recovery.call_args.args[6], {name: 0})
        self.assertFalse(result['training_admission_granted'])

    def test_registered_camera_at_native_minimum_does_not_request_feature_refinement(self):
        complete = pycolmap.Reconstruction(self.model)
        name = self.weaken()
        image = next(image for image in complete.images.values() if image.name == name)
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        indices = [index for index, point in enumerate(image.points2D) if point.has_point3D()]
        for index in indices[minimum:]:
            complete.delete_observation(image.image_id, index)
        output = self.root / 'registered_native_minimum'
        with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
            with mock.patch.object(camera_required_relocalization, 'recover_required_features') as recovery:
                result = camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        recovery.assert_not_called()
        self.assertEqual(result['target_support_before_mandatory_rebuild'][name], minimum)
        self.assertEqual(result['point_rebuilding_mode'], 'native_incremental_preserve_geometry_filtered_tracks')
        self.assertGreater(result['actual_database_keypoint_bindings_checked'], 0)
        self.assertTrue(result['full_required_point_support_pending'])
        self.assertFalse(result['training_admission_granted'])

    def test_ordinary_relocalization_removes_bad_geometry_before_preserving_native_tracks(self):
        complete = pycolmap.Reconstruction(self.model)
        self.weaken()
        bad_point_id, bad_point = next(iter(complete.points3D.items()))
        bad_point.xyz = bad_point.xyz + np.array([1000., 1000., 1000.])
        identity = model_identity(self.source)
        output = self.root / 'ordinary_geometry_filtered'
        with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
            with mock.patch.object(camera_required_relocalization, 'recover_required_features') as recovery:
                result = camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        recovery.assert_not_called()
        self.assertGreater(result['native_filtered_observations'], 0)
        self.assertEqual(result['point_rebuilding_mode'], 'native_incremental_preserve_geometry_filtered_tracks')
        actual = pycolmap.Reconstruction(result['model'])
        required_native_support(actual, self.names)
        self.assertNotIn(bad_point_id, actual.points3D)
        self.assertEqual(model_identity(self.source), identity)
        self.assertFalse(result['training_admission_granted'])

    def test_ordinary_relocalization_rejects_changed_feature_coordinates_before_preservation(self):
        complete = pycolmap.Reconstruction(self.model)
        self.weaken()
        image = next(iter(complete.images.values()))
        image.points2D[0].xy = image.points2D[0].xy + np.array([1., 0.])
        identity = model_identity(self.source)
        output = self.root / 'ordinary_feature_mismatch'
        with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
            with self.assertRaisesRegex(ValueError, 'coordinates or indices differ'):
                camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        self.assertTrue((output / 'FAILURE.json').is_file())
        self.assertFalse((output / 'RESULT.json').exists())
        self.assertEqual(model_identity(self.source), identity)

    def test_registered_weak_refinement_failure_is_preserved_and_never_admitted(self):
        self.weaken()
        complete = pycolmap.Reconstruction(self.source)
        identity = model_identity(self.source)
        output = self.root / 'registered_weak_failed'
        with mock.patch.object(camera_required_relocalization, 'run_native', side_effect=self.complete_mapper(complete)):
            with mock.patch.object(camera_required_relocalization, 'recover_required_features',
                    side_effect=ValueError('Strict native support remains incomplete')):
                with self.assertRaisesRegex(ValueError, 'Strict native support remains incomplete'):
                    camera_required_relocalization.run(self.source, self.names, self.database, self.images, output, 180)
        self.assertTrue((output / 'MAPPING_READBACK.json').is_file())
        self.assertFalse((output / 'RESULT.json').exists())
        failure = read_json(output / 'FAILURE.json')
        self.assertTrue(failure['source_model_unchanged'])
        self.assertIn('Strict native support remains incomplete', failure['traceback'])
        self.assertEqual(model_identity(self.source), identity)


if __name__ == '__main__':
    unittest.main()
