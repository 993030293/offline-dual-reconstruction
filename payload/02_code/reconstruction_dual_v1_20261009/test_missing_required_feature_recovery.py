"""Check missing-camera recovery, fixed anchors, and actual database handoff."""

from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest import mock

import pycolmap

import camera_point_rebuild
import camera_visual_recovery
import test_camera_required_feature_recovery as feature_fixture
from camera_required_feature_recovery import run as feature_run, verify_feature_binding
from camera_support_recovery import model_identity
from common import atomic_json, read_json


class MissingRequiredFeatureRecoveryTests(unittest.TestCase):
    def setUp(self):
        feature_fixture.NativeFeatureRecoveryTests.setUp(self)

    def missing_source(self):
        image = min(self.model.images.values(), key=lambda value: value.image_id)
        name = image.name
        self.model.deregister_frame(image.frame_id)
        self.model.write(self.source)
        return name

    def restore_feature_fixture(self, command, output, label, budget, *args, **kwargs):
        """Replace only image extraction/matching with existing synthetic pixel data."""
        destination = Path(command[command.index('--database_path')+1])
        tables = ('keypoints', 'descriptors') if label == 'features' else ('matches', 'two_view_geometries')
        with closing(sqlite3.connect('file:' + self.database.as_posix() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(destination)) as database:
                with database:
                    for table in tables:
                        rows = source.execute('SELECT * FROM '+table).fetchall()
                        if rows:
                            database.executemany('INSERT OR REPLACE INTO '+table+' VALUES ('
                                + ','.join('?' for _ in rows[0])+')', rows)
        return .01

    def test_actual_native_sdk_registers_a_missing_required_camera_with_fixed_anchors(self):
        name = self.missing_source()
        original = pycolmap.Reconstruction(self.source)
        identity = model_identity(self.source)
        with mock.patch('camera_required_feature_recovery.run_native', side_effect=self.restore_feature_fixture):
            result = feature_run(self.source, self.names, self.database, self.images,
                self.root / 'native_missing', 120, {name: 0}, target_scope='missing_required')
        actual = pycolmap.Reconstruction(result['model'])
        self.assertEqual({image.name for image in actual.images.values() if image.has_pose}, self.names)
        restored = next(image for image in actual.images.values() if image.name == name)
        self.assertGreaterEqual(restored.num_points3D, 30)
        self.assertLessEqual(camera_visual_recovery.fixed_camera_invariants(original, actual), 1e-9)
        self.assertEqual(verify_feature_binding(actual, result['database']),
            sum(len(image.points2D) for image in actual.images.values() if image.has_pose))
        self.assertEqual(result['initially_missing_required_names'], [name])
        self.assertEqual(result['target_scope'], 'missing_required')
        self.assertFalse(result['training_admission_granted'])
        self.assertEqual(model_identity(self.source), identity)
        decisions = read_json(self.root / 'native_missing/DECISIONS.json')['rows']
        self.assertEqual(decisions[0]['native_registration_method'], 'native_PnP')
        self.assertEqual(decisions[0]['native_registration_attempts'],
                         [{'method': 'native_PnP', 'native_registered': True}])

    def test_default_scope_still_rejects_unregistered_required_cameras(self):
        name = self.missing_source()
        with mock.patch('camera_required_feature_recovery.run_native') as command:
            with self.assertRaisesRegex(ValueError, 'actual same-run weak required camera support'):
                feature_run(self.source, self.names, self.database, self.images,
                    self.root / 'default_rejects', 120, {name: 0})
        command.assert_not_called()

    def test_missing_scope_cannot_refine_an_already_registered_camera(self):
        name = self.missing_source()
        other = min(self.names-{name})
        with mock.patch('camera_required_feature_recovery.run_native') as command:
            with self.assertRaisesRegex(ValueError, 'actual same-run weak required camera support'):
                feature_run(self.source, self.names, self.database, self.images,
                    self.root / 'fabricated_target', 120, {other: 0}, target_scope='missing_required')
        command.assert_not_called()

    def visual_inputs(self):
        initial = self.root / 'initial'
        initial.mkdir()
        atomic_json(initial / 'INITIAL_FRAME_MANIFEST.json', {
            'selected_frames': [{'image_name': name} for name in sorted(self.names)]})
        (initial / 'frame_decisions.jsonl').write_text(''.join(
            __import__('json').dumps({'image_name': name, 'coarse_pose_valid': True,
                'excluded_role': None, 'decision': 'selected_initial_sfm_pending'})+'\n'
            for name in sorted(self.names)), encoding='utf-8')
        bridge = self.root / 'bridge'
        bridge.mkdir()
        original = pycolmap.Reconstruction(self.source)
        image = min(original.images.values(), key=lambda value: value.image_id)
        missing = image.name
        original.deregister_frame(image.frame_id)
        trusted = bridge / 'trusted_source'
        trusted.mkdir()
        original.write(trusted)
        identity = model_identity(self.source)
        atomic_json(bridge / 'CONTRACT.json', {'source_identity': identity, 'additions': []})
        atomic_json(bridge / 'RESULT.json', {
            'status': 'raw_quality_bridges_matched_geometry_pending', 'source_identity': identity,
            'database': str(self.database), 'images': str(self.images)})
        return initial, bridge, missing

    def incomplete_mapper(self, command, output, label, budget):
        seed = Path(command[command.index('--input_path')+1])
        destination = Path(command[command.index('--output_path')+1])
        pycolmap.Reconstruction(seed).write(destination)
        return .1

    def test_visual_stage_routes_real_missing_names_and_returns_new_database(self):
        initial, bridge, missing = self.visual_inputs()
        identity = model_identity(self.source)
        replaced = self.root / 'replaced_database.db'

        def recovery(source, required, database, images, output, budget, targets, *, target_scope):
            output.mkdir(parents=True, exist_ok=False)
            models = output / 'models'
            models.mkdir()
            self.model.write(models)
            with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as incoming:
                with closing(sqlite3.connect(replaced)) as destination:
                    incoming.backup(destination)
            self.assertEqual(targets, {missing: 0})
            self.assertEqual(target_scope, 'missing_required')
            self.assertGreater(budget, 60)
            self.assertLess(budget, 600)
            self.assertAlmostEqual(budget, 599.9)
            self.assertNotEqual(source, self.source)
            return {'model': str(models), 'database': str(replaced),
                'target_scope': target_scope, 'training_admission_granted': False}

        output = self.root / 'visual_routing'
        ticks = [1000.]

        def timed_mapper(*args):
            # Consume the mock command's declared duration independently of OS clock resolution.
            elapsed = self.incomplete_mapper(*args)
            ticks[0] += elapsed
            return elapsed

        clock = SimpleNamespace(monotonic=lambda: ticks[0])
        with mock.patch.object(camera_visual_recovery, 'time', clock):
            with mock.patch.object(camera_visual_recovery, 'run_native', side_effect=timed_mapper):
                with mock.patch('camera_required_feature_recovery.run', side_effect=recovery) as recover:
                    result = camera_visual_recovery.run(self.source, initial, self.database,
                        self.images, output, bridge)
        recover.assert_called_once()
        self.assertEqual(Path(result['database']), replaced)
        self.assertEqual(result['registered_frames'], len(self.names))
        self.assertEqual(read_json(output / 'MAPPING_READBACK.json')['missing_required_names'], [missing])
        self.assertFalse(result['training_admission_granted'])
        self.assertTrue(result['full_native_point_support_pending'])
        self.assertEqual(model_identity(self.source), identity)
        self.assertEqual(Path(result['model']).parent, output)
        rebuilt = camera_point_rebuild.run(result['model'], initial, result['database'],
            self.images, output / 'RESULT.json', self.root / 'actual_point_handoff')
        actual = pycolmap.Reconstruction(rebuilt['model'])
        self.assertEqual({image.name for image in actual.images.values() if image.has_pose}, self.names)
        self.assertTrue(rebuilt['full_required_native_point_support_verified'])
        self.assertGreaterEqual(rebuilt['minimum_required_support_after_rebuilding'], 30)
        self.assertEqual(verify_feature_binding(actual, rebuilt['database']),
            sum(len(image.points2D) for image in actual.images.values() if image.has_pose))
        self.assertFalse(rebuilt['training_admission_granted'])

    def test_changed_fixed_calibration_is_rejected_before_missing_feature_recovery(self):
        initial, bridge, _ = self.visual_inputs()

        def bad_mapper(command, output, label, budget):
            seed = Path(command[command.index('--input_path')+1])
            destination = Path(command[command.index('--output_path')+1])
            model = pycolmap.Reconstruction(seed)
            camera = next(iter(model.cameras.values()))
            params = camera.params.copy()
            params[0] += 1.
            camera.params = params
            model.write(destination)
            return .1

        with mock.patch.object(camera_visual_recovery, 'run_native', side_effect=bad_mapper):
            with mock.patch('camera_required_feature_recovery.run') as recover:
                with self.assertRaisesRegex(ValueError, 'fixed optical calibration'):
                    camera_visual_recovery.run(self.source, initial, self.database,
                        self.images, self.root / 'bad_calibration', bridge)
        recover.assert_not_called()


if __name__ == '__main__':
    unittest.main()
