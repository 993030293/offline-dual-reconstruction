"""Exercise genuine PnP for an unregistered initial camera with a real feature database."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pycolmap

from camera_initialization_roles import visual_recovery_targets, VISUAL_RECOVERY_PROTOCOL
from camera_support_recovery import model_identity, run as match_run
from camera_support_pnp import run as pnp_run
from common import atomic_json


class MissingInitialRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / 'database.db'
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 8
        options.num_points3D = 100
        with pycolmap.Database.open(self.database) as db:
            model = pycolmap.synthesize_dataset(options, db)
        self.images = []
        for image in sorted(model.images.values(), key=lambda image: image.image_id):
            image.name = f'frame_{image.image_id:05d}_{1_000_000_000 + image.image_id * 40_000_000}.jpg'
            self.images.append({'image_name': image.name, 'image_id': image.image_id,
                'coarse_pose_valid': True, 'excluded_role': False, 'decision': 'selected_initial_sfm_pending',
                'C_W_m': image.projection_center().tolist()})
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executemany('UPDATE images SET name=? WHERE image_id=?',
                [(row['image_name'], row['image_id']) for row in self.images])
            connection.commit()
        missing = max(model.images.values(), key=lambda image: image.image_id)
        self.missing_name = missing.name
        self.missing_pose = missing.cam_from_world().matrix().copy()
        model.deregister_frame(missing.frame_id)
        self.source = self.root / 'models'
        self.source.mkdir()
        model.write(self.source)
        self.identity = model_identity(self.source)
        retained = pycolmap.Reconstruction(self.source)
        self.original_poses = {image.name: image.cam_from_world().matrix().copy()
            for image in retained.images.values() if image.has_pose}
        self.initial = self.root / 'initial'
        self.initial.mkdir()
        atomic_json(self.initial / 'INITIAL_FRAME_MANIFEST.json',
            {'selected_frames': [{'image_name': row['image_name']} for row in self.images]})
        (self.initial / 'frame_decisions.jsonl').write_text('\n'.join(json.dumps(row) for row in self.images), encoding='utf-8')
        photos = self.root / 'images'
        photos.mkdir()
        for row in self.images:
            (photos / row['image_name']).write_bytes(b'not-decoded-by-this-numeric-regression')

    def test_missing_initial_is_a_target_without_any_weak_registered_pose(self):
        result = visual_recovery_targets({'observed': 100}, {'observed', 'unregistered'}, {'observed': 0}, 30, .6)
        self.assertEqual(result['registered_suspect_names'], [])
        self.assertEqual(result['target_names'], ['unregistered'])
        self.assertEqual(result['required_names'], ['observed', 'unregistered'])

    def test_matching_only_covers_missing_initial_and_uses_actual_database(self):
        result = match_run(self.source, self.initial, self.root / 'matched', matching_only=True)
        self.assertEqual(result['status'], 'verified_missing_matches_ready')
        self.assertEqual(result['all_recovery_target_names'], [self.missing_name])
        self.assertEqual(result['visual_recovery_protocol'], VISUAL_RECOVERY_PROTOCOL)
        with closing(sqlite3.connect(result['database'])) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM images').fetchone()[0], 8)
        self.assertEqual(model_identity(self.source), self.identity)

    def test_real_absolute_pose_recovers_unregistered_initial_with_trusted_geometry_unchanged(self):
        result = pnp_run(self.source, self.initial, self.database, self.root / 'pnp')
        actual = pycolmap.Reconstruction(result['model'])
        self.assertEqual(actual.num_reg_images(), 8)
        self.assertEqual(result['required_missing_initial_names_recovered'], [self.missing_name])
        self.assertTrue(result['required_initial_names_all_registered'])
        for image in actual.images.values():
            if image.name in self.original_poses:
                self.assertTrue(np.array_equal(image.cam_from_world().matrix(), self.original_poses[image.name]))
        image = next(image for image in actual.images.values() if image.name == self.missing_name)
        self.assertGreaterEqual(image.num_points3D, 30)
        self.assertTrue(np.allclose(image.cam_from_world().matrix(), self.missing_pose, atol=.01))
        self.assertEqual(model_identity(self.source), self.identity)

    def test_legacy_matching_scope_cache_is_rejected(self):
        output = self.root / 'legacy_match'
        output.mkdir()
        atomic_json(output / 'RESULT.json', {'status': 'verified_missing_matches_ready',
            'source_identity': self.identity})
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            match_run(self.source, self.initial, output, matching_only=True)


if __name__ == '__main__':
    unittest.main()
