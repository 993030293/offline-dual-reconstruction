"""Check numeric equivalence and reject cross-attempt coarse-pose provenance."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from run_initial_safe import load_native, pose_provenance
from coarse_handoff import run as generate_handoff


class InitialProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        raw = self.root / 'input.mcap'
        raw.write_bytes(b'fixture')
        backend = self.root / 'raw_sensor_backend'
        (backend / 'coarse').mkdir(parents=True)
        trajectory = backend / 'coarse/coarse_body_odometry.npz'
        trajectory.write_bytes(b'fixture')
        identity = {'raw': str(raw), 'raw_bytes': raw.stat().st_size,
            'raw_mtime_ns': raw.stat().st_mtime_ns}
        (self.root / 'SENSOR_BACKEND.json').write_text(json.dumps({**identity,
            'backend': 'fresh_raw_gyro_translation_local_map'}), encoding='utf-8')
        (backend / 'RESULT.json').write_text(json.dumps({**identity,
            'status': 'fresh_raw_registered_scans_complete', 'run_root': str(self.root)}), encoding='utf-8')
        self.args = SimpleNamespace(raw=raw, coarse_trajectory=trajectory, output=self.root / 'initial')
        poses = [[0., 0., 0., 0., 0., 0., 1.], [1., .5, -.2, 0., 0., .2, np.sqrt(.96)]]
        self.curve = self.native.baseline.CoarseCameraCurve([100, 200], poses, np.eye(4))

    def test_fresh_serial_and_batch_keep_numeric_values_and_rejections(self):
        serial = self.curve.at(150)
        batch = self.native.coarse_camera_at_many(self.curve, [50, 100, 150, 200, 250])
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with pose_provenance(self.native, self.args):
                updated_serial = self.curve.at(150)
                updated_batch = self.native.coarse_camera_at_many(self.curve, [50, 100, 150, 200, 250])
        for before, after in [(serial, updated_serial), *zip(batch, updated_batch)]:
            if 'C_W_m' not in before:
                self.assertEqual(before, after)
                continue
            self.assertNotEqual(before['source_role'], after['source_role'])
            self.assertNotEqual(before['pose_producer'], after['pose_producer'])
            self.assertEqual(after['coarse_source_run_root'], str(self.root))
            for key in set(before) - {'source_role', 'pose_producer'}:
                self.assertEqual(before[key], after[key], key)

    def test_recorded_source_remains_unconfirmed(self):
        self.args.coarse_trajectory = None
        before = self.native.coarse_camera_at_many(self.curve, [150])
        with patch.dict(os.environ, {}, clear=True):
            with pose_provenance(self.native, self.args):
                self.assertEqual(before, self.native.coarse_camera_at_many(self.curve, [150]))

    def test_generated_source_requires_both_binding_and_trajectory(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'same-attempt'):
                with pose_provenance(self.native, self.args):
                    self.fail('An unbound trajectory must fail before initial selection')
        self.args.coarse_trajectory = None
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(ValueError, 'same-attempt'):
                with pose_provenance(self.native, self.args):
                    self.fail('A bound source must have its actual trajectory')

    def test_other_attempt_trajectory_is_rejected(self):
        other = self.root / 'other.npz'
        other.write_bytes(self.args.coarse_trajectory.read_bytes())
        self.args.coarse_trajectory = other
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(ValueError, 'another attempt'):
                with pose_provenance(self.native, self.args):
                    self.fail('Equal fixture bytes do not imply same-attempt lineage')

    def test_changed_raw_identity_is_rejected(self):
        self.args.raw.write_bytes(b'changed')
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(ValueError, 'frozen original'):
                with pose_provenance(self.native, self.args):
                    self.fail('A changed original input must not retain its source label')

    def test_output_outside_attempt_is_rejected(self):
        self.args.output = self.root.parent / 'other_attempt_initial'
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaises(ValueError):
                with pose_provenance(self.native, self.args):
                    self.fail('Generated trajectory use must remain within its attempt')

    def test_callbacks_restored_after_failure(self):
        original_at = self.native.baseline.CoarseCameraCurve.at
        original_many = self.native.coarse_camera_at_many
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with pose_provenance(self.native, self.args):
                    raise RuntimeError('fixture failure')
        self.assertIs(self.native.baseline.CoarseCameraCurve.at, original_at)
        self.assertIs(self.native.coarse_camera_at_many, original_many)

    def create_handoff(self):
        source = self.args.coarse_trajectory.parent
        for name in ('COARSE_RECEIPT.json', 'CONTRACT.json', 'INITIAL_IMU_DIAGNOSTIC.json'):
            (source / name).write_text('{}', encoding='utf-8')
        destination = self.root / 'camera_front/coarse'
        generate_handoff(self.root, self.args.raw, destination)
        self.args.coarse_trajectory = destination / 'coarse_body_odometry.npz'
        return destination

    def test_actual_same_attempt_copy_handoff_is_accepted_without_numeric_change(self):
        self.create_handoff()
        original = self.curve.at(150)
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with pose_provenance(self.native, self.args):
                actual = self.curve.at(150)
        self.assertEqual(actual['C_W_m'], original['C_W_m'])
        self.assertEqual(actual['R_WC'], original['R_WC'])
        self.assertEqual(actual['coarse_source_run_root'], str(self.root))

    def test_changed_handoff_payload_is_rejected_even_with_valid_receipt(self):
        self.create_handoff()
        self.args.coarse_trajectory.write_bytes(b'changed copy')
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(ValueError, 'payload differs'):
                with pose_provenance(self.native, self.args):
                    self.fail('A copied receipt cannot certify changed numeric inputs')

    def test_handoff_cannot_claim_another_attempt_root(self):
        destination = self.create_handoff()
        path = destination / 'SAME_ATTEMPT_HANDOFF.json'
        value = json.loads(path.read_text())
        value['run_root'] = str(self.root.parent)
        path.write_text(json.dumps(value), encoding='utf-8')
        with patch.dict(os.environ, {'RECON_FRESH_RAW_SENSOR_RUN_ROOT': str(self.root)}):
            with self.assertRaisesRegex(ValueError, 'same-attempt sensor lineage'):
                with pose_provenance(self.native, self.args):
                    self.fail('A cross-attempt marker must not certify a local copy')


if __name__ == '__main__':
    unittest.main()
