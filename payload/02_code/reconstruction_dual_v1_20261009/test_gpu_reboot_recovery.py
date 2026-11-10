"""Reject unsafe GPU-owner reclamation and preserve interrupted native receipts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from common import atomic_json, read_json
import gpu_lease as leases


class RebootOwnerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.shared = self.root / 'shared'
        self.shared.mkdir()
        self.lease = self.root / 'features_GPU_LEASE.json'
        self.owner_path = self.shared / 'OWNER.json'
        self.owner = {'status': 'acquired', 'label': 'features', 'pid': 900001, 'process_birth': 50.,
            'source': {'path': str(Path(leases.__file__).resolve())}, 'receipt': str(self.lease),
            'started_unix': 60., 'acquired_unix': 61.}
        for binding in (patch.object(leases, 'SHARED', self.shared), patch.object(leases.psutil, 'boot_time', return_value=100.)):
            binding.start()
            self.addCleanup(binding.stop)

    def fixture(self, legacy=False, command=None):
        command = command or [str(leases.NATIVE_COLMAP), 'feature_extractor']
        lease = dict(self.owner)
        if legacy:
            atomic_json(self.root / 'STATE.json', {'controller_pid': 900001, 'stages': [
                {'name': 'features', 'status': 'running', 'commands': [command],
                    'owned_births': [{'pid': 900002, 'birth': 55.}]}]})
        else:
            lease.update(command=command, child_pid=900002, child_birth=55.)
        atomic_json(self.lease, lease)
        atomic_json(self.owner_path, self.owner)
        return self.lease.read_bytes()

    def test_direct_native_owner_after_boot_is_reclaimed_under_actual_byte_lock(self):
        payload = self.fixture()
        current = self.root / 'new_GPU_LEASE.json'
        with leases.gpu_lease(current, 'new', maximum_wait_seconds=1) as value:
            recovery = read_json(value['offline_reboot_owner_recovery_receipt'])
            self.assertTrue(recovery['exclusive_byte_lock_acquired'])
            self.assertTrue(recovery['native_COLMAP_only'])
            self.assertFalse(recovery['WSL_process_termination_inferred'])
            self.assertFalse(recovery['runtime_language_model_used'])
            self.assertFalse(recovery['manual_intervention_used'])
            self.assertEqual(recovery['original_owner'], self.owner)
            self.assertEqual(Path(recovery['new_lease_receipt']), current)
        self.assertEqual(self.lease.read_bytes(), payload)
        self.assertEqual(read_json(self.owner_path)['status'], 'released')
        self.assertEqual(len(list(self.shared.glob('HOST_REBOOT_NATIVE_OWNER_RECOVERY_*.json'))), 1)

    def test_legacy_sfm_journal_supplies_actual_native_child_ownership(self):
        payload = self.fixture(legacy=True)
        evidence = leases.reboot_native_owner_evidence(self.owner)
        self.assertEqual(len(evidence['recorded_process_identities']), 2)
        self.assertEqual(Path(evidence['native_command_evidence']), self.root / 'STATE.json')
        self.assertEqual(self.lease.read_bytes(), payload)

    def test_same_boot_dead_owner_is_not_silently_reclaimed(self):
        self.owner['process_birth'] = 110.
        self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'prior GPU owner'):
            with leases.gpu_lease(self.root / 'new_GPU_LEASE.json', 'new', maximum_wait_seconds=1):
                self.fail('An unverified same-boot owner must not admit execution')
        self.assertEqual(read_json(self.owner_path), self.owner)

    def test_live_recorded_identity_rejects_recovery_even_with_an_old_timestamp(self):
        self.fixture()
        fake = type('Process', (), {'create_time': lambda self: 50.})()
        with patch.object(leases.psutil, 'Process', return_value=fake):
            self.assertIsNone(leases.reboot_native_owner_evidence(self.owner))

    def test_wsl_owner_is_never_inferred_ended_from_windows_identity(self):
        self.fixture(command=['wsl.exe', '-d', 'unverified', '--exec', 'python'])
        self.assertIsNone(leases.reboot_native_owner_evidence(self.owner))

    def test_missing_native_child_evidence_remains_quarantined(self):
        atomic_json(self.lease, self.owner)
        atomic_json(self.owner_path, self.owner)
        self.assertIsNone(leases.reboot_native_owner_evidence(self.owner))

    def test_changed_original_receipt_is_explicitly_rejected_without_writes(self):
        self.fixture()
        changed = read_json(self.lease)
        changed['pid'] = 900003
        atomic_json(self.lease, changed)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            leases.reboot_native_owner_evidence(self.owner)
        self.assertEqual(read_json(self.owner_path), self.owner)
        self.assertFalse(list(self.shared.glob('HOST_REBOOT_NATIVE_OWNER_RECOVERY_*.json')))

    def test_post_boot_child_or_acquisition_timestamp_does_not_recover(self):
        for field in ('child_birth', 'acquired_unix'):
            with self.subTest(field=field):
                self.fixture()
                changed = read_json(self.lease)
                changed[field] = 110.
                atomic_json(self.lease, changed)
                owner = dict(self.owner)
                if field == 'acquired_unix':
                    owner[field] = 110.
                self.assertIsNone(leases.reboot_native_owner_evidence(owner))

    def test_explicit_failed_drain_is_not_converted_to_a_reboot_success(self):
        self.fixture()
        owner = dict(self.owner, status='quarantined_owned_command_not_drained')
        self.assertIsNone(leases.reboot_native_owner_evidence(owner))

    def test_unknown_owner_source_is_not_recovered(self):
        self.fixture()
        owner = dict(self.owner, source={'path': str(self.root / 'unknown.py')})
        self.assertIsNone(leases.reboot_native_owner_evidence(owner))


if __name__ == '__main__':
    unittest.main()
