"""Recover only exact ended native ownership recorded by a failed real supervisor."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from common import atomic_json, read_json
import gpu_lease as leases


class InterruptedOwnerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.shared = self.root / 'shared'
        self.shared.mkdir()
        self.native = self.root / 'side/camera_backend/raw_quality_bridge'
        self.native.mkdir(parents=True)
        self.lease = self.native / 'matching_GPU_LEASE.json'
        self.lifecycle = self.native / 'matching_COMMAND_LIFECYCLE.json'
        self.guard = self.root / 'side/metric_process.json'
        self.owner_path = self.shared / 'OWNER.json'
        boot = psutil.boot_time()
        self.owner = {'status': 'acquired', 'label': 'matching', 'pid': 900201,
            'process_birth': boot + 1, 'source': {'path': str(Path(leases.__file__).resolve())},
            'receipt': str(self.lease), 'started_unix': boot + 2, 'acquired_unix': boot + 3}
        self.child = {'pid': 900202, 'creation_time': boot + 2}
        self.command = [str(leases.NATIVE_COLMAP), 'matches_importer']
        binding = patch.object(leases, 'SHARED', self.shared)
        binding.start()
        self.addCleanup(binding.stop)

    def fixture(self):
        lease = dict(self.owner, command=self.command, child_pid=self.child['pid'],
            child_birth=self.child['creation_time'], command_drain_verified=False)
        atomic_json(self.lease, lease)
        atomic_json(self.owner_path, self.owner)
        atomic_json(self.lifecycle, {'status': 'running', 'command': self.command,
            'child_pid': self.child['pid'], 'owned_tree_drained': False,
            'owned_process_identities': [self.child]})
        source = leases.ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005/run_bounded_task.py'
        atomic_json(self.guard, {'status': 'failed_controller', 'returncode': 15,
            'error': 'PermissionError(13, access denied)', 'source': leases.data_identity(source),
            'owned_process_identities': [
                {'pid': self.owner['pid'], 'creation_time': self.owner['process_birth']}, self.child]})
        return self.lease.read_bytes(), self.guard.read_bytes()

    def test_actual_byte_lock_recovers_bound_interruption_without_changing_old_evidence(self):
        old_lease, old_guard = self.fixture()
        current = self.root / 'new_GPU_LEASE.json'
        with leases.gpu_lease(current, 'new', maximum_wait_seconds=1) as value:
            record = read_json(value['offline_interrupted_owner_recovery_receipt'])
            self.assertTrue(record['exclusive_byte_lock_acquired'])
            self.assertTrue(record['failed_supervisor_owned_native_parent_and_child'])
            self.assertFalse(record['original_native_command_exit_code_known'])
            self.assertFalse(record['runtime_language_model_used'])
            self.assertFalse(record['manual_intervention_used'])
        self.assertEqual(self.lease.read_bytes(), old_lease)
        self.assertEqual(self.guard.read_bytes(), old_guard)
        self.assertEqual(read_json(self.owner_path)['status'], 'released')

    def test_dead_native_parent_and_child_without_failed_supervisor_are_rejected(self):
        self.fixture()
        self.guard.unlink()
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_successful_or_unrelated_guard_does_not_authorize_interruption_recovery(self):
        self.fixture()
        value = read_json(self.guard)
        value['status'], value['returncode'] = 'completed', 0
        atomic_json(self.guard, value)
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))
        value['status'], value['returncode'] = 'failed_controller', 15
        value['owned_process_identities'] = [self.child]
        atomic_json(self.guard, value)
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_live_exact_native_child_prevents_recovery(self):
        self.fixture()
        fake = type('Process', (), {'create_time': lambda item: self.child['creation_time']})()
        def process(pid):
            if pid == self.child['pid']:
                return fake
            raise psutil.NoSuchProcess(pid)
        with patch.object(leases.psutil, 'Process', side_effect=process):
            self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_missing_or_changed_native_lifecycle_is_rejected(self):
        self.fixture()
        value = read_json(self.lifecycle)
        value['command'] = [str(leases.NATIVE_COLMAP), 'different']
        atomic_json(self.lifecycle, value)
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))
        self.lifecycle.unlink()
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_wsl_command_or_quarantined_drain_is_never_reclaimed(self):
        self.command = ['wsl.exe', '--exec', 'python']
        self.fixture()
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))
        self.owner['status'] = 'quarantined_owned_command_not_drained'
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_changed_original_lease_is_not_converted_to_a_valid_owner(self):
        self.fixture()
        value = read_json(self.lease)
        value['pid'] = 900203
        atomic_json(self.lease, value)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            leases.interrupted_native_owner_evidence(self.owner)
        self.assertEqual(read_json(self.owner_path), self.owner)

    def test_unknown_or_changed_supervisor_source_is_rejected(self):
        self.fixture()
        value = read_json(self.guard)
        value['source']['sha256'] = 'invalid'
        atomic_json(self.guard, value)
        self.assertIsNone(leases.interrupted_native_owner_evidence(self.owner))

    def test_unknown_owner_source_is_not_reclaimed(self):
        self.fixture()
        owner = dict(self.owner, source={'path': str(self.root / 'unknown.py')})
        self.assertIsNone(leases.interrupted_native_owner_evidence(owner))


if __name__ == '__main__':
    unittest.main()
