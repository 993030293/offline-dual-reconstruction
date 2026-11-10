"""Exercise real timeout and descendant cleanup without using GPU computations."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import psutil

from native_command import run_native
import gpu_lease as leases


class NativeCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shared = self.root / 'shared'
        self.binding = patch.object(leases, 'SHARED', self.shared)
        self.binding.start()
        self.addCleanup(self.binding.stop)
        self.addCleanup(self.temporary.cleanup)

    def command(self, exit_code=None):
        identity = self.root / 'descendant.json'
        script = '\n'.join([
            'import json, pathlib, psutil, subprocess, sys, time',
            'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])',
            'born = psutil.Process(child.pid).create_time()',
            'pathlib.Path(sys.argv[1]).write_text(json.dumps({"pid": child.pid, "born": born}))',
            'time.sleep(30)' if exit_code is None else 'time.sleep(2)',
            'sys.exit(0)' if exit_code is None else 'sys.exit(' + str(exit_code) + ')',
        ])
        return [sys.executable, '-c', script, str(identity)]

    def assert_descendant_gone(self):
        identity = json.loads((self.root / 'descendant.json').read_text())
        try:
            process = psutil.Process(identity['pid'])
            self.assertNotEqual(process.create_time(), identity['born'], 'An owned descendant is still alive')
        except psutil.NoSuchProcess:
            pass

    def assert_released(self):
        owner = json.loads((self.shared / 'OWNER.json').read_text())
        self.assertEqual(owner['status'], 'released')
        self.assertTrue(owner['command_drain_verified'])

    def test_timeout_drains_real_descendant_before_lease_release(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_native(self.command(), self.root, 'timeout', 1.5, gpu=True)
        self.assert_descendant_gone()
        self.assert_released()

    def test_successful_native_command_releases_only_after_exit(self):
        elapsed = run_native([sys.executable, '-c', 'print("finite command completed")'],
            self.root, 'success', 10, gpu=True)
        self.assertGreater(elapsed, 0)
        self.assert_released()
        self.assertIn('finite command completed', (self.root / 'success.log').read_text())

    def test_native_failure_drains_real_descendant_and_preserves_exit_code(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_native(self.command(7), self.root, 'failure', 10, gpu=True)
        self.assertEqual(failure.exception.returncode, 7)
        self.assert_descendant_gone()
        self.assert_released()

    def test_success_with_lingering_descendant_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'live descendant'):
            run_native(self.command(0), self.root, 'lingering', 10, gpu=True)
        self.assert_descendant_gone()
        self.assert_released()
        lifecycle = json.loads((self.root / 'lingering_COMMAND_LIFECYCLE.json').read_text())
        self.assertEqual(lifecycle['status'], 'failed_native_command')
        self.assertTrue(lifecycle['owned_tree_drained'])
        self.assertEqual(lifecycle['returncode'], 0)
        self.assertTrue(lifecycle['owned_at_parent_exit'])

    def test_parent_exit_snapshot_is_rechecked_after_natural_kernel_exit(self):
        import native_command
        actual_live = native_command.live_owned
        actual_popen = native_command.subprocess.Popen
        references = []
        snapshots = [0]

        def capture_process(*arguments, **keywords):
            child = actual_popen(*arguments, **keywords)
            references.append(psutil.Process(child.pid))
            return child

        def stale_exit_snapshot(tracked):
            owned = actual_live(tracked)
            if not owned and tracked and snapshots[0] == 0:
                snapshots[0] += 1
                return references[:1]
            return owned

        with patch.object(native_command, 'live_owned', side_effect=stale_exit_snapshot), \
                patch.object(native_command.subprocess, 'Popen', side_effect=capture_process):
            run_native([sys.executable, '-c', 'print("finite command completed")'],
                self.root, 'kernel_exit', 10, gpu=True)
        self.assertEqual(snapshots[0], 1)
        self.assert_released()
        lifecycle = json.loads((self.root / 'kernel_exit_COMMAND_LIFECYCLE.json').read_text())
        self.assertEqual(lifecycle['status'], 'completed_native_command_owned_tree_drained')
        self.assertTrue(lifecycle['owned_tree_drained'])
        self.assertTrue(lifecycle['owned_at_parent_exit'])
        self.assertLessEqual(lifecycle['natural_exit_wait_seconds'], 1.1)

    def test_undrained_lease_is_quarantined_and_blocks_next_owner(self):
        with self.assertRaisesRegex(RuntimeError, 'simulated drain failure'):
            with leases.gpu_lease(self.root / 'quarantine.json', 'test') as state:
                state['command_drain_verified'] = False
                raise RuntimeError('simulated drain failure')
        owner = json.loads((self.shared / 'OWNER.json').read_text())
        self.assertEqual(owner['status'], 'quarantined_owned_command_not_drained')
        with self.assertRaisesRegex(RuntimeError, 'prior GPU owner'):
            with leases.gpu_lease(self.root / 'next.json', 'test'):
                self.fail('A quarantined lease cannot be claimed')

    def test_windows_guard_rejects_wsl_commands(self):
        with self.assertRaisesRegex(ValueError, 'finite native'):
            run_native(['wsl.exe', 'sleep', '1'], self.root, 'wsl', 1, gpu=True)
        self.assertFalse((self.root / 'wsl.log').exists())

    def test_existing_lease_callers_without_drain_marker_keep_release_behavior(self):
        with leases.gpu_lease(self.root / 'legacy.json', 'legacy'):
            pass
        owner = json.loads((self.shared / 'OWNER.json').read_text())
        self.assertEqual(owner['status'], 'released')
        self.assertNotIn('command_drain_verified', owner)

    def test_sequential_native_gpu_commands_have_distinct_preserved_receipts(self):
        command = [sys.executable, '-c', 'print("finite command completed")']
        for label in ('features', 'matching'):
            run_native(command, self.root, label, 10, gpu=True,
                lease_filename=label + '_GPU_LEASE.json', lease_label='bridge_' + label)
        for label in ('features', 'matching'):
            value = json.loads((self.root / (label + '_GPU_LEASE.json')).read_text())
            self.assertEqual(value['status'], 'released')
            self.assertTrue(value['command_drain_verified'])
            self.assertEqual(value['label'], 'bridge_' + label)

    def test_lease_filename_cannot_escape_stage_output(self):
        with self.assertRaisesRegex(ValueError, 'local JSON filename'):
            run_native([sys.executable, '-c', 'pass'], self.root, 'invalid', 10,
                gpu=True, lease_filename='../outside.json')
        self.assertFalse((self.root / 'invalid.log').exists())


if __name__ == '__main__':
    unittest.main()
