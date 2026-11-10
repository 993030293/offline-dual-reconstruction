"""Verify real Windows heartbeat locks without hiding terminal or permission failures."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import threading
import time
import unittest

import psutil

from common import OLD, SFM_PYTHON, atomic_json, read_json
from atomic_json_windows import delete_sharing_evidence
from run_owned_task import OwnedReceiptPublisher


class LockedReader:
    def __init__(self, path):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        self.close_handle = kernel.CloseHandle
        self.close_handle.argtypes = (wintypes.HANDLE,)
        self.close_handle.restype = wintypes.BOOL
        self.handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if self.handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            self.close_handle(self.handle)
            self.handle = None


@unittest.skipUnless(os.name == 'nt', 'Windows file-sharing semantics are required')
class ReceiptRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.path = self.root / 'PROCESS.json'
        self.publisher = OwnedReceiptPublisher(maximum_retry_seconds=.03)
        self.state = {'status': 'running', 'elapsed_seconds': .1, 'maximum_seconds': 12,
            'maximum_RSS_bytes': 1024**3, 'owned_process_identities': [
                {'pid': 900001, 'creation_time': 50.}]}
        self.publisher.publish(self.path, self.state)

    def lock(self):
        reader = LockedReader(self.path)
        self.addCleanup(reader.close)
        return reader

    def test_actual_locked_reader_is_proved_and_pending_heartbeat_is_later_published(self):
        reader = self.lock()
        evidence = delete_sharing_evidence(self.path)
        self.assertTrue(evidence['verified_delete_sharing_conflict'])
        self.assertEqual(evidence['delete_access_probe_winerror'], 32)
        self.publisher.publish(self.path, self.state)
        records = list(self.root.glob('*.HEARTBEAT_SHARING_*.json'))
        self.assertEqual(len(records), 1)
        proof = read_json(records[0])
        self.assertFalse(proof['runtime_language_model_used'])
        self.assertTrue(proof['final_receipt_publication_still_required'])
        reader.close()
        self.publisher.publish(self.path, dict(self.state, status='completed', returncode=0))
        final = read_json(self.path)
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['receipt_publication_before_current_write']['deferred_heartbeat_publications'], 1)

    def test_final_publication_is_not_deferred_even_when_the_reader_is_proved_locked(self):
        self.lock()
        with self.assertRaises(PermissionError):
            self.publisher.publish(self.path, dict(self.state, status='completed', returncode=0))
        self.assertFalse(list(self.root.glob('*.HEARTBEAT_SHARING_*.json')))
        self.assertEqual(read_json(self.path)['status'], 'running')

    def test_initial_running_publication_is_required_before_deferral(self):
        self.lock()
        publisher = OwnedReceiptPublisher(maximum_retry_seconds=.03)
        with self.assertRaises(PermissionError):
            publisher.publish(self.path, self.state)
        self.assertFalse(list(self.root.glob('*.HEARTBEAT_SHARING_*.json')))

    def test_real_readonly_destination_permission_is_not_a_sharing_recovery(self):
        os.chmod(self.path, stat.S_IREAD)
        try:
            self.assertFalse(delete_sharing_evidence(self.path)['verified_delete_sharing_conflict'])
            with self.assertRaises(PermissionError):
                self.publisher.publish(self.path, self.state)
            self.assertFalse(list(self.root.glob('*.HEARTBEAT_SHARING_*.json')))
        finally:
            os.chmod(self.path, stat.S_IWRITE)

    def test_heartbeat_delay_does_not_extend_the_original_task_deadline(self):
        self.lock()
        with self.assertRaises(TimeoutError):
            self.publisher.publish(self.path, dict(self.state, maximum_seconds=.11))
        self.assertEqual(self.publisher.stats['deferred_heartbeat_publications'], 1)

    def test_real_owned_guard_survives_the_same_four_second_lock_as_the_frozen_failure(self):
        receipt, log = self.root / 'actual_PROCESS.json', self.root / 'actual_child.log'
        command = [str(SFM_PYTHON), str(OLD / 'run_owned_task.py'), '--receipt', str(receipt),
            '--log', str(log), '--maximum-seconds', '12', '--maximum-rss-gib', '1', '--',
            str(SFM_PYTHON), '-c', 'import time; time.sleep(7)']
        with (self.root / 'actual_guard.log').open('x', encoding='utf-8') as stream:
            guard = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            deadline, reader = time.monotonic() + 8, None
            try:
                while time.monotonic() < deadline and guard.poll() is None:
                    if receipt.exists() and read_json(receipt).get('owned_process_identities'):
                        reader = LockedReader(receipt)
                        break
                    time.sleep(.01)
                self.assertIsNotNone(reader)
                timer = threading.Timer(4, reader.close)
                timer.start()
                code = guard.wait(timeout=15)
                timer.join(timeout=5)
            finally:
                if reader is not None:
                    reader.close()
                if guard.poll() is None:
                    guard.kill()
                    guard.wait()
        final = read_json(receipt)
        self.assertEqual(code, 0, (self.root / 'actual_guard.log').read_text(encoding='utf-8'))
        self.assertEqual((final['status'], final['returncode']), ('completed', 0))
        self.assertGreaterEqual(final['receipt_publication_before_current_write']['deferred_heartbeat_publications'], 1)
        self.assertEqual(final['maximum_seconds'], 12)
        self.assertEqual(final['maximum_RSS_bytes'], 1024**3)
        for row in final['owned_process_identities']:
            try:
                self.assertNotEqual(psutil.Process(row['pid']).create_time(), row['creation_time'])
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    unittest.main()
