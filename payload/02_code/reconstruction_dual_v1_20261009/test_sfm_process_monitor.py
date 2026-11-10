"""Reproduce the native multi-command process-exit sampling race and real failures."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import psutil

from common import OLD


def load_native():
    spec = importlib.util.spec_from_file_location('native_sfm_monitor_regression', OLD / 'run_sfm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SfmProcessMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native()

    def test_parent_exit_during_children_query_is_not_a_job_failure(self):
        process = SimpleNamespace(pid=101, poll=Mock(side_effect=[None, 0]))
        parent = SimpleNamespace(create_time=lambda: 10.0,
            children=Mock(side_effect=psutil.NoSuchProcess(101)))
        with patch.object(self.native.psutil, 'Process', return_value=parent):
            self.assertEqual(self.native.process_resources([process], {101: 10.0}), 0)
        self.assertEqual(process.poll.call_count, 2)

    def test_nonzero_exit_is_preserved_during_the_same_sampling_race(self):
        process = SimpleNamespace(pid=102, poll=Mock(side_effect=[None, 7]))
        with patch.object(self.native.psutil, 'Process', side_effect=psutil.NoSuchProcess(102)):
            self.assertEqual(self.native.process_resources([process], {102: 10.0}), 0)
        self.assertEqual(process.poll.call_args_list, [call()] * 2)

    def test_unconfirmed_disappearance_is_not_swallowed(self):
        process = SimpleNamespace(pid=103, poll=lambda: None)
        with patch.object(self.native.psutil, 'Process', side_effect=psutil.NoSuchProcess(103)):
            with self.assertRaises(psutil.NoSuchProcess):
                self.native.process_resources([process], {103: 10.0})

    def test_reused_root_pid_cannot_replace_the_owned_birth_identity(self):
        process = SimpleNamespace(pid=104, poll=lambda: None)
        parent = SimpleNamespace(create_time=lambda: 11.0)
        tracked = {104: 10.0}
        with patch.object(self.native.psutil, 'Process', return_value=parent):
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                self.native.process_resources([process], tracked)
        self.assertEqual(tracked, {104: 10.0})

    def test_exited_descendant_does_not_hide_the_live_parent_memory(self):
        process = SimpleNamespace(pid=105, poll=lambda: None)
        child = SimpleNamespace(pid=106, create_time=lambda: 12.0,
            ppid=lambda: 105,
            memory_info=Mock(side_effect=psutil.NoSuchProcess(106)))
        parent = SimpleNamespace(pid=105, create_time=lambda: 10.0,
            children=lambda recursive: [child], memory_info=lambda: SimpleNamespace(rss=1234))
        tracked = {105: 10.0}
        with patch.object(self.native.psutil, 'Process', return_value=parent):
            self.assertEqual(self.native.process_resources([process], tracked), 1234)
        self.assertEqual(tracked[106], 12.0)

    def test_stale_child_of_reused_native_parent_is_not_tracked_or_charged(self):
        process = SimpleNamespace(pid=107, poll=lambda: None)
        orphan = SimpleNamespace(pid=109, create_time=lambda: 11.0, ppid=lambda: 108,
            memory_info=lambda: SimpleNamespace(rss=999999))
        child = SimpleNamespace(pid=108, create_time=lambda: 12.0, ppid=lambda: 107,
            memory_info=lambda: SimpleNamespace(rss=12))
        parent = SimpleNamespace(pid=107, create_time=lambda: 10.0,
            children=lambda recursive: [child, orphan], memory_info=lambda: SimpleNamespace(rss=1234))
        nodes = {107: parent, 108: child, 109: orphan}
        tracked = {107: 10.0}
        with patch.object(self.native.psutil, 'Process', side_effect=lambda pid: nodes[pid]):
            self.assertEqual(self.native.process_resources([process], tracked), 1246)
        self.assertNotIn(109, tracked)
        self.assertEqual(process.ancestry_ownership_rejections[0]['pid'], 109)

    def jobs(self, root):
        # Compile the exact production closure with its original captured values.
        tree = ast.parse((OLD / 'run_sfm.py').read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        worker = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'native_jobs')
        namespace = dict(self.native.__dict__)
        state = {'status': 'test_native_commands', 'stages': []}
        namespace.update(args=SimpleNamespace(output=root), state=state)
        exec(compile(ast.Module(body=[worker], type_ignores=[]), str(OLD / 'run_sfm.py'), 'exec'), namespace)
        return namespace['native_jobs'], state

    def assert_drained(self, stage):
        self.assertTrue(stage['command_drain_verified'])
        self.assertEqual(stage['still_live_owned_births'], [])
        for row in stage['owned_births']:
            try:
                self.assertNotEqual(psutil.Process(row['pid']).create_time(), row['birth'])
            except psutil.NoSuchProcess:
                pass

    def test_real_two_command_completion_preserves_both_exit_codes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, state = self.jobs(root)
            commands = [[sys.executable, '-c', 'import time; time.sleep(.3); print("first completed")'],
                [sys.executable, '-c', 'import time; time.sleep(.9); print("second completed")']]
            lease = {}
            result = jobs('real_success', commands, 10, lease)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['returncodes'], [0, 0])
            self.assert_drained(result)
            self.assertTrue(lease['command_drain_verified'])
            self.assertEqual(len(lease['owned_process_identities']), len(result['owned_births']))
            self.assertIn('first completed', (root / 'real_success/0.log').read_text())
            self.assertEqual(json.loads((root / 'STATE.json').read_text())['stages'][-1], result)

    def test_real_nonzero_worker_is_failed_and_other_worker_is_drained(self):
        with TemporaryDirectory() as directory:
            jobs, state = self.jobs(Path(directory))
            commands = [[sys.executable, '-c', 'import time,sys; time.sleep(.3); sys.exit(7)'],
                [sys.executable, '-c', 'import time; time.sleep(30)']]
            lease = {}
            with self.assertRaises(ChildProcessError):
                jobs('real_failure', commands, 10, lease)
            result = state['stages'][-1]
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(state['status'], 'failed_native_stage')
            self.assertEqual(state['failed_stage'], 'real_failure')
            self.assertEqual(result['returncodes'][0], 7)
            self.assertIn('ChildProcessError', result['traceback'])
            self.assert_drained(result)
            self.assertTrue(lease['command_drain_verified'])

    def test_real_timeout_is_retained_and_owned_workers_exit(self):
        with TemporaryDirectory() as directory:
            jobs, state = self.jobs(Path(directory))
            with self.assertRaisesRegex(RuntimeError, 'budget was exceeded'):
                jobs('real_timeout', [[sys.executable, '-c', 'import time; time.sleep(30)']], .2)
            result = state['stages'][-1]
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(state['failed_stage'], 'real_timeout')
            self.assert_drained(result)


if __name__ == '__main__':
    unittest.main()
