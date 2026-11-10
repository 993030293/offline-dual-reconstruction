"""Reject hidden browser failures and retain bounded offline retry evidence."""

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common import atomic_json, read_json
from viewer_review import retryable_initialization, run_review


def initialization_timeout():
    return {'status': 'failed_viewer_interaction', 'phase': 'initialization',
        'failure_kind': 'readiness_timeout', 'readiness_timeout_ms': 90000,
        'domLoaded': True, 'browserClosed': True,
        'snapshot': {'documentReadyState': 'complete', 'error': '', 'viewer': None},
        'denied': [], 'pageErrors': [], 'consoleErrors': [], 'badResponses': [], 'failedRequests': []}


def successful_review():
    return {'status': 'both_viewers_actual_interactions_verified', 'browserClosed': True,
        'onlyLoopbackRequestsAllowed': True, 'pageErrors': [], 'consoleErrors': [],
        'badResponses': [], 'failedRequests': [], 'externalRequestsDenied': [],
        'rows': [{'kind': kind, 'final': {'ready': True, 'errors': []}, 'flightMeters': .12,
            'ascentMeters': .07, 'cropSwitchVerified': True, 'capturePathPlaybackVerified': True}
            for kind in ('gaussian', 'surface')]}


class ViewerReviewTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.output = Path(self.scratch.name)
        self.calls = []

    def native(self, outcomes, drained=True):
        def run(command, folder, label, maximum_seconds):
            self.calls.append({'folder': folder, 'budget': maximum_seconds})
            outcome = outcomes[len(self.calls) - 1]
            failure = outcome.get('status') != 'both_viewers_actual_interactions_verified'
            code = outcome.get('native_returncode', 1) if failure else 0
            atomic_json(folder / (label + '_COMMAND_LIFECYCLE.json'),
                {'returncode': code, 'owned_tree_drained': drained})
            atomic_json(folder / ('FAILURE.json' if failure else 'RESULT.json'), outcome)
            if failure:
                raise subprocess.CalledProcessError(code, command)
            return 1.
        return run

    def review(self):
        return run_review('node', 'playwright', 'http://127.0.0.1:8214/', self.output, 'case-optimized', 'target/index.html')

    def test_only_complete_closed_error_free_initialization_timeout_is_retryable(self):
        self.assertTrue(retryable_initialization(initialization_timeout()))
        for field, value in [('phase', 'interaction'), ('failure_kind', 'viewer_error'),
                ('browserClosed', False), ('domLoaded', False), ('readiness_timeout_ms', 180000)]:
            with self.subTest(field=field):
                self.assertFalse(retryable_initialization({**initialization_timeout(), field: value}))

    def test_script_network_console_or_unknown_diagnostic_never_retries(self):
        for field in ('denied', 'pageErrors', 'consoleErrors', 'badResponses', 'failedRequests'):
            with self.subTest(field=field):
                self.assertFalse(retryable_initialization({**initialization_timeout(), field: ['error']}))
                value = initialization_timeout()
                del value[field]
                self.assertFalse(retryable_initialization(value))

    def test_known_viewer_error_or_failed_snapshot_is_not_a_transient_timeout(self):
        for snapshot in (None, {'snapshotError': 'renderer not responding'},
                {'documentReadyState': 'loading', 'viewer': None},
                {'documentReadyState': 'complete', 'error': 'asset integrity error', 'viewer': None},
                {'documentReadyState': 'complete', 'viewer': {'ready': False, 'errors': ['bad model']}},
                {'documentReadyState': 'complete', 'viewer': {'ready': True, 'errors': []}}):
            with self.subTest(snapshot=snapshot):
                self.assertFalse(retryable_initialization({**initialization_timeout(), 'snapshot': snapshot}))

    def test_success_uses_one_closed_isolated_attempt(self):
        with patch('viewer_review.run_native', self.native([successful_review()])):
            value = self.review()
        self.assertEqual(value['attempt_count'], 1)
        self.assertEqual(value['original_initialization_failures'], [])
        self.assertTrue(value['all_attempts_owned_trees_drained'])
        self.assertEqual(len(self.calls), 1)

    def test_automatic_retry_preserves_original_exit_one_and_uses_remaining_budget(self):
        with patch('viewer_review.run_native', self.native([initialization_timeout(), successful_review()])), \
                patch('viewer_review.time', SimpleNamespace(monotonic=Mock(side_effect=[0., 0., 5., 6.]))):
            value = self.review()
        self.assertEqual(value['attempt_count'], 2)
        self.assertEqual(value['original_initialization_failures'][0]['returncode'], 1)
        self.assertFalse(value['original_initialization_failures'][0]['root_cause_established'])
        self.assertTrue((self.output / 'attempt_001/FAILURE.json').is_file())
        self.assertEqual([row['budget'] for row in self.calls], [270., 265.])
        self.assertNotEqual(self.calls[0]['folder'], self.calls[1]['folder'])

    def test_actual_viewer_error_fails_closed_without_retry(self):
        failure = {**initialization_timeout(), 'pageErrors': ['SyntaxError']}
        with patch('viewer_review.run_native', self.native([failure])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.review()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.output / 'RESULT.json').exists())

    def test_two_timeouts_preserve_both_failures_and_stop(self):
        with patch('viewer_review.run_native', self.native([initialization_timeout(), initialization_timeout()])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.review()
        value = read_json(self.output / 'ATTEMPTS.json')
        self.assertEqual([row['returncode'] for row in value['attempts']], [1, 1])
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / 'RESULT.json').exists())

    def test_unverified_process_cleanup_cannot_start_another_browser(self):
        with patch('viewer_review.run_native', self.native([initialization_timeout()], drained=False)):
            with self.assertRaises(subprocess.CalledProcessError):
                self.review()
        self.assertEqual(len(self.calls), 1)

    def test_unknown_native_exit_code_cannot_start_another_browser(self):
        with patch('viewer_review.run_native', self.native([{**initialization_timeout(), 'native_returncode': 7}])):
            with self.assertRaises(subprocess.CalledProcessError):
                self.review()
        self.assertEqual(len(self.calls), 1)

    def test_zero_exit_without_both_real_interaction_records_is_rejected(self):
        incomplete = successful_review()
        incomplete['rows'] = incomplete['rows'][:1]
        with patch('viewer_review.run_native', self.native([incomplete])):
            with self.assertRaises(ValueError):
                self.review()
        self.assertFalse((self.output / 'RESULT.json').exists())

    def test_original_interaction_thresholds_are_still_required(self):
        failed = successful_review()
        failed['rows'][0]['ascentMeters'] = .01
        with patch('viewer_review.run_native', self.native([failed])):
            with self.assertRaises(ValueError):
                self.review()
        self.assertFalse((self.output / 'RESULT.json').exists())

    def test_existing_attempt_evidence_is_never_overwritten(self):
        (self.output / 'attempt_001').mkdir()
        with patch('viewer_review.run_native') as run:
            with self.assertRaises(FileExistsError):
                self.review()
        run.assert_not_called()

    def test_delivery_budget_cannot_be_extended(self):
        with self.assertRaises(ValueError):
            run_review('node', 'playwright', 'http://127.0.0.1/', self.output, 'case', 'target', 300)


if __name__ == '__main__':
    unittest.main()
