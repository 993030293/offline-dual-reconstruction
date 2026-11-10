"""Exercise strict typed fetch recovery and the complete native result contract."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import native_packing as packing

URL = 'http://127.0.0.1:8215/object-optimized/cases/packed/export.html'
OFFLINE = {'status': 'only_loopback_requests_observed', 'denied': [], 'requests': []}
NATIVE = {'status': 'both_native_assets_published', 'records': [
    {'complete_native_arrays_equal': True, 'numSplats': 50},
    {'complete_native_arrays_equal': True, 'numSplats': 20}]}


def failure():
    return {'status': 'failed_native_packing_or_viewer', 'phase': 'full_scene_initialization',
        'domLoaded': True, 'browserClosed': True, 'native_assets_validated': True,
        'snapshot': {'documentReadyState': 'complete', 'viewer': {
            'ready': False, 'errors': ['TypeError: Failed to fetch\n    at loadJSON']}},
        'pageErrors': [], 'badResponses': [], 'consoleErrors': ['Failed to load resource: net::ERR_FAILED'],
        'failedRequests': [{'url': URL.replace('export.html', 'camera_path.json'), 'error': 'net::ERR_FAILED'}],
        'native': deepcopy(NATIVE)}


class MetadataFetchPolicyTests(unittest.TestCase):
    def test_observed_loopback_fetch_failure_is_retryable(self):
        self.assertTrue(packing.retryable_metadata_fetch(failure(), OFFLINE, URL))

    def test_original_incomplete_failure_log_is_not_reclassified(self):
        self.assertFalse(packing.retryable_metadata_fetch({'status': 'failed_native_packing_or_viewer',
            'error': 'TypeError: Failed to fetch'}, OFFLINE, URL))

    def test_http_error_is_not_retryable(self):
        value = failure()
        value['badResponses'] = [{'url': value['failedRequests'][0]['url'], 'status': 404}]
        self.assertFalse(packing.retryable_metadata_fetch(value, OFFLINE, URL))

    def test_wrong_native_geometry_or_browser_state_is_not_retryable(self):
        mutations = [lambda v: v.update(browserClosed=False), lambda v: v.update(native_assets_validated=False),
            lambda v: v.update(pageErrors=['SyntaxError']), lambda v: v.update(domLoaded=False),
            lambda v: v.update(phase='packing'), lambda v: v['native']['records'][0].update(complete_native_arrays_equal=False),
            lambda v: v['snapshot']['viewer'].update(errors=['Error: invalid camera model'])]
        for change in mutations:
            with self.subTest(change=change):
                value = failure()
                change(value)
                self.assertFalse(packing.retryable_metadata_fetch(value, OFFLINE, URL))

    def test_unobserved_or_nontransport_failure_is_not_retryable(self):
        for error in (None, 'net::ERR_ABORTED', 'net::ERR_CERT_AUTHORITY_INVALID'):
            value = failure()
            value['failedRequests'][0]['error'] = error
            self.assertFalse(packing.retryable_metadata_fetch(value, OFFLINE, URL))
        value = failure()
        value['failedRequests'] = []
        self.assertFalse(packing.retryable_metadata_fetch(value, OFFLINE, URL))

    def test_other_origin_or_nonmetadata_assets_are_not_retryable(self):
        for url in ('https://example.com/camera_path.json', URL.replace('8215', '8216').replace('export.html', 'camera_path.json'),
                URL.replace('export.html', 'scene.spk')):
            value = failure()
            value['failedRequests'][0]['url'] = url
            self.assertFalse(packing.retryable_metadata_fetch(value, OFFLINE, URL))

    def test_denied_external_requests_are_not_retryable(self):
        offline = {'status': 'external_requests_denied', 'denied': [{'host': 'example.com'}]}
        self.assertFalse(packing.retryable_metadata_fetch(failure(), offline, URL))


class NativePackingControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def write(self, folder, name, value):
        (folder / name).write_text(json.dumps(value), encoding='utf-8')

    def callback(self, fail_first=False, permanent=False, drained=True, wrong_count=False):
        calls = []
        def native(command, folder, label, seconds):
            calls.append((folder, seconds))
            reject = permanent or (fail_first and len(calls) == 1)
            self.write(folder, 'packing_native_COMMAND_LIFECYCLE.json', {'returncode': 1 if reject else 0,
                'owned_tree_drained': drained, 'status': 'failed_native_command' if reject else 'completed_native_command_owned_tree_drained'})
            self.write(folder, 'OFFLINE_BROWSER_RESULT.json', OFFLINE)
            if reject:
                self.write(folder, 'FAILURE.json', failure())
                raise subprocess.CalledProcessError(1, command)
            self.write(folder, 'RESULT.json', {'status': 'native_assets_and_viewer_ready',
                'browserClosed': True, 'pageErrors': [], 'failedRequests': [], 'badResponses': [], 'consoleErrors': [],
                'native': NATIVE, 'viewer': {'ready': True, 'errors': [], 'fullGaussianCount': 51 if wrong_count else 50}})
        return native, calls

    def test_normal_result_uses_one_actual_attempt(self):
        callback, calls = self.callback()
        with patch.object(packing, 'run_native', side_effect=callback):
            result = packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['attempt_count'], 1)
        self.assertEqual(result['original_metadata_fetch_failures'], [])

    def test_first_failure_retained_second_attempt_bounded(self):
        callback, calls = self.callback(fail_first=True)
        with patch.object(packing, 'run_native', side_effect=callback), \
                patch.object(packing, 'require_live_preview') as preview:
            result = packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(calls[1][1], calls[0][1])
        self.assertEqual(result['original_metadata_fetch_failures'][0]['returncode'], 1)
        self.assertTrue((self.output / 'attempt_001/FAILURE.json').exists())
        preview.assert_called_once()

    def test_two_failures_are_terminal_without_third_attempt(self):
        callback, calls = self.callback(permanent=True)
        with patch.object(packing, 'run_native', side_effect=callback), patch.object(packing, 'require_live_preview'):
            with self.assertRaises(subprocess.CalledProcessError):
                packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(len(calls), 2)
        self.assertFalse((self.output / 'RESULT.json').exists())

    def test_undrained_failure_cannot_retry(self):
        callback, calls = self.callback(fail_first=True, drained=False)
        with patch.object(packing, 'run_native', side_effect=callback):
            with self.assertRaises(subprocess.CalledProcessError):
                packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(len(calls), 1)

    def test_missing_server_ownership_cannot_retry(self):
        callback, calls = self.callback(fail_first=True)
        with patch.object(packing, 'run_native', side_effect=callback), \
                patch.object(packing, 'require_live_preview', side_effect=ValueError('owner changed')):
            with self.assertRaisesRegex(ValueError, 'owner changed'):
                packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(len(calls), 1)

    def test_wrong_native_population_cannot_be_success(self):
        callback, _ = self.callback(wrong_count=True)
        with patch.object(packing, 'run_native', side_effect=callback):
            with self.assertRaises(ValueError):
                packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)

    def test_missing_telemetry_preserves_original_native_exit_without_retry(self):
        calls = []
        def missing(command, folder, label, seconds):
            calls.append(folder)
            self.write(folder, 'packing_native_COMMAND_LIFECYCLE.json', {
                'returncode': 1, 'owned_tree_drained': True})
            raise subprocess.CalledProcessError(1, command)
        with patch.object(packing, 'run_native', side_effect=missing):
            with self.assertRaises(subprocess.CalledProcessError) as stopped:
                packing.run_packing('node', 'playwright', URL, self.output, self.output, 10)
        self.assertEqual(stopped.exception.returncode, 1)
        self.assertEqual(len(calls), 1)
        value = json.loads((self.output / 'ATTEMPTS.json').read_text(encoding='utf-8'))['attempts'][0]
        self.assertFalse(value['failure_details_recorded'])
        self.assertFalse(value['offline_details_recorded'])
        self.assertFalse(value['retryable_metadata_fetch'])

    def test_original_maximum_budget_cannot_be_extended(self):
        with self.assertRaises(ValueError):
            packing.run_packing('node', 'playwright', URL, self.output, self.output, 361)


if __name__ == '__main__':
    unittest.main()
