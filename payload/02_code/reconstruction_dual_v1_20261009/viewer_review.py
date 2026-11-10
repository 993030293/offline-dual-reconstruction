"""Bound isolated browser retries and retain every original initialization failure."""

from pathlib import Path
import subprocess
import time

from common import HERE, atomic_json, read_json
from native_command import run_native


def retryable_initialization(failure):
    snapshot = failure.get('snapshot')
    if not isinstance(snapshot, dict) or snapshot.get('snapshotError'):
        return False
    viewer = snapshot.get('viewer')
    return bool(failure.get('status') == 'failed_viewer_interaction'
        and failure.get('phase') == 'initialization'
        and failure.get('failure_kind') == 'readiness_timeout'
        and failure.get('readiness_timeout_ms') == 90000
        and failure.get('domLoaded') is True and failure.get('browserClosed') is True
        and snapshot.get('documentReadyState') in ('interactive', 'complete')
        and not snapshot.get('error')
        and (viewer is None or (isinstance(viewer, dict) and viewer.get('ready') is False
                               and viewer.get('errors') == []))
        and all(failure.get(key) == [] for key in
                ('denied', 'pageErrors', 'consoleErrors', 'badResponses', 'failedRequests')))


def run_review(node, playwright, base_url, output, branch, point_page, maximum_seconds=270):
    output = Path(output)
    if not 0 < maximum_seconds <= 270 or (output / 'RESULT.json').exists():
        raise ValueError('Viewer review requires a fresh result and the original bounded delivery budget')
    started = time.monotonic()
    attempts, original_failures = [], []
    for number in (1, 2):
        folder = output / ('attempt_' + str(number).zfill(3))
        folder.mkdir(exist_ok=False)
        command = [node, HERE / 'review_viewers.cjs', playwright, base_url, folder, branch, point_page]
        remaining = maximum_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('The combined original viewer review budget was exhausted')
        try:
            run_native(command, folder, 'review_native', remaining)
        except subprocess.CalledProcessError as error:
            lifecycle = read_json(folder / 'review_native_COMMAND_LIFECYCLE.json')
            failure = read_json(folder / 'FAILURE.json')
            row = {'attempt': number, 'returncode': error.returncode,
                'failure_receipt': str(folder / 'FAILURE.json'),
                'lifecycle_receipt': str(folder / 'review_native_COMMAND_LIFECYCLE.json'),
                'owned_tree_drained': lifecycle.get('owned_tree_drained') is True,
                'retryable_initialization': retryable_initialization(failure),
                'root_cause_established': False}
            attempts.append(row)
            original_failures.append(row)
            atomic_json(output / 'ATTEMPTS.json', {'status': 'original_browser_failure_retained',
                'attempts': attempts, 'maximum_attempts': 2, 'maximum_total_seconds': maximum_seconds})
            if (number == 2 or error.returncode != 1 or not row['owned_tree_drained']
                    or not row['retryable_initialization']):
                raise
            continue
        lifecycle = read_json(folder / 'review_native_COMMAND_LIFECYCLE.json')
        result = read_json(folder / 'RESULT.json')
        if (lifecycle.get('owned_tree_drained') is not True or lifecycle.get('returncode') != 0
                or result.get('browserClosed') is not True
                or result.get('status') != 'both_viewers_actual_interactions_verified'
                or result.get('onlyLoopbackRequestsAllowed') is not True
                or {row['kind'] for row in result['rows']} != {'gaussian', 'surface'}
                or len(result['rows']) != 2
                or any(result.get(key) != [] for key in
                       ('pageErrors', 'consoleErrors', 'badResponses', 'failedRequests', 'externalRequestsDenied'))):
            raise ValueError('A successful process did not verify both actual offline viewers')
        for row in result['rows']:
            if (not row['final']['ready'] or row['final']['errors']
                    or row['flightMeters'] <= .08 or row['ascentMeters'] <= .06
                    or not row['cropSwitchVerified'] or not row['capturePathPlaybackVerified']):
                raise ValueError('An actual interaction did not meet the unchanged viewer contract')
        attempts.append({'attempt': number, 'returncode': 0,
            'result_receipt': str(folder / 'RESULT.json'),
            'lifecycle_receipt': str(folder / 'review_native_COMMAND_LIFECYCLE.json'), 'owned_tree_drained': True})
        result.update(attempt_count=number, attempts=attempts, original_initialization_failures=original_failures,
            maximum_attempts=2, maximum_total_seconds=maximum_seconds,
            elapsed_seconds=time.monotonic() - started, all_attempts_owned_trees_drained=True,
            automatic_runtime_code_repair_used=False, runtime_language_model_used=False,
            initialization_retry_is_bounded_operational_recovery_not_root_cause_proof=True)
        atomic_json(output / 'RESULT.json', result)
        atomic_json(output / 'ATTEMPTS.json', {'status': 'actual_viewer_review_complete', 'attempts': attempts,
            'maximum_attempts': 2, 'maximum_total_seconds': maximum_seconds})
        return result
    raise RuntimeError('No viewer review attempt completed')
