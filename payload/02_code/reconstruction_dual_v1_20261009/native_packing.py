"""Bound local metadata-fetch recovery while preserving complete native packing."""

from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import psutil

from common import HERE, atomic_json, read_json
from native_command import run_native
from start_viewer import start

TRANSPORT_ERRORS = {'net::ERR_FAILED', 'net::ERR_CONNECTION_RESET', 'net::ERR_CONNECTION_CLOSED',
    'net::ERR_NETWORK_CHANGED', 'net::ERR_EMPTY_RESPONSE'}


def retryable_metadata_fetch(failure, offline, export_url):
    """Accept only observed local transport failures after verified asset publication."""
    source = urlsplit(export_url)
    if source.scheme != 'http' or source.hostname not in ('127.0.0.1', 'localhost', '::1'):
        return False
    paths = {source.path.rsplit('/', 1)[0] + '/' + name
        for name in ('display_manifest.json', 'camera_path.json')}
    snapshot = failure.get('snapshot')
    viewer = snapshot.get('viewer') if isinstance(snapshot, dict) else None
    requests = failure.get('failedRequests')
    if (failure.get('status') != 'failed_native_packing_or_viewer'
            or failure.get('phase') != 'full_scene_initialization'
            or failure.get('domLoaded') is not True or failure.get('browserClosed') is not True
            or failure.get('native_assets_validated') is not True
            or not isinstance(snapshot, dict) or snapshot.get('snapshotError')
            or snapshot.get('documentReadyState') not in ('interactive', 'complete')
            or not isinstance(viewer, dict) or viewer.get('ready') is not False
            or not viewer.get('errors') or not isinstance(requests, list) or not requests
            or failure.get('pageErrors') != [] or failure.get('badResponses') != []
            or offline.get('status') != 'only_loopback_requests_observed' or offline.get('denied') != []):
        return False
    if any(not isinstance(error, str) or error.split('\n')[0] != 'TypeError: Failed to fetch'
            for error in viewer['errors']):
        return False
    native = failure.get('native', {})
    records = native.get('records', [])
    if (native.get('status') != 'both_native_assets_published' or len(records) != 2
            or any(row.get('complete_native_arrays_equal') is not True for row in records)):
        return False
    for row in requests:
        url = urlsplit(row.get('url', ''))
        if ((url.scheme, url.hostname, url.port) != (source.scheme, source.hostname, source.port)
                or url.path not in paths or url.query or url.fragment or url.username or url.password
                or row.get('error') not in TRANSPORT_ERRORS):
            return False
    console = failure.get('consoleErrors')
    if not isinstance(console, list) or any(not isinstance(value, str) or not any(
            value == 'Failed to load resource: ' + error for error in TRANSPORT_ERRORS) for value in console):
        return False
    return True


def require_live_preview(root):
    """Require the original owned server and its delivery identity before recovery."""
    record = read_json(root / 'PREVIEW_SERVER.json')
    expected_script = str(HERE / 'serve_delivery.py')
    expected_root = str(root / 'viewer')
    identities = [(record['pid'], record['birth'])]
    identities += [(row['pid'], row['birth']) for row in record['owned_children']]
    if any(psutil.Process(pid).create_time() != birth for pid, birth in identities):
        raise ValueError('The original packing server ownership changed')
    command = psutil.Process(record['pid']).cmdline()
    if len(command) < 2 or command[1] != expected_script or expected_root not in command:
        raise ValueError('The packing server command is not the original owned local program')
    if start(root) != record:
        raise ValueError('Packing recovery cannot substitute a new preview server')


def run_packing(node, playwright, export_url, output, root, maximum_seconds=360):
    output, root = Path(output), Path(root)
    if not 0 < maximum_seconds <= 360 or (output / 'RESULT.json').exists():
        raise ValueError('Native packing requires a fresh result and the unchanged total budget')
    started = time.monotonic()
    attempts, original_failures = [], []
    for number in (1, 2):
        folder = output / ('attempt_' + str(number).zfill(3))
        folder.mkdir(exist_ok=False)
        remaining = maximum_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('The original combined native packing budget expired')
        command = [node, HERE / 'pack_preview.cjs', playwright, export_url, folder, str(remaining * 1000)]
        try:
            run_native(command, folder, 'packing_native', remaining)
        except subprocess.CalledProcessError as error:
            lifecycle = read_json(folder / 'packing_native_COMMAND_LIFECYCLE.json')
            failure_path, offline_path = folder / 'FAILURE.json', folder / 'OFFLINE_BROWSER_RESULT.json'
            failure = read_json(failure_path) if failure_path.is_file() else {'status': 'failure_details_not_published'}
            offline = read_json(offline_path) if offline_path.is_file() else {'status': 'offline_details_not_published'}
            row = {'attempt': number, 'returncode': error.returncode,
                'failure_receipt': str(folder / 'FAILURE.json'),
                'offline_receipt': str(folder / 'OFFLINE_BROWSER_RESULT.json'),
                'failure_details_recorded': failure_path.is_file(),
                'offline_details_recorded': offline_path.is_file(),
                'lifecycle_receipt': str(folder / 'packing_native_COMMAND_LIFECYCLE.json'),
                'owned_tree_drained': lifecycle.get('owned_tree_drained') is True,
                'retryable_metadata_fetch': retryable_metadata_fetch(failure, offline, export_url),
                'root_cause_established': False}
            attempts.append(row)
            original_failures.append(row)
            atomic_json(output / 'ATTEMPTS.json', {'status': 'original_native_packing_failure_retained',
                'attempts': attempts, 'maximum_attempts': 2, 'maximum_total_seconds': maximum_seconds,
                'export_url': export_url})
            if (number == 2 or error.returncode != 1 or not row['owned_tree_drained']
                    or not row['retryable_metadata_fetch']):
                raise
            require_live_preview(root)
            continue
        result = read_json(folder / 'RESULT.json')
        lifecycle = read_json(folder / 'packing_native_COMMAND_LIFECYCLE.json')
        offline = read_json(folder / 'OFFLINE_BROWSER_RESULT.json')
        native, viewer = result.get('native', {}), result.get('viewer', {})
        records = native.get('records', [])
        if (result['status'] != 'native_assets_and_viewer_ready' or result.get('browserClosed') is not True
                or result.get('pageErrors') != [] or result.get('failedRequests') != []
                or result.get('badResponses') != [] or result.get('consoleErrors') != []
                or lifecycle['returncode'] != 0 or lifecycle.get('owned_tree_drained') is not True
                or offline['status'] != 'only_loopback_requests_observed' or offline['denied'] != []
                or native.get('status') != 'both_native_assets_published' or len(records) != 2
                or any(row.get('complete_native_arrays_equal') is not True for row in records)
                or viewer.get('ready') is not True or viewer.get('errors') != []
                or viewer.get('fullGaussianCount') != records[0].get('numSplats')):
            raise ValueError('The actual native assets, browser, network, or owned tree failed acceptance')
        attempts.append({'attempt': number, 'returncode': 0, 'result_receipt': str(folder / 'RESULT.json'),
            'offline_receipt': str(folder / 'OFFLINE_BROWSER_RESULT.json'),
            'lifecycle_receipt': str(folder / 'packing_native_COMMAND_LIFECYCLE.json'), 'owned_tree_drained': True})
        result.update(attempt_count=number, attempts=attempts,
            original_metadata_fetch_failures=original_failures, maximum_attempts=2,
            maximum_total_seconds=maximum_seconds, combined_elapsed_seconds=time.monotonic() - started,
            bounded_transport_recovery_is_not_root_cause_proof=True,
            runtime_language_model_used=False, automatic_runtime_code_repair_used=False,
            export_url=export_url)
        if result['combined_elapsed_seconds'] > maximum_seconds:
            raise TimeoutError('Native packing finished beyond its original combined budget')
        atomic_json(output / 'OFFLINE_BROWSER_RESULT.json', offline)
        atomic_json(output / 'RESULT.json', result)
        atomic_json(output / 'ATTEMPTS.json', {'status': 'actual_native_packing_complete',
            'attempts': attempts, 'maximum_attempts': 2, 'maximum_total_seconds': maximum_seconds,
            'export_url': export_url})
        return result
    raise RuntimeError('No native packing attempt completed')


def packing_recovery_accounting(root):
    """Re-read raw failures, offline observations and native command drain evidence."""
    return packing_directory_accounting(Path(root) / 'native_packing')


def packing_directory_accounting(folder):
    """Check a declared packing directory without rewriting its actual records."""
    folder = Path(folder)
    path = folder / 'ATTEMPTS.json'
    if not path.is_file():
        return {'attempts': [], 'failures': [], 'verified': True,
            'explicit_retry_contract_present': False, 'legacy_result_does_not_claim_retry': True}
    contract = read_json(path)
    rows = contract['attempts']
    valid = (contract['status'] == 'actual_native_packing_complete'
        and contract['maximum_attempts'] == 2 and 0 < contract['maximum_total_seconds'] <= 360
        and 1 <= len(rows) <= 2 and [row['attempt'] for row in rows] == list(range(1, len(rows) + 1))
        and rows[-1]['returncode'] == 0)
    failures = []
    for row in rows:
        attempt_folder = folder / ('attempt_' + str(row['attempt']).zfill(3))
        lifecycle_path = Path(row['lifecycle_receipt']).resolve(strict=True)
        offline_path = Path(row['offline_receipt']).resolve()
        if (lifecycle_path != (attempt_folder / 'packing_native_COMMAND_LIFECYCLE.json').resolve(strict=True)
                or offline_path != (attempt_folder / 'OFFLINE_BROWSER_RESULT.json').resolve()):
            raise ValueError('A native packing recovery record crosses attempts')
        lifecycle = read_json(lifecycle_path)
        offline = read_json(offline_path) if offline_path.is_file() else {'status': 'offline_details_not_published'}
        command = lifecycle['command']
        valid &= (lifecycle.get('owned_tree_drained') is True and lifecycle['returncode'] == row['returncode']
            and lifecycle.get('status') == ('failed_native_command' if row['returncode'] else 'completed_native_command_owned_tree_drained')
            and bool(lifecycle.get('owned_process_identities'))
            and len(command) == 6 and Path(command[1]).resolve() == (HERE / 'pack_preview.cjs').resolve()
            and command[3] == contract['export_url'] and Path(command[4]).resolve() == attempt_folder.resolve()
            and 0 < float(command[5]) <= contract['maximum_total_seconds'] * 1000
            and offline['status'] == 'only_loopback_requests_observed' and offline['denied'] == [])
        for identity in lifecycle['owned_process_identities']:
            try:
                if psutil.Process(identity['pid']).create_time() == identity['creation_time']:
                    valid = False
            except psutil.NoSuchProcess:
                pass
        if row['returncode']:
            failure_path = Path(row['failure_receipt']).resolve()
            if failure_path != (attempt_folder / 'FAILURE.json').resolve():
                raise ValueError('A native packing failure belongs to another attempt')
            raw_failure = read_json(failure_path) if failure_path.is_file() else {'status': 'failure_details_not_published'}
            actual_retryable = retryable_metadata_fetch(raw_failure, offline, contract['export_url'])
            valid &= row['returncode'] == 1 and row.get('retryable_metadata_fetch') is True and actual_retryable
            failures.append({**row, 'actual_raw_failure_revalidated': actual_retryable})
        else:
            result_path = Path(row['result_receipt']).resolve(strict=True)
            if result_path != (attempt_folder / 'RESULT.json').resolve(strict=True):
                raise ValueError('A native packing result belongs to another attempt')
            value = read_json(result_path)
            valid &= value.get('browserClosed') is True and value['status'] == 'native_assets_and_viewer_ready'
    final = read_json(folder / 'RESULT.json') if (folder / 'RESULT.json').is_file() else {}
    valid &= (final.get('attempts') == rows and final.get('original_metadata_fetch_failures') ==
        [row for row in rows if row['returncode']] and final.get('maximum_total_seconds') == contract['maximum_total_seconds']
        and final.get('combined_elapsed_seconds', float('inf')) <= contract['maximum_total_seconds'])
    return {'attempts': rows, 'failures': failures, 'verified': bool(valid),
        'explicit_retry_contract_present': True, 'root_cause_established': False}
