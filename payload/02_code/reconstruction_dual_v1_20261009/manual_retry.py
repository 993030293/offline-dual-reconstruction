"""Archive an explicit failed attempt before reusing same-run completed inputs."""

from pathlib import Path
import shutil
import time

import psutil

from common import atomic_json, read_json

ORDER = ('preflight', 'raw_times', 'front', 'shared', 'target_inputs', 'gaussian',
         'pgsr', 'target_geometry', 'delivery')
FRONT = ('initial', 'sfm', 'recovery', 'temporal', 'pose_gap', 'metric',
         'metric_sim3_renderer_recovery', 'roles', 'export')


def require_stopped(receipt):
    for item in read_json(receipt).get('owned_process_identities', []):
        try:
            if psutil.Process(item['pid']).create_time() == item['creation_time']:
                raise ValueError('An explicitly failed stage still has a live owned process')
        except psutil.NoSuchProcess:
            pass


def archive_failed_front(root, case, leaf, note):
    root = Path(root).resolve(strict=True)
    state = read_json(root / 'STATE.json')
    outer = root / 'stages/front'
    side = root / case / 'optimized'
    native = side / (leaf + '_process.json')
    if (not state['status'].startswith('failed') or state['current_stage'] != 'front'
            or leaf not in FRONT or not note):
        raise ValueError('Manual front retry requires the recorded failed boundary and a repair note')
    if (outer / 'COMPLETED.json').exists() or not read_json(outer / 'PROCESS.json')['status'].startswith('failed'):
        raise ValueError('Only an explicitly unfinished front stage can be archived')
    if not read_json(native)['status'].startswith('failed'):
        raise ValueError('The requested native front leaf did not fail')
    for label in ORDER[ORDER.index('front') + 1:]:
        if (root / 'stages' / label / 'COMPLETED.json').is_file():
            raise ValueError('A downstream completed stage prevents upstream retry')
    require_stopped(outer / 'PROCESS.json')
    require_stopped(native)
    archive = root / 'failures' / ('front_' + leaf + '_' + str(time.time_ns()))
    archive.resolve().relative_to(root)
    paths = [outer, native, side / (leaf + '.log'), side / (leaf + '_wrapper.json'), side / leaf]
    for path in paths:
        path.resolve().relative_to(root)
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('STATE.json', 'EXCEPTION_LEDGER.json', 'RUN_REPORT.md'):
        path = root / name
        if path.is_file():
            shutil.copyfile(path, archive / ('PRIOR_' + name))
    if (side / 'PIPELINE_STATE.json').is_file():
        shutil.copyfile(side / 'PIPELINE_STATE.json', archive / 'PRIOR_FRONT_STATE.json')
    moved = []
    for path in paths:
        if path.exists():
            destination = archive / 'retained' / path.relative_to(root)
            destination.resolve().relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            moved.append({'original': str(path), 'retained': str(destination)})
    atomic_json(archive / 'RETRY.json', {'status': 'explicit_assistant_or_operator_retry',
        'repair_note': note, 'algorithm_autonomous_recovery': False,
        'successful_upstream_same_raw_run_reused': True, 'moved_failed_evidence': moved,
        'single_unassisted_cold_start_claim_allowed': False})
    return archive
