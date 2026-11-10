"""Run staged native prefix lifecycles and retain complete work fees."""

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005'))
import close_ordering_prefix_negative as old
from runtime import atomic_json, data_identity

CODE = Path(__file__).resolve().parent
OUT = old.OWN / 'v7'


def mode_fee(folder, mode):
    result = old.read(folder / 'RESULT.json')
    rows = []
    if mode == 'context':
        for file in folder.glob('CONTEXT_*.json'):
            value = old.read(file)
            captures = [value['restoredFirst']['capture'], value['initialCapture'], value['preLossPose0']['capture']]
            captures.extend(row['capture'] for row in value['views'])
            rows.append({'scope': str(file), 'ordering_write_bytes': value['disposed']['timing']['orderingUpload']['totals']['orderingSubmittedBytes'],
                         'QA_read_bytes': sum(old.pixels(value) for value in captures) + 2 * max(map(old.pixels, captures))})
    elif mode == 'lifecycle':
        for value in result['cases']:
            snapshot = value['restored']['timing'] if value['kind'] == 'fault' else value['observed']
            captures = [row['capture'] for row in value['completedViews']]
            rows.append({'scope': mode + '_' + value['kind'] + '_' + str(value['enabled']),
                         'ordering_write_bytes': snapshot['orderingUpload']['totals']['orderingSubmittedBytes'],
                         'QA_read_bytes': sum(map(old.pixels, captures))})
    elif mode == 'public':
        rows.append({'scope': mode, 'ordering_write_bytes': result['disposed']['timing']['orderingUpload']['totals']['orderingSubmittedBytes'],
                     'QA_read_bytes': 0, 'only_pose_controller_component_no_extra_Gaussian_pixel_capture': True})
    else:
        for value in result['episodes']:
            captures = [value['first'], value['flight']['capture']] + [row['capture'] for row in value['views']]
            rows.append({'scope': 'fresh_complete_' + str(value['hooks']),
                         'ordering_write_bytes': value['disposed']['orderingUpload']['totals']['orderingSubmittedBytes'],
                         'QA_read_bytes': sum(map(old.pixels, captures))})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *[CODE / name for name in ('run_ordering_prefix_lifecycle_v3.cjs',
             'run_ordering_prefix_lifecycle_v2.cjs', 'run_ordering_prefix_lifecycle.cjs',
             'run_ordering_prefix_complete_v3.cjs', 'run_ordering_prefix_complete.cjs',
             'ordering_active_prefix.js', 'gl_resource_generation_v2.js')]]
    ids = [data_identity(path) for path in paths]
    archive = OUT / 'source_snapshot'
    archive.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, archive / (f'{index:03d}_' + path.name))
    previous = old.read(old.OWN / 'NEGATIVE_SOURCE_COST_CLOSURE_002.json')
    for tag in ('c4', 'c5', 'c6'):
        result = old.read(old.OWN / tag / 'RESULT.json')
        if not all(result['checks'].values()) or not all(old.matches(row) for row in result['sources'] + result['inputs']):
            raise AssertionError('A historical complete native prefix identity or quality check changed')
    state = {'status': 'running_V7_missing_validation', 'sources': ids, 'nodes': [], 'new_GPU_fee_rows': [],
             'historical_GPU_fee_bytes': previous['known_GPU_payload_fee_bytes'], 'new_optimization_count': 0}
    atomic_json(OUT / 'STATE.json', state)
    B, C = 177471488, 44171264
    for mode in ('context', 'lifecycle', 'public', 'fresh'):
        folder = OUT / mode
        script = 'run_ordering_prefix_complete_v3.cjs' if mode == 'fresh' else 'run_ordering_prefix_lifecycle_v3.cjs'
        cmd = ['node', str(CODE / script), str(folder)] + ([] if mode == 'fresh' else [mode])
        bounded = [sys.executable, '-X', 'utf8', str(CODE / 'run_bounded_journal_task.py'), '--receipt',
                   str(OUT / (mode + '_GUARD.json')), '--log', str(OUT / (mode + '.log')),
                   '--maximum-seconds', '180', '--maximum-rss-gib', '4', '--', *cmd]
        code = subprocess.call(bounded)
        guard = old.read(OUT / (mode + '_GUARD.json'))
        state['nodes'].append({'mode': mode, 'guard': guard, 'returncode': code})
        atomic_json(OUT / 'STATE.json', state)
        if code or guard['returncode']:
            raise RuntimeError('A complete native prefix validation node failed: ' + mode)
        result = old.read(folder / 'RESULT.json')
        if not all(result['checks'].values()):
            raise AssertionError('A complete native prefix lifecycle contract failed')
        state['new_GPU_fee_rows'].extend(mode_fee(folder, mode))
        fee = state['historical_GPU_fee_bytes'] + sum(row['ordering_write_bytes'] + row['QA_read_bytes'] for row in state['new_GPU_fee_rows'])
        state['net_work_upper_before_remaining_validation'] = (B - C) / (B + fee / 6)
        atomic_json(OUT / 'STATE.json', state)
        print({'mode_passed': mode, 'net_work_upper': state['net_work_upper_before_remaining_validation']}, flush=True)
        if state['net_work_upper_before_remaining_validation'] < .10:
            state.update(status='closed_negative_net_work_upper_under10percent', new_optimization_count=0)
            atomic_json(OUT / 'RESULT.json', state)
            return
    state.update(status='all_missing_V7_validation_passed_pending_separate_review',
                 sources_unchanged=ids == [data_identity(path) for path in paths])
    if not state['sources_unchanged']:
        raise AssertionError('A native prefix validation source changed')
    atomic_json(OUT / 'RESULT.json', state)


if __name__ == '__main__':
    main()
