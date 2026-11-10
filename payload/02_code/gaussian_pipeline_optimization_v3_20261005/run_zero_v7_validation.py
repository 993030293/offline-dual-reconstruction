"""Complete missing R157 validation while preserving all historical evidence."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

import run_native_zero_feature_campaign as parent
from runtime import atomic_json, data_identity

OUT = parent.OUT / 'v7'


def run(mode, entry, label, leaf=None, checked=False):
    leaf = leaf or OUT / label
    node = OUT / (label + '_node')
    command = parent.command(leaf, node, mode)
    command[command.index(str(parent.CODE / 'native_zero_feature_node.py'))] = str(
        parent.CODE / ('native_zero_checked_node.py' if checked else 'native_zero_v7_node.py'))
    original_entry = '/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_native_feature_zero.py'
    command[command.index(original_entry)] = original_entry.replace('profile_native_feature_zero.py', entry)
    begin = time.perf_counter()
    with (OUT / (label + '.log')).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        owner = {'pid': process.pid, 'creation_time': psutil.Process(process.pid).create_time()}
        code = process.wait(timeout=380)
    row = {'owner': owner, 'actual_spawn_to_exit_seconds': time.perf_counter() - begin,
           'returncode': code, 'leaf': str(leaf), 'node': str(node), 'command': command}
    atomic_json(OUT / (label + '_EXIT.json'), row)
    if code:
        raise RuntimeError('A bounded R157 validation node failed: ' + label)
    record = parent.read(node / 'NODE.json')
    if (record['status'] != 'completed' or record['live_Windows_after_wait']
            or any(value['alive'] for value in record['actual_Linux_after_wait'])
            or not record['Linux_resource_receipts']):
        raise AssertionError('A bounded R157 process did not close with native evidence')
    print({'node_completed': label, 'seconds': row['actual_spawn_to_exit_seconds']}, flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[parent.CODE / name for name in (
        'profile_native_zero_sigint_v7.py', 'native_zero_v7_node.py', 'profile_native_zero_fault.py',
        'check_zero_transfer_contracts.py', 'native_feature_zero.py', 'native_feature_zero_v2.py',
        'native_transfer_meter.py', 'native_zero_checked_node.py', 'profile_native_zero_bounded.py')]]
    ids = [data_identity(path) for path in sources]
    archive = OUT / 'source_snapshot'
    archive.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, archive / (f'{index:03d}_' + path.name))
    old = parent.read(parent.OUT / 'c2/RESULT.json')
    if old['sources'] != [data_identity(Path(row['path'])) for row in old['sources']]:
        raise AssertionError('A historical complete-pair source changed; do not reuse it')
    for pair in old['pairs']:
        root = parent.OUT / 'c2' / ('t' + str(pair['index']))
        parent.compare(root / 'native', root / 'direct')
    partial = {'status': 'running_missing_V7_validation', 'sources': ids,
               'historical_three_pairs_source_and_contracts_rechecked': True, 'new_optimization_count': 0}
    atomic_json(OUT / 'STATE.json', partial)
    generic = run('native', 'check_zero_transfer_contracts.py', 'generic', leaf=parent.OUT / 'g3', checked=True)
    g = parent.read(parent.OUT / 'g3/RESULT.json')
    if not all(g['checks'].values()):
        raise AssertionError('The actual public or native generic contract failed')
    faults = {mode: run(mode, 'profile_native_zero_fault.py', 'fault_' + mode) for mode in ('native', 'direct')}
    f = {mode: parent.read(Path(row['leaf']) / 'FAULT_RESULT.json') for mode, row in faults.items()}
    checks = {'all_native_meter_generic_public_checks': all(g['checks'].values()),
              'both_full_original_error_prefixes_and_inputs_exact':
              all(all(value['checks'].values()) for value in f.values())
              and f['native']['error'] == f['direct']['error']
              and f['native']['evidence']['actual_original_model_prefix'] == f['direct']['evidence']['actual_original_model_prefix']
              and f['native']['inputs'] == f['direct']['inputs']}
    if not all(checks.values()):
        raise AssertionError('The actual original complete error contract changed')
    partial.update(generic=generic, faults=faults, checks=checks)
    atomic_json(OUT / 'STATE.json', partial)
    interrupts = {mode: run(mode, 'profile_native_zero_sigint_v7.py', 'sigint_' + mode) for mode in ('native', 'direct')}
    i = {mode: parent.read(Path(row['leaf']) / 'FAULT_RESULT.json') for mode, row in interrupts.items()}
    checks['both_real_SIGINT_complete_inputs_prefix_and_drain_exact'] = (
        all(all(value['checks'].values()) for value in i.values())
        and i['native']['error'] == i['direct']['error']
        and i['native']['evidence']['actual_original_model_prefix'] == i['direct']['evidence']['actual_original_model_prefix']
        and i['native']['inputs'] == i['direct']['inputs'])
    if not checks['both_real_SIGINT_complete_inputs_prefix_and_drain_exact']:
        raise AssertionError('The actual complete native SIGINT contract changed')
    private = OUT / 'private_recovery_copy'
    private.mkdir()
    removed = []
    for name in ('input.ply', 'cameras.json'):
        source = Path(faults['direct']['leaf']) / 'scene' / name
        destination = private / name
        shutil.copyfile(source, destination)
        if destination.resolve().parent != private.resolve() or destination.is_symlink():
            raise ValueError('A private derived recovery copy escaped its directory')
        before = data_identity(destination)
        destination.unlink()
        removed.append({'private_copy': before, 'removed': not destination.exists(),
                        'failed_original_retained': source.is_file()})
    recovery = run('direct', 'profile_native_zero_bounded.py', 'recovered', checked=True)
    comparison = parent.compare(parent.OUT / 'c2/t0/native', Path(recovery['leaf']))
    checks.update(fresh_complete1016_recovery_all_original_contracts=all(comparison['checks'].values()),
                  own_two_derived_copies_deleted_without_touching_originals=len(removed) == 2
                  and all(value['removed'] and value['failed_original_retained'] for value in removed),
                  all_algorithm_and_validation_sources_unchanged=ids == [data_identity(path) for path in sources],
                  family_output_disk_budget=sum(path.stat().st_size for path in parent.OUT.rglob('*') if path.is_file()) <= 2 * 1024**3
                  and shutil.disk_usage(OUT).free >= 20 * 1024**3)
    work = g['extra_actual_native_HtoD_bytes'] + sum(value['extra_actual_native_HtoD_bytes'] for value in (*f.values(), *i.values()))
    work += parent.read(Path(recovery['leaf']) / 'ZERO_TRANSFER_TRACE.json')['actual_native_HtoD_bytes']
    result = {'status': 'missing_V7_validation_passed' if all(checks.values()) else 'failed',
              'checks': checks, 'sources': ids, 'generic': generic, 'faults': faults, 'interrupts': interrupts,
              'private_removed': removed, 'recovery': recovery, 'comparison': comparison,
              'extra_actual_native_HtoD_bytes': work, 'new_optimization_count': 0,
              'historical_complete_pairs_are_observed_development_evidence_not_fresh_or_blind': True}
    atomic_json(OUT / 'RESULT.json', result)
    atomic_json(OUT / 'STATE.json', result)
    if not all(checks.values()):
        raise AssertionError('A complete R157 V7 validation check failed')


if __name__ == '__main__':
    main()
