"""Run two real isolated PGSR iterations to check diagnostic/training counter isolation."""

import argparse
import os
from pathlib import Path
import shutil

from common import HERE, atomic_json, read_json, execute, linux_command, linux_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-counter-failure', action='store_true')
    args = parser.parse_args()
    production, output = args.production_root.resolve(), args.output.resolve()
    output.relative_to(production / 'diagnostics')
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(production / 'CONFIG.json')
    config['pgsr_iterations'] = 2
    atomic_json(output / 'CONFIG.json', config)
    atomic_json(output / 'DIAGNOSTIC_SPEC.json', {'purpose': 'real_native_training_iteration_counter_regression',
        'iterations': 2, 'source': str(production / 'shared'),
        'formal_full_training_or_new_reconstruction_claimed': False,
        'production_models_modified': False, 'expected_counter_failure': args.expect_counter_failure})
    shared = output / 'shared'
    dataset = shared / 'pgsr'
    (dataset / 'images').mkdir(parents=True)
    shutil.copytree(production / 'shared/pgsr/sparse', dataset / 'sparse')
    for source in (production / 'shared/pgsr/images').iterdir():
        if not source.is_file():
            raise ValueError('The validated flat image input changed')
        os.link(source, dataset / 'images' / source.name)
    atomic_json(shared / 'RESULT.json', read_json(production / 'shared/RESULT.json'))
    try:
        result = execute(output, 'training_counter',
            linux_command(HERE / 'pgsr_branch.py', 'train', linux_path(output)), 300,
            output / 'pgsr/TRAINING_RESULT.json', gpu=True)
    except RuntimeError:
        failure = read_json(output / 'pgsr/TRAINING_FAILURE.json')
        if (not args.expect_counter_failure or failure['iteration'] != 3 or failure['target_iterations'] != 2
                or 'PGSR did not finish every requested iteration' not in failure['traceback']):
            raise
        atomic_json(output / 'RESULT.json', {'status': 'expected_counter_bug_reproduced_in_actual_native_training',
            'actual_iterations_counted': failure['iteration'], 'requested_iterations': 2,
            'process': str(output / 'stages/training_counter/PROCESS.json'),
            'native_failure': str(output / 'pgsr/TRAINING_FAILURE.json'), 'formal_training_affected': False})
        print('expected_counter_bug_reproduced_in_actual_native_training')
        return
    if args.expect_counter_failure or result['iteration'] != 2 or result['target_iterations'] != 2:
        raise ValueError('Diagnostic and training counter separation was not verified')
    atomic_json(output / 'RESULT.json', {'status': 'two_actual_native_iterations_and_diagnostic_counter_isolation_verified',
        'actual_iterations_counted': result['iteration'], 'requested_iterations': 2,
        'native_plane_diagnostic_passed': result['native_plane_kernel_diagnostic']['status'],
        'registered_frames': result['registered_frames'], 'formal_full_training_claimed': False,
        'process': str(output / 'stages/training_counter/PROCESS.json')})
    print('two_actual_native_iterations_and_diagnostic_counter_isolation_verified')


if __name__ == '__main__':
    main()
