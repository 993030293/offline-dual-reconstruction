"""Hand freshly generated motion to the unchanged same-attempt camera frontend."""

import argparse
from pathlib import Path
import shutil

from common import atomic_json, read_json
from sensor_backend import checked_generated_source


def run(root, raw, output):
    root = root.resolve(strict=True)
    output = output.resolve()
    output.relative_to(root)
    backend, result = checked_generated_source(root, raw)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('coarse_body_odometry.npz', 'COARSE_RECEIPT.json', 'CONTRACT.json', 'INITIAL_IMU_DIAGNOSTIC.json'):
        shutil.copyfile(backend / 'coarse' / name, output / name)
    atomic_json(output / 'SAME_ATTEMPT_HANDOFF.json', {'status': 'fresh_motion_handoff_verified',
        'source': str(backend / 'coarse'), 'run_root': str(root),
        'source_sensor_receipt': str(backend / 'RESULT.json'),
        'source_raw': result['raw'], 'cross_attempt_inputs_used': False,
        'coarse_solver_rerun_during_handoff': False, 'runtime_language_model_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.run_root, args.raw, args.output)
