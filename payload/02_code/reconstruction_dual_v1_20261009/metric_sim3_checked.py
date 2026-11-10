"""Prevent a similarity-only branch from validating coarse-initialized cameras."""

import importlib.util
from pathlib import Path
import sys

from common import OLD, atomic_json, read_json


def main():
    failed = Path(sys.argv[sys.argv.index('--failed-metric') + 1]).resolve(strict=True)
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    backend = failed.parent / 'camera_backend/RESULT.json'
    if backend.is_file() and read_json(backend)['requires_native_BA_coarse_initialization']:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'RESULT.json', {'status': 'failed_coarse_initialization_requires_actual_metric_BA',
            'source_failed_metric': str(failed), 'BA_failure_preserved': True,
            'reason': 'A similarity transform cannot verify newly coarse-initialized camera orientations or positions',
            'unverified_source_model_not_promoted': True})
        raise ValueError('Coarse-initialized cameras require real native BA; similarity recovery is forbidden')
    spec = importlib.util.spec_from_file_location('preserved_metric_similarity_recovery', OLD / 'metric_sim3_recovery.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.main()


if __name__ == '__main__':
    main()
