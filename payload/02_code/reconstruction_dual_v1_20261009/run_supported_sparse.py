"""Bind visual recovery annotations and the accepted sparse graph in its child."""

import importlib
import importlib.util
import sys

from pathlib import Path

from common import ROOT, atomic_json, importable_path

sys.path.insert(0, str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005'))
from pose_gap_support import install


def bind_reference_camera_tools():
    """Bind the preserved camera decoder through a Windows extended path."""
    source = ROOT / '02_code/box2_initial_pose_quality_v2_20261002'
    source_value = str(source)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    baseline = importlib.import_module('initial_selection')
    tools = Path(importable_path(baseline.TOOLS))
    required = [tools / name for name in ('camera_motion.py', 'mcap_audit.py', 'color_contract.py')]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('The sparse augmentation camera support tools are unavailable')
    baseline.TOOLS = tools
    tools_value = str(tools)
    if tools_value not in sys.path:
        sys.path.insert(0, tools_value)
    camera_motion = importlib.import_module('camera_motion')
    importlib.import_module('mcap_audit')
    importlib.import_module('color_contract')
    return camera_motion


def main():
    flag = '--recovery-input-only'
    if sys.argv.count(flag) > 1:
        raise ValueError('A recovery input scope must be declared only once')
    recovery_input = flag in sys.argv
    if recovery_input:
        sys.argv.remove(flag)
        if ('--purpose' not in sys.argv or sys.argv[sys.argv.index('--purpose') + 1] != 'registration'
                or '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode') + 1] != 'optimized'):
            raise ValueError('A partial temporal input cannot replace pose-gap or baseline registration checks')
    install()
    bind_reference_camera_tools()
    path = ROOT / '02_code/gaussian_box1_time20_v1_20261009/run_native_sparse.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('supported_sparse_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.validate_registration_guards
    if recovery_input:
        from temporal_recovery_input import verify
        initial = Path(sys.argv[sys.argv.index('--initial') + 1])
        recovery = Path(sys.argv[sys.argv.index('--recovery') + 1])
        output = Path(sys.argv[sys.argv.index('--output') + 1])

        def validate(result):
            value = verify(initial, recovery, output, result)
            atomic_json(output / 'TEMPORAL_RECOVERY_INPUT.json', value)
            return original(result) if value['original_registration_guard_passed'] else value['scope']

        module.validate_registration_guards = validate
    try:
        module.main()
    finally:
        module.validate_registration_guards = original


if __name__ == '__main__':
    main()
