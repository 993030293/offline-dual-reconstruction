"""Apply the same sensor annotations before native authorization replay."""

from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '02_code/gaussian_full_raw_ab_20261008'),
    str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005')]


def main():
    if '--kind' not in sys.argv or '--output' not in sys.argv:
        from pose_gap_support import install
        install()
        import run_native_stage
        run_native_stage.main()
        return
    kind = sys.argv[sys.argv.index('--kind') + 1]
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    backend = output.parent / 'camera_backend'
    if kind == 'metric':
        original = Path(sys.argv[sys.argv.index('--model') + 1]).resolve(strict=True)
        command = [sys.executable, str(Path(__file__).with_name('prepare_camera_backend.py')),
            '--source-model', str(original), '--initial-frames', sys.argv[sys.argv.index('--initial-frames') + 1],
            '--raw', sys.argv[sys.argv.index('--raw') + 1], '--output', str(backend)]
        subprocess.run(command, check=True, timeout=7200, creationflags=subprocess.CREATE_NO_WINDOW)
        prepared = json.loads((backend / 'RESULT.json').read_text())
        sys.argv[sys.argv.index('--model') + 1] = prepared['model']
        if '--offline-physical-recovery' in sys.argv:
            raise ValueError('The production metric adapter already owns the offline recovery option')
        sys.argv.append('--offline-physical-recovery')
        from atomic_json_windows import atomic_json
        atomic_json(output.parent / ('metric_input_adapter_' + str(time.time_ns()) + '.json'),
            {'original_source_model': str(original), 'actual_prepared_source': prepared['model'],
                'backend_receipt': str(backend / 'RESULT.json'),
                'coarse_initialization_requires_native_BA': prepared['requires_native_BA_coarse_initialization'],
                'native_metric_factor_projection_and_admission_guards_unchanged': True})
    elif kind == 'export' and (backend / 'RESULT.json').is_file():
        prepared = json.loads((backend / 'RESULT.json').read_text())
        if prepared['status'] != 'same_run_camera_geometry_prepared_BA_required':
            raise ValueError('The camera backend did not finish its input contract')
        sys.argv[sys.argv.index('--images') + 1] = prepared['images']
    from pose_gap_support import install
    install()
    import run_native_stage
    if kind == 'export':
        from joint_camera_admission import installed
        with installed(run_native_stage):
            run_native_stage.main()
    else:
        run_native_stage.main()


if __name__ == '__main__':
    main()
