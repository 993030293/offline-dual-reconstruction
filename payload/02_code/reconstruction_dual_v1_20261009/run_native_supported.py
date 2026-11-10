"""Apply the same sensor annotations before native authorization replay."""

from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '02_code/gaussian_full_raw_ab_20261008'),
    str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005')]


def metric_retriangulation_inputs(backend, prepared):
    direct_database = prepared.get('retriangulation_database')
    direct_images = prepared.get('retriangulation_images')
    candidates = []
    if direct_database and direct_images:
        candidates.append((Path(direct_database), Path(direct_images), backend / 'RESULT.json'))
    for name in ('all_point_rebuilding', 'native_visual_recovery', 'raw_quality_bridge', 'missing_matches'):
        receipt = backend / name / 'RESULT.json'
        if not receipt.is_file():
            continue
        value = json.loads(receipt.read_text())
        if value.get('database') and value.get('images'):
            candidates.append((Path(value['database']), Path(value['images']), receipt))
    for database, images, receipt in candidates:
        if database.is_file() and images.is_dir():
            return database.resolve(), images.resolve(), receipt.resolve()
    return None


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
        if ('--offline-physical-recovery' in sys.argv
                or '--offline-metric-retriangulation' in sys.argv):
            raise ValueError('The production metric adapter already owns the offline recovery option')
        recovery_inputs = metric_retriangulation_inputs(backend, prepared)
        if recovery_inputs:
            database, images, recovery_receipt = recovery_inputs
            sys.argv.extend(['--offline-metric-retriangulation',
                '--retriangulation-database', str(database),
                '--retriangulation-images', str(images)])
            recovery_mode = 'coarse_center_retriangulation_joint_BA'
        else:
            sys.argv.append('--offline-physical-recovery')
            database, images, recovery_receipt = None, None, None
            recovery_mode = 'legacy_same_track_physical_recovery_no_database_available'
        from atomic_json_windows import atomic_json
        atomic_json(output.parent / ('metric_input_adapter_' + str(time.time_ns()) + '.json'),
            {'original_source_model': str(original), 'actual_prepared_source': prepared['model'],
                'backend_receipt': str(backend / 'RESULT.json'),
                'coarse_initialization_requires_native_BA': prepared['requires_native_BA_coarse_initialization'],
                'offline_recovery_mode': recovery_mode,
                'retriangulation_database': str(database) if database else None,
                'retriangulation_images': str(images) if images else None,
                'retriangulation_source_receipt': str(recovery_receipt) if recovery_receipt else None,
                'original_metric_candidate_failure_preserved': True,
                'downstream_point_filter_thresholds_unchanged': True})
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
