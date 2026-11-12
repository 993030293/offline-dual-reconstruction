"""Run raw-to-dual delivery with the established target branch as the default."""

import argparse
import os
from pathlib import Path
import re
import time
import traceback
import psutil

from common import ROOT, HERE, OLD, SFM_PYTHON, atomic_json, read_json, execute, linux_command, linux_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--case', default='box1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repair-note')
    parser.add_argument('--retry-front-stage', choices=('initial', 'sfm', 'recovery', 'temporal', 'pose_gap',
        'metric', 'metric_sim3_renderer_recovery', 'roles', 'export'))
    parser.add_argument('--body-camera-calibration', type=Path,
        help='Explicit local camera_optical to base_link calibration when absent from raw TF')
    parser.add_argument('--stop-after', choices=('sensors', 'front', 'complete'), default='complete')
    parser.add_argument('--point-profile', choices=('AdaptiveSensor', 'ReliableLidar', 'VisualTargetDiagnostic'), default='AdaptiveSensor')
    parser.add_argument('--gaussian-iterations', type=int, default=12000)
    parser.add_argument('--pgsr-iterations', type=int, default=15000)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', args.case):
        raise ValueError('A safe lowercase case identifier is required')
    root, raw = args.run_root.resolve(), args.raw.resolve(strict=True)
    if args.retry_front_stage and (not args.resume or not args.repair_note):
        raise ValueError('An explicit front retry requires resume and a recorded repair note')
    config = {'schema': 3, 'case': args.case, 'raw': str(raw), 'raw_bytes': raw.stat().st_size,
        'raw_mtime_ns': raw.stat().st_mtime_ns, 'gaussian_iterations': args.gaussian_iterations,
        'pgsr_iterations': args.pgsr_iterations, 'point_profile': args.point_profile,
        'point_scope': 'target_roi_after_M3', 'initial_solver': 'COLMAP_native_global_CPU',
        'target_reference_package': 'e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate',
        'surface_voxel_m': .005, 'surface_truncation_m': .020, 'surface_depth_max_m': 10.,
        'surface_sample_count': 1000000, 'GT_used': False, 'historical_derived_inputs_used': False,
        'visual_frame_admission': 'RequireEveryVisualFrame' if args.point_profile == 'ReliableLidar' else 'PreserveVisualFrames'}
    if args.body_camera_calibration:
        from calibration_binding import load_calibration
        config['explicit_body_camera_calibration'] = load_calibration(args.body_camera_calibration)
    if args.resume:
        if read_json(root / 'CONFIG.json') != config:
            raise ValueError('A frozen input changed; legacy runs cannot be resumed as a new sensor contract')
        state = read_json(root / 'STATE.json')
        if state['status'] == 'running' and state.get('pid'):
            try:
                if psutil.Process(state['pid']).create_time() == state['process_birth']:
                    raise ValueError('The owned delivery controller is still active')
            except psutil.NoSuchProcess:
                pass
        if state['status'].startswith('failed'):
            if not args.retry_front_stage:
                raise ValueError('A failed run is retained. Use a new root or an explicit recorded front-stage retry.')
            from manual_retry import archive_failed_front
            archive_failed_front(root, args.case, args.retry_front_stage, args.repair_note)
            invalidated = {'front', 'shared', 'target_inputs', 'gaussian', 'pgsr', 'target_geometry', 'delivery'}
            state['stages'] = [row for row in state.get('stages', []) if row['stage'] not in invalidated]
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / 'CONFIG.json', config)
        atomic_json(root / 'CASES.json', {'cases': [{'case': args.case, 'raw': str(raw), 'output': str(root / args.case)}]})
        state = {'status': 'running', 'stages': [], 'first_started_unix': time.time()}
    if args.body_camera_calibration:
        from calibration_binding import freeze_binding
        freeze_binding(root, args.case, config['explicit_body_camera_calibration'], args.resume)
    if args.repair_note:
        atomic_json(root / ('MANUAL_INTERVENTION_' + str(time.time_ns()) + '.json'),
            {'note': args.repair_note, 'algorithm_autonomous_recovery': False})
    side, target = root / args.case / 'optimized', root / 'target'
    state.update(status='running', pid=psutil.Process().pid, process_birth=psutil.Process().create_time())
    if 'traceback' in state:
        state['historical_controller_traceback'] = state.pop('traceback')
    state['controller_attempt_count'] = state.get('controller_attempt_count', 0) + 1
    atomic_json(root / 'OFFLINE_CONTRACT.json', {'status': 'local_runtime_only',
        'runtime_language_model_required': False, 'runtime_downloads_allowed': False,
        'Python_network_policy': 'loopback_only', 'viewer_binding': '127.0.0.1',
        'development_assistant_repairs_are_not_algorithm_recovery': True})

    def stage(label, command, budget, result, gpu=False):
        completed = root / 'stages' / label / 'COMPLETED.json'
        if completed.exists():
            receipt = read_json(completed)
            if receipt['status'] != 'completed' or not Path(receipt['result']).is_file():
                raise ValueError('An incomplete stage cannot be reused')
            return read_json(receipt['result'])
        state.update(current_stage=label)
        atomic_json(root / 'STATE.json', state)
        value = execute(root, label, command, budget, result, gpu)
        state['stages'].append({'stage': label, 'receipt': str(completed)})
        atomic_json(root / 'STATE.json', state)
        return value

    try:
        stage('preflight', [SFM_PYTHON, HERE / 'preflight.py', '--run-root', root,
            '--output', root / 'PREFLIGHT.json'], 120, root / 'PREFLIGHT.json')
        sensor = stage('sensor_backend', [SFM_PYTHON, HERE / 'sensor_backend.py', '--raw', raw,
            '--output', root / 'SENSOR_BACKEND.json'], 120, root / 'SENSOR_BACKEND.json')
        os.environ['RECON_LIDAR_FRAME_ADMISSION'] = config['visual_frame_admission']
        if sensor['backend'] == 'fresh_raw_gyro_translation_local_map':
            generated = root / 'raw_sensor_backend'
            stage('raw_motion', [SFM_PYTHON, OLD / 'run_coarse_c0.py', '--raw', raw,
                '--output', generated / 'coarse', '--measured-floor-constraint'],
                3600, generated / 'coarse/COARSE_RECEIPT.json')
            stage('raw_registration', [SFM_PYTHON, HERE / 'register_raw_lidar.py', '--run-root', root],
                900, generated / 'RESULT.json')
            os.environ['RECON_FRESH_RAW_SENSOR_RUN_ROOT'] = str(root)
        elif 'RECON_FRESH_RAW_SENSOR_RUN_ROOT' in os.environ:
            raise ValueError('A recorded localization run must not inherit a generated-motion source')
        if args.point_profile != 'VisualTargetDiagnostic':
            time_command = [SFM_PYTHON, HERE / 'capture_lidar_timestamps.py', '--raw', raw,
                '--output', root / 'raw_time_index']
            if sensor['backend'] == 'fresh_raw_gyro_translation_local_map':
                time_command += ['--generated-run-root', root]
            stage('raw_times', time_command, 300, root / 'raw_time_index/RESULT.json')
            os.environ['RECON_REGISTERED_LIDAR_TIMESTAMPS'] = str(root / 'raw_time_index/registered_lidar_header_ns.npy')
        elif 'RECON_REGISTERED_LIDAR_TIMESTAMPS' in os.environ:
            raise ValueError('The explicit visual profile must not inherit a LiDAR admission index')
        if args.stop_after == 'sensors':
            state['status'] = 'sensor_preflight_complete_reconstruction_pending'
            atomic_json(root / 'STATE.json', state)
            return
        stage('front', [SFM_PYTHON, HERE / 'front_end.py', '--case', args.case, '--mode', 'optimized',
            '--campaign', root, '--stop-after', 'export', '--render-near-recovery'], 14400,
            side / 'export/TRAINING_ADMISSION.json')
        if args.stop_after == 'front':
            state['status'] = 'front_complete_target_and_training_pending'
            atomic_json(root / 'STATE.json', state)
            return
        stage('shared', [SFM_PYTHON, HERE / 'prepare_shared.py', '--run-root', root], 900, root / 'shared/RESULT.json')
        stage('target_inputs', [SFM_PYTHON, HERE / 'run_target_pipeline.py', '--run-root', root,
            '--output-root', target, '--profile', args.point_profile, '--phase', 'inputs'], 1800,
            target / 'INPUT_GATE_RESULT.json')
        count = read_json(root / 'shared/RESULT.json')['registered_frames']
        stage('gaussian', linux_command(HERE / 'gaussian_branch.py', '--mode', 'optimized', '--runtime',
            linux_path(ROOT / '04_outputs/box2_3dgs_display_v1_20260930'), '--dataset', linux_path(root / 'shared/gaussian'),
            '--output', linux_path(side / 'training'), '--projection', 'full', '--seed', '0',
            '--iterations', str(args.gaussian_iterations), '--maximum-seconds', '2400', '--input-version', root.name,
            '--expected-image-count', str(count), '--gsplat-build',
            linux_path(ROOT / '04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001'),
            '--vectorized-ply', '--verify-native-ply'), 3000, side / 'training/STATE.json', True)
        stage('pgsr', linux_command(HERE / 'pgsr_branch.py', 'train', linux_path(root)), 7500,
            root / 'pgsr/TRAINING_RESULT.json', True)
        stage('target_geometry', [SFM_PYTHON, HERE / 'run_target_pipeline.py', '--run-root', root,
            '--output-root', target, '--profile', args.point_profile, '--phase', 'complete', '--resume'], 9000,
            target / 'RESULT.json')
        stage('delivery', [SFM_PYTHON, HERE / 'package_delivery.py', '--run-root', root,
            '--target-root', target], 900, target / 'DELIVERY_RESULT.json')
        result = read_json(target / 'DELIVERY_RESULT.json')
        if (raw.stat().st_size, raw.stat().st_mtime_ns) != (config['raw_bytes'], config['raw_mtime_ns']):
            raise ValueError('The original recording changed')
        state.update(status='dual_target_outputs_complete' if (result['strict_ReliableLidar_profile_completed']
            or result.get('adaptive_sensor_contract_completed', False))
            else 'visual_target_diagnostic_complete_strict_lidar_not_claimed',
            delivery_result=str(target / 'DELIVERY_RESULT.json'), independent_geometry_accuracy_verified=False,
            wall_seconds=time.time() - state['first_started_unix'])
        atomic_json(root / 'STATE.json', state)
        atomic_json(root / 'RESULT.json', state)
        from audit_target_run import review
        review(root)
    except BaseException:
        state.update(status='failed_delivery', traceback=traceback.format_exc(),
                     wall_seconds=time.time() - state['first_started_unix'])
        atomic_json(root / 'STATE.json', state)
        atomic_json(root / ('FAILURE_' + str(time.time_ns()) + '.json'), state)
        from audit_target_run import review
        review(root)
        raise


if __name__ == '__main__':
    main()
