"""Run the established target geometry stages with explicit offline failure records."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback
import psutil

from common import ROOT, HERE, SFM_PYTHON, atomic_json, read_json, execute, linux_command, linux_path
from target_prepare import PACKAGE
from surface_roi_policy import POLICY_VERSION as ROI_TRIM_POLICY_VERSION


def m3_registered_scan_arguments(root, raw):
    """Pass the same full scan source used by anchors, before ground cropping."""
    from sensor_backend import checked_generated_source
    backend = read_json(root / 'SENSOR_BACKEND.json')['backend']
    if backend == 'fresh_raw_gyro_translation_local_map':
        folder, _ = checked_generated_source(root, raw)
        return ['-RegisteredScanReceipt', folder / 'RESULT.json']
    if backend == 'recorded_registered_lidar_and_odometry':
        return []
    raise ValueError('M3 has no compatible bound registered scan source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--profile', choices=('AdaptiveSensor', 'ReliableLidar', 'VisualTargetDiagnostic'), default='AdaptiveSensor')
    parser.add_argument('--phase', choices=('inputs', 'complete'), default='complete')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repair-note')
    args = parser.parse_args()
    root, target = args.run_root.resolve(strict=True), args.output_root.resolve()
    target.relative_to(root)
    runtime = read_json(PACKAGE / 'config/local_runtime.json')
    config = {'run_root': str(root), 'profile': args.profile,
        'shared_input': read_json(root / 'shared/RESULT.json')['private_sparse_binaries'],
        'raw': read_json(root / 'CONFIG.json')['raw'], 'frozen_reference_package': str(PACKAGE),
        'sensor_contract': 'visual_admission_independent_from_partial_lidar_v1'}
    if args.resume:
        if read_json(target / 'TARGET_CONFIG.json') != config:
            raise ValueError('The frozen target input or profile changed')
        state = read_json(target / 'CONTROLLER_STATE.json')
        if state['status'] == 'running' and state.get('controller_pid'):
            try:
                if psutil.Process(state['controller_pid']).create_time() == state['controller_birth']:
                    raise ValueError('The owned target controller is still active')
            except psutil.NoSuchProcess:
                pass
        if state['status'].startswith('failed') and not args.repair_note:
            raise ValueError('A failed target run requires a recorded repair and a new output root')
        if args.repair_note:
            atomic_json(target / ('MANUAL_INTERVENTION_' + str(time.time_ns()) + '.json'),
                {'note': args.repair_note, 'algorithm_autonomous_recovery': False})
    else:
        target.mkdir(parents=True, exist_ok=False)
        atomic_json(target / 'TARGET_CONFIG.json', config)
        atomic_json(target / 'SENSOR_CONTRACT.json', {'profile': args.profile,
            'frozen_before_sensor_evaluation': True,
            'frontend_requires_every_visual_frame_lidar_pair': args.profile == 'ReliableLidar',
            'target_stage_silently_removes_visual_frames': False,
            'maximum_lidar_pair_delta_ms': 60., 'stable_voxel_m': .02, 'minimum_independent_scan_support': 3,
            'minimum_independent_stable_anchors': 10000, 'minimum_contributing_frame_anchors': 10,
            'multi_view_neighbors': 8, 'minimum_neighbor_support': 3,
            'minimum_trusted_visual_fraction': .05, 'minimum_projected_lidar_pixels': 10000,
            'minimum_selected_sparse_support_fraction': .95,
            'surface_candidate_selection': ('explicit_broad_visual_diagnostic' if args.profile == 'VisualTargetDiagnostic'
                else 'surface_candidate_admission_v2'),
            'roi_trim_policy': ROI_TRIM_POLICY_VERSION,
            'unsupported_lidar_behavior': 'explicit_abstain_for_that_frame_keep_visual_evidence',
            'legacy_all_frame_coverage_claimed': args.profile == 'ReliableLidar',
            'independent_geometry_accuracy_claimed': False})
        state = {'status': 'running', 'stages': [], 'started_unix': time.time()}
        sources = []
        for source in sorted((PACKAGE / 'code').rglob('*.py')) + sorted((PACKAGE / 'ground/code').rglob('*.py')):
            relative = source.relative_to(PACKAGE)
            snapshot = target / 'reference_sources' / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            sources.append({'source': str(source), 'snapshot': str(snapshot),
                            'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()})
        atomic_json(target / 'REFERENCE_SOURCE_MANIFEST.json', {'sources': sources, 'captured_before_execution': True})
    for item in read_json(target / 'REFERENCE_SOURCE_MANIFEST.json')['sources']:
        if hashlib.sha256(Path(item['source']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('The bound reference implementation changed')
    os.environ['PYTHONPATH'] = str(HERE / 'offline_guard') + os.pathsep + runtime['rosbags_vendor']
    state.update(controller_pid=psutil.Process().pid, controller_birth=psutil.Process().create_time())

    def stage(label, command, budget, result, gpu=False):
        completed = target / 'stages' / label / 'COMPLETED.json'
        if completed.exists():
            previous = read_json(completed)
            if previous['status'] != 'completed' or not Path(previous['result']).is_file():
                raise ValueError('An incomplete target stage cannot be reused')
            return read_json(previous['result'])
        state.update(status='running', current_stage=label)
        atomic_json(target / 'CONTROLLER_STATE.json', state)
        value = execute(target, label, command, budget, result, gpu)
        state['stages'].append({'stage': label, 'receipt': str(completed)})
        atomic_json(target / 'CONTROLLER_STATE.json', state)
        return value

    try:
        stage('prepare', [SFM_PYTHON, HERE / 'target_prepare.py', '--run-root', root,
            '--output-root', target, '--before-training'], 300, target / 'INPUT_RESULT.json')
        stage('roi', [SFM_PYTHON, HERE / 'derive_target_roi.py', '--metric-sparse',
            target / 'metric_sparse', '--output', target / 'vehicle_roi_receipt.json'],
            300, target / 'vehicle_roi_receipt.json')
        if args.profile != 'VisualTargetDiagnostic':
            stage('anchors', [SFM_PYTHON, HERE / 'target_anchors.py', '--run-root', root,
                '--output-root', target, '--profile', args.profile], 900, target / 'stable_lidar/stable_lidar_receipt.json')
        if args.phase == 'inputs':
            state['status'] = 'target_inputs_complete_training_pending'
            atomic_json(target / 'CONTROLLER_STATE.json', state)
            atomic_json(target / 'INPUT_GATE_RESULT.json', {'status': state['status'],
                'profile': args.profile, 'strict_lidar_inputs_admitted': args.profile == 'ReliableLidar',
                'adaptive_sensor_inputs_admitted': args.profile == 'AdaptiveSensor',
                'input_gate_does_not_require_training_to_have_run': True})
            return
        folder = target / ('07_visual_target_tsdf' if args.profile == 'VisualTargetDiagnostic' else '07_reliable_lidar_tsdf')
        surface = stage('surface', linux_command(HERE / 'target_surface.py', '--run-root', linux_path(root),
            '--output-root', linux_path(target), '--profile', args.profile), 7500,
            folder / 'reliable_lidar_tsdf_receipt.json', gpu=True)
        shell = shutil.which('pwsh')
        if not shell:
            raise FileNotFoundError('The installed PowerShell 7 executable is required; no online installation is permitted')
        m3 = target / '09_ground_target_m3'
        stage('m3', [shell, '-NoProfile', '-File', PACKAGE / 'RUN_FROM_S07_M3.ps1',
            '-RawMesh', surface['mesh']['path'], '-TsdfReceipt', folder / 'reliable_lidar_tsdf_receipt.json',
            '-OutputRoot', m3, '-RawMcap', config['raw'], '-RuntimeConfig', PACKAGE / 'config/local_runtime.json',
            '-Python', runtime['generic_python'],
            *m3_registered_scan_arguments(root, Path(config['raw']))], 900, m3 / 'FINAL_AUDIT.json')
        extraction = read_json(m3 / 'EXTRACTION_RECEIPT.json')
        stage('support', [runtime['generic_python'], PACKAGE / 'code/audit_target_preservation.py',
            '--metric-sparse', target / 'metric_sparse/points3D.txt', '--raw-mesh', surface['mesh']['path'],
            '--target-mesh', extraction['output']['mesh'], '--tsdf-receipt', folder / 'reliable_lidar_tsdf_receipt.json',
            '--output', m3 / 'TARGET_SUPPORT_AUDIT.json'], 300, m3 / 'TARGET_SUPPORT_AUDIT.json')
        state.update(status='target_geometry_complete', surface_profile=args.profile,
            strict_ReliableLidar_profile_completed=surface['strict_ReliableLidar_profile_completed'],
            adaptive_sensor_contract_completed=surface.get('adaptive_sensor_contract_completed', False),
            elapsed_seconds=time.time() - state['started_unix'])
        atomic_json(target / 'CONTROLLER_STATE.json', state)
        atomic_json(target / 'RESULT.json', state)
    except BaseException:
        state.update(status='failed_target_pipeline', traceback=traceback.format_exc(),
                     elapsed_seconds=time.time() - state['started_unix'])
        atomic_json(target / 'CONTROLLER_STATE.json', state)
        atomic_json(target / ('FAILURE_' + str(time.time_ns()) + '.json'), state)
        raise


if __name__ == '__main__':
    main()
