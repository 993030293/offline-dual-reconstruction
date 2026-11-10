"""Generate dense scans in the same freshly computed raw-motion world frame."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from common import ROOT, atomic_json, importable_path, read_json

TOOLS = ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/reference_sources/c0_lio_fallback/tools'


def prior_velocities(timestamps, translations):
    """Return the velocity used for each scan, without future translations."""
    dt = np.maximum(np.diff(timestamps).astype(np.float64) / 1e9, .001)
    values = np.zeros_like(translations, dtype=np.float64)
    if len(values) > 2:
        values[2:] = np.diff(translations, axis=0)[:-1] / dt[:-1, None]
    return values


def world_points(points, orientations, point_ns, end_ns, translation, previous_velocity):
    # This is algebraically identical to the retained rotation/translation deskew.
    dt = (point_ns - end_ns).astype(np.float64) / 1e9
    return orientations.apply(points) + translation + dt[:, None] * previous_velocity


def run(root):
    root = root.resolve(strict=True)
    config = read_json(root / 'CONFIG.json')
    binding = read_json(root / 'SENSOR_BACKEND.json')
    raw = Path(config['raw']).resolve(strict=True)
    if binding['backend'] != 'fresh_raw_gyro_translation_local_map' or binding['raw'] != str(raw):
        raise ValueError('Dense raw registration requires this attempt raw-motion sensor binding')
    backend, coarse = root / 'raw_sensor_backend', root / 'raw_sensor_backend/coarse'
    report = read_json(coarse / 'COARSE_RECEIPT.json')
    if report['raw_file'] != str(raw) or report['SfM_used'] or report['GT_used']:
        raise ValueError('The dense scan source must be fresh raw motion, without visual or GT alignment')
    sys.path.insert(0, importable_path(TOOLS))
    from mcap_audit import McapIndex
    from lio_baseline import load_imu, build_orientation_slerp, decode_points, LIDAR_TOPIC
    index = McapIndex(raw)
    imu_ns, gyro, _ = load_imu(index)
    slerp, bias = build_orientation_slerp(imu_ns, gyro, 11.5)
    if not np.allclose(bias, report['gyro_bias_rad_s'], atol=1e-12, rtol=0):
        raise ValueError('The dense scan and coarse-pose gyro integrations differ')
    with np.load(coarse / 'coarse_body_odometry.npz', allow_pickle=False) as motion:
        stamps = motion['timestamp_ns'].copy()
        poses = motion['pose_xyz_xyzw'].copy()
        accepted = motion['accepted'].copy()
        floor_fields = {'gyro_world_correction_xyzw', 'measured_floor_supported', 'measured_floor_motion_accepted'}
        present = floor_fields & set(motion.files)
        if present and present != floor_fields:
            raise ValueError('The raw measured plane correction fields are incomplete')
        correction = motion['gyro_world_correction_xyzw'].copy() if present else None
        floor_supported = motion['measured_floor_supported'].copy() if present else None
        floor_accepted = motion['measured_floor_motion_accepted'].copy() if present else None
    with np.load(coarse / 'coarse_sensor_diagnostics.npz', allow_pickle=False) as diagnostic:
        rmse = diagnostic['rmse'].copy()
        correspondences = diagnostic['correspondences'].copy()
    if (len(stamps) != index.message_count(LIDAR_TOPIC) or len(poses) != len(stamps)
            or poses.shape[1] != 7 or not np.isfinite(poses).all() or not (np.diff(stamps) > 0).all()):
        raise ValueError('Raw scan and freshly generated motion coverage disagree')
    floor_enabled = bool(report.get('measured_floor_constraint_used'))
    if floor_enabled != (correction is not None):
        raise ValueError('The actual raw-motion correction and report disagree')
    if floor_enabled:
        from floor_motion_constraint import PROTOCOL
        if (correction.shape != (len(stamps), 4) or floor_supported.shape != (len(stamps),)
                or floor_accepted.shape != (len(stamps),) or not np.isfinite(correction).all()
                or not np.allclose(np.linalg.norm(correction, axis=1), 1., rtol=0, atol=1e-10)
                or not np.array_equal(floor_accepted, accepted)
                or np.any(floor_accepted & ~floor_supported)
                or report.get('measured_floor_constraint_protocol') != PROTOCOL):
            raise ValueError('The actual raw measured plane corrections or acceptance are invalid')
        floor_receipt = read_json(coarse / 'MEASURED_FLOOR_RECEIPT.json')
        if (floor_receipt['protocol'] != PROTOCOL or floor_receipt['actual_scan_count'] != len(stamps)
                or [row['timestamp_ns'] for row in floor_receipt['per_scan']] != list(stamps)
                or [bool(row['motion_accepted']) for row in floor_receipt['per_scan']] != list(accepted)):
            raise ValueError('The actual plane measurements differ from the raw-motion clocks')
    eligible = accepted & np.isfinite(rmse) & (rmse <= .45 * .75) & (correspondences >= 100)
    if np.count_nonzero(eligible) < 3:
        raise ValueError('Too few raw scans have usable measured local-map registration support')
    velocity = prior_velocities(stamps, poses[:, :3])
    scan_dir = backend / 'scans'
    scan_dir.mkdir(parents=True, exist_ok=False)
    started, rows = time.monotonic(), []
    for number, message in enumerate(index.iter_messages(LIDAR_TOPIC)):
        if number >= len(stamps):
            raise ValueError('The raw scan stream grew after coarse-motion generation')
        header, points, offsets = decode_points(message['payload'])
        end = int(stamps[number])
        if not (header <= end <= min(header + int(offsets.max()), int(imu_ns[-1]))):
            raise ValueError('The generated scan end does not belong to its original scan clock interval')
        end_rotation = slerp([end / 1e9])[0]
        world_correction = Rotation.identity() if correction is None else Rotation.from_quat(correction[number])
        end_rotation = world_correction * end_rotation
        if (Rotation.from_quat(poses[number, 3:]).inv() * end_rotation).magnitude() > 1e-9:
            raise ValueError('Dense registration is in a different orientation frame from the coarse camera poses')
        path = None
        if eligible[number]:
            point_ns = np.clip(header + offsets, imu_ns[0], imu_ns[-1])
            xyz = world_points(points, world_correction * slerp(point_ns.astype(np.float64) / 1e9), point_ns,
                end, poses[number, :3], velocity[number])
            if not np.isfinite(xyz).all():
                raise ValueError('Freshly registered points contain nonfinite world coordinates')
            target = scan_dir / ('scan_' + str(end) + '.npz')
            np.savez_compressed(target, points_world_lio_m=xyz.astype(np.float32), lidar_timestamp_ns=end)
            path = target.relative_to(backend).as_posix()
        rows.append({'index': number, 'raw_header_ns': header, 'scan_end_ns': end,
            'accepted_native_step': bool(accepted[number]), 'eligible_registered_evidence': bool(eligible[number]),
            'native_rmse_m': float(rmse[number]) if np.isfinite(rmse[number]) else None,
            'native_correspondences': int(correspondences[number]), 'valid_raw_points': len(points), 'path': path})
        if len(rows) % 25 == 0:
            state = {'status': 'registering_fresh_raw_scans', 'scans_read': len(rows),
                'total_scans': len(stamps), 'elapsed_seconds': time.monotonic() - started}
            atomic_json(backend / 'STATE.json', state)
            print(state, flush=True)
    if len(rows) != len(stamps):
        raise ValueError('The dense raw scan stream was incomplete')
    if (raw.stat().st_size, raw.stat().st_mtime_ns) != (config['raw_bytes'], config['raw_mtime_ns']):
        raise ValueError('The original recording changed during raw registration')
    np.save(backend / 'eligible_scan_end_ns.npy', stamps[eligible], allow_pickle=False)
    result = {'status': 'fresh_raw_registered_scans_complete', 'run_root': str(root), 'raw': str(raw),
        'raw_bytes': config['raw_bytes'], 'raw_mtime_ns': config['raw_mtime_ns'], 'topic': LIDAR_TOPIC,
        'source_frame': 'fresh_local_gyro_translation_lio_world', 'raw_scans_read': len(rows),
        'eligible_scans': int(eligible.sum()), 'rejected_evidence_scans': int((~eligible).sum()), 'scans': rows,
        'coarse_receipt': str(coarse / 'COARSE_RECEIPT.json'), 'official_Point_LIO': False,
        'SfM_used_for_scan_registration': False, 'GT_used': False, 'runtime_language_model_used': False,
        'initial_static_hold_seconds': 11.5, 'initial_static_hold_independently_confirmed': False,
        'dense_registration': ('saved_measured_plane_gyro_correction_and_causal_previous_translation_velocity'
            if floor_enabled else 'native_gyro_deskew_and_causal_previous_translation_velocity'),
        'measured_floor_constraint_used': floor_enabled,
        'saved_per_scan_rotation_correction_applied_to_all_raw_points': floor_enabled,
        'actual_coarse_and_dense_orientation_check_retained': True,
        'eligible_times': str(backend / 'eligible_scan_end_ns.npy'), 'historical_derived_inputs_used': False,
        'independent_geometry_accuracy_verified': False, 'elapsed_seconds': time.monotonic() - started}
    atomic_json(backend / 'RESULT.json', result)
    atomic_json(backend / 'STATE.json', {key: value for key, value in result.items() if key != 'scans'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    args = parser.parse_args()
    run(args.run_root)
