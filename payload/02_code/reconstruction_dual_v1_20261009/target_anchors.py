"""Pair admitted images with genuine scans while preserving visual-only frames."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from common import ROOT, atomic_json, read_json
from package_delivery import identity
from target_prepare import PACKAGE
from sensor_coverage import assess_partial_coverage

sys.path.insert(0, str(PACKAGE / 'code'))
sys.path.insert(0, str(ROOT / '05_envs/e26_rosbags_vendor_v1'))
from extract_stable_registered_lidar import TOPIC, decode_xyz, timestamp_ns, assess_anchor_coverage


def improved_images(times, stamp, best_delta, window_ns):
    # Dense images may share a scan, but each image still chooses its nearest raw scan.
    first = np.searchsorted(times, stamp - window_ns, side='left')
    end = np.searchsorted(times, stamp + window_ns, side='right')
    indices = np.arange(first, end)
    delta = np.abs(times[indices] - stamp)
    return indices[delta < best_delta[indices]]


def supported_voxels(scans, voxel_m=.02, minimum=3):
    rows = []
    for index, stamp in enumerate(sorted(scans)):
        voxels = np.unique(np.floor(scans[stamp] / voxel_m).astype(np.int32), axis=0)
        rows.append(np.column_stack((voxels, np.full(len(voxels), index, np.int32))))
    if not rows or not sum(len(row) for row in rows):
        raise ValueError('No raw registered LiDAR exists inside the derived ROI')
    pairs = np.unique(np.concatenate(rows), axis=0)
    voxels, count = np.unique(pairs[:, :3], axis=0, return_counts=True)
    return voxels[count >= minimum]


def scan_source(root, raw):
    """Yield recorded scans or scans generated in this very cold attempt."""
    from rosbags.highlevel import AnyReader
    binding_path = root / 'SENSOR_BACKEND.json'
    if (binding_path.is_file()
            and read_json(binding_path)['backend'] == 'fresh_raw_gyro_translation_local_map'):
        from sensor_backend import checked_generated_source
        backend, receipt = checked_generated_source(root, raw)
        for row in receipt['scans']:
            loader = None
            if row['eligible_registered_evidence']:
                path = (backend / row['path']).resolve(strict=True)
                path.relative_to(backend / 'scans')

                def load(path=path, stamp=row['scan_end_ns']):
                    with np.load(path, allow_pickle=False) as data:
                        if int(data['lidar_timestamp_ns']) != stamp:
                            raise ValueError('A generated scan file has a different source clock')
                        return data['points_world_lio_m'].copy()

                loader = load
            yield row['scan_end_ns'], loader
        return
    with AnyReader([raw]) as reader:
        connections = [row for row in reader.connections if row.topic == TOPIC]
        if len(connections) != 1:
            raise ValueError('Exactly one recorded registered LiDAR connection is required')
        for connection, _, payload in reader.messages(connections=connections):
            message = reader.deserialize(payload, connection.msgtype)
            if str(message.header.frame_id).lstrip('/') != 'odom':
                raise ValueError('Recorded registered LiDAR is not in the common odom frame')
            yield timestamp_ns(message.header.stamp), lambda message=message: decode_xyz(message)


def run(root, output, profile='AdaptiveSensor'):
    from rosbags.highlevel import AnyReader
    config = read_json(root / 'CONFIG.json')
    raw = Path(config['raw']).resolve(strict=True)
    if (raw.stat().st_size, raw.stat().st_mtime_ns) != (config['raw_bytes'], config['raw_mtime_ns']):
        raise ValueError('The raw input changed')
    manifest_path = output / 'registered_image_manifest.json'
    roi_path = output / 'vehicle_roi_receipt.json'
    manifest, roi = read_json(manifest_path), read_json(roi_path)
    if roi['status'] != 'complete':
        raise ValueError('Derived target ROI is incomplete')
    times = np.array([row['camera_header_ns'] for row in manifest], np.int64)
    if not (np.diff(times) > 0).all():
        raise ValueError('Image header timestamps must be unique and sorted')
    minimum, maximum = [np.asarray(roi['roi_world_lio_m'][key]) for key in ('minimum', 'maximum')]
    ground_z = float(roi['roi_derivation']['metric_sparse_vehicle_support']['ground_z_m'])
    anchors = output / 'stable_lidar'
    (anchors / 'scans').mkdir(parents=True, exist_ok=False)
    delta = np.full(len(times), np.iinfo(np.int64).max, np.int64)
    stamps = np.full(len(times), -1, np.int64)
    points = [None] * len(times)
    started = time.monotonic()
    scan_count = 0
    for stamp, loader in scan_source(root, raw):
        indices = improved_images(times, stamp, delta, 60_000_000) if loader else []
        if len(indices):
            xyz = loader()
            valid = (np.isfinite(xyz).all(1) & (xyz >= minimum).all(1)
                     & (xyz <= maximum).all(1) & (xyz[:, 2] > ground_z + .03))
            cropped = np.ascontiguousarray(xyz[valid])
            for index in indices:
                delta[index], stamps[index], points[index] = abs(stamp - times[index]), stamp, cropped
        scan_count += 1
        if scan_count % 100 == 0:
            atomic_json(anchors / 'STATE.json', {'status': 'pairing_raw_scans',
                'raw_scans_read': scan_count, 'images_paired': int((stamps >= 0).sum()),
                'elapsed_seconds': time.monotonic() - started})
            print('raw LiDAR scans', scan_count, 'images paired', int((stamps >= 0).sum()), flush=True)
    if profile == 'ReliableLidar' and ((stamps < 0).any() or (delta > 60_000_000).any()):
        failed = [manifest[i]['image_name'] for i in np.flatnonzero((stamps < 0) | (delta > 60_000_000))]
        atomic_json(anchors / 'PAIRING_FAILURE.json', {'status': 'failed_pairing', 'images': failed})
        raise ValueError('At least one image has no raw registered scan within 60 ms')
    unique = {int(stamp): xyz for stamp, xyz in zip(stamps, points) if stamp >= 0}
    keep = supported_voxels(unique)
    dtype = np.dtype([(name, '<i4') for name in ('x', 'y', 'z')])
    keep_keys = np.ascontiguousarray(keep).view(dtype).reshape(-1)
    retained_by_scan, paths = {}, {}
    for stamp, xyz in unique.items():
        voxel = np.ascontiguousarray(np.floor(xyz / .02).astype(np.int32))
        retained = np.ascontiguousarray(xyz[np.isin(voxel.view(dtype).reshape(-1), keep_keys)])
        path = anchors / 'scans' / ('scan_' + str(stamp) + '.npz')
        np.savez_compressed(path, points_world_lio_m=retained,
                            lidar_timestamp_ns=stamp)
        retained_by_scan[stamp], paths[stamp] = retained, path.relative_to(anchors).as_posix()
    rows = []
    for index, (row, xyz, stamp) in enumerate(zip(manifest, points, stamps)):
        paired = stamp >= 0
        rows.append({'image_name': row['image_name'], 'image_timestamp_ns': int(times[index]),
            'lidar_timestamp_ns': int(stamp), 'absolute_delta_ms': float(delta[index] / 1e6) if paired else None,
            'cropped_non_ground_point_count': len(xyz) if paired else 0,
            'stable_anchor_point_count': len(retained_by_scan[int(stamp)]) if paired else 0,
            'path': paths[int(stamp)] if paired else None})
    try:
        coverage = (assess_anchor_coverage(rows) if profile == 'ReliableLidar' else
            assess_partial_coverage(rows, {stamp: len(xyz) for stamp, xyz in retained_by_scan.items()}))
    except (RuntimeError, ValueError) as error:
        atomic_json(anchors / 'COVERAGE_FAILURE.json', {'status': 'failed_anchor_coverage',
            'error': str(error), 'per_frame': rows, 'elapsed_seconds': time.monotonic() - started,
            'profile': profile, 'failed_outputs_are_not_admitted': True})
        raise
    paired_delta = delta[stamps >= 0]
    sensor_path = root / 'SENSOR_BACKEND.json'
    generated = sensor_path.is_file() and read_json(sensor_path)['backend'] == 'fresh_raw_gyro_translation_local_map'
    result = {'schema_version': 'e26_stable_registered_lidar_anchors_v3', 'status': 'complete', 'profile': profile,
        'source_mcap': str(raw), 'topic': '/sensing/lidar/points_raw' if generated else TOPIC,
        'source_frame': 'fresh_local_gyro_translation_lio_world' if generated else 'odom',
        'sensor_source': 'same_attempt_raw_motion_and_deskew' if generated else 'recorded_registered_scans',
        'generated_sensor_receipt': str(root / 'raw_sensor_backend/RESULT.json') if generated else None,
        'target_geometry_frame': 'world_lio_same_as_fresh_coarse_metric_context',
        'image_manifest': str(manifest_path), 'image_manifest_sha256': identity(manifest_path)['sha256'],
        'roi_receipt': str(roi_path), 'image_count': len(rows), 'per_frame': rows,
        'raw_scans_read': scan_count, 'unique_lidar_scan_count': len(unique),
        'pair_delta_ms': {'p50': float(np.median(paired_delta) / 1e6), 'p95': float(np.quantile(paired_delta, .95) / 1e6),
                          'maximum': float(paired_delta.max() / 1e6), 'denominator': 'time_paired_images'},
        'voxel_size_m': .02, 'minimum_scan_support': 3, 'minimum_ground_clearance_m': .03,
        'stable_voxel_count': len(keep), 'stable_anchor_point_count': sum(row['stable_anchor_point_count'] for row in rows),
        'anchor_coverage': coverage, 'dense_frame_pairing': 'nearest_scan_per_image_within_frozen_60ms',
        'independent_stable_anchor_point_count': sum(len(xyz) for xyz in retained_by_scan.values()),
        'stable_membership_and_npz_computed_once_per_independent_scan': True,
        'duplicate_images_count_as_independent_scans': False, 'ground_z_m': ground_z,
        'language_model_used': False, 'semantic_model_used': False, 'manual_selection_used': False,
        'elapsed_seconds': time.monotonic() - started}
    atomic_json(anchors / 'stable_lidar_receipt.json', result)
    atomic_json(anchors / 'STATE.json', {key: value for key, value in result.items() if key != 'per_frame'})
    print('raw stable anchors complete', len(rows), len(unique), result['stable_anchor_point_count'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--profile', choices=('AdaptiveSensor', 'ReliableLidar'), default='AdaptiveSensor')
    args = parser.parse_args()
    run(args.run_root.resolve(strict=True), args.output_root.resolve(strict=True), args.profile)
