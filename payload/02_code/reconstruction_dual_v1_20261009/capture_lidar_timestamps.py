"""Index raw registered-scan times before admitting recovered image frames."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from common import ROOT, atomic_json


def run(raw, output, generated_run_root=None):
    sys.path.insert(0, str(ROOT / '05_envs/e26_rosbags_vendor_v1'))
    from rosbags.highlevel import AnyReader
    source = raw.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if generated_run_root is not None:
        from sensor_backend import checked_generated_source
        backend, receipt = checked_generated_source(generated_run_root, source)
        times = np.load(backend / 'eligible_scan_end_ns.npy', allow_pickle=False)
        if times.dtype != np.int64 or len(times) < 3 or not (np.diff(times) > 0).all():
            raise ValueError('Fresh raw scan evidence has invalid strictly increasing scan-end timestamps')
        np.save(output / 'registered_lidar_header_ns.npy', times, allow_pickle=False)
        atomic_json(output / 'RESULT.json', {'status': 'fresh_raw_registered_scan_times_indexed',
            'raw': str(source), 'raw_bytes': source.stat().st_size, 'raw_mtime_ns': source.stat().st_mtime_ns,
            'topic': receipt['topic'], 'header_frame': receipt['source_frame'],
            'timestamp_kind': 'scan_end_after_native_raw_deskew', 'count': len(times),
            'first_header_ns': int(times[0]), 'last_header_ns': int(times[-1]),
            'maximum_pair_delta_ms': 60., 'timestamps_are_not_ground_truth': True,
            'generated_sensor_receipt': str(backend / 'RESULT.json'),
            'elapsed_seconds': time.monotonic() - started})
        return
    stamps = []
    frames = set()
    with AnyReader([source]) as reader:
        connections = [row for row in reader.connections if row.topic == '/localization/lidar/points_registered_full']
        if len(connections) != 1:
            raise ValueError('The reliable target branch requires one registered LiDAR connection')
        for connection, _, payload in reader.messages(connections=connections):
            message = reader.deserialize(payload, connection.msgtype)
            frames.add(str(message.header.frame_id).lstrip('/'))
            stamps.append(int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec))
    if frames != {'odom'} or len(stamps) < 3:
        raise ValueError('Registered LiDAR timestamps or coordinate frame are invalid')
    times = np.asarray(stamps, np.int64)
    if not (np.diff(times) > 0).all():
        raise ValueError('Registered scan timestamps must be strictly increasing')
    np.save(output / 'registered_lidar_header_ns.npy', times, allow_pickle=False)
    atomic_json(output / 'RESULT.json', {'status': 'raw_registered_scan_times_indexed',
        'raw': str(source), 'raw_bytes': source.stat().st_size, 'raw_mtime_ns': source.stat().st_mtime_ns,
        'topic': '/localization/lidar/points_registered_full', 'header_frame': 'odom',
        'count': len(times), 'first_header_ns': int(times[0]), 'last_header_ns': int(times[-1]),
        'maximum_pair_delta_ms': 60., 'timestamps_are_not_ground_truth': True,
        'elapsed_seconds': time.monotonic() - started})
    print('raw registered scans indexed', len(times), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generated-run-root', type=Path)
    args = parser.parse_args()
    run(args.raw, args.output, args.generated_run_root)
