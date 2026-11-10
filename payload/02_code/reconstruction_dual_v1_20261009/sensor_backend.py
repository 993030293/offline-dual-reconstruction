"""Select an actual sensor source before executing either camera or surface code."""

import argparse
from pathlib import Path
import sys

from common import ROOT, atomic_json, read_json

REGISTERED = '/localization/lidar/points_registered_full'
ODOMETRY = '/localization/pose/odom'
RAW_LIDAR = '/sensing/lidar/points_raw'
RAW_IMU = '/sensing/lidar/imu'


def select_backend(counts):
    if counts.get(REGISTERED, 0) and counts.get(ODOMETRY, 0):
        return 'recorded_registered_lidar_and_odometry'
    if counts.get(REGISTERED, 0) or counts.get(ODOMETRY, 0):
        raise ValueError('Partially recorded localization requires an explicit compatible sensor adapter')
    if counts.get(RAW_LIDAR, 0) >= 3 and counts.get(RAW_IMU, 0) >= 100:
        return 'fresh_raw_gyro_translation_local_map'
    raise ValueError('No supported coarse-pose and registered-scan source exists in this recording')


def inspect(raw, output):
    sys.path.insert(0, str(ROOT / '05_envs/e26_rosbags_vendor_v1'))
    from rosbags.highlevel import AnyReader
    raw = raw.resolve(strict=True)
    with AnyReader([raw]) as reader:
        counts = {}
        for connection in reader.connections:
            counts[connection.topic] = counts.get(connection.topic, 0) + connection.msgcount
        backend = select_backend(counts)
        transforms = {}
        if backend == 'fresh_raw_gyro_translation_local_map':
            connections = [row for row in reader.connections if row.topic == '/tf_static']
            for connection, _, payload in reader.messages(connections=connections):
                for row in reader.deserialize(payload, connection.msgtype).transforms:
                    key = (str(row.header.frame_id).lstrip('/'), str(row.child_frame_id).lstrip('/'))
                    t, q = row.transform.translation, row.transform.rotation
                    value = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
                    if key in transforms and transforms[key] != value:
                        raise ValueError('A static sensor transform changed within the raw recording')
                    transforms[key] = value
            if transforms.get(('base_link', 'lidar_cloud')) != [0., 0., 0., 0., 0., 0., 1.]:
                raise ValueError('The retained raw-motion backend requires the recorded identity body/LiDAR rig')
            imu = transforms.get(('lidar_cloud', 'lidar_imu'))
            if imu is None or imu[3:] != [0., 0., 0., 1.]:
                raise ValueError('The retained gyro backend requires aligned recorded LiDAR/IMU axes')
    value = {'status': 'raw_sensor_backend_selected', 'backend': backend, 'raw': str(raw),
        'raw_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
        'topic_message_counts': counts, 'runtime_language_model_used': False,
        'GT_used': False, 'historical_motion_or_geometry_used': False,
        'raw_only_rig_transforms': [{'parent': key[0], 'child': key[1], 'xyz_xyzw': val}
            for key, val in sorted(transforms.items())]}
    atomic_json(output, value)
    return value


def checked_generated_source(root, raw):
    root, raw = Path(root).resolve(strict=True), Path(raw).resolve(strict=True)
    binding = read_json(root / 'SENSOR_BACKEND.json')
    if (binding['backend'] != 'fresh_raw_gyro_translation_local_map'
            or binding['raw'] != str(raw)
            or (binding['raw_bytes'], binding['raw_mtime_ns']) != (raw.stat().st_size, raw.stat().st_mtime_ns)):
        raise ValueError('The generated sensor binding does not describe this frozen original recording')
    backend = root / 'raw_sensor_backend'
    receipt = read_json(backend / 'RESULT.json')
    if (receipt['status'] != 'fresh_raw_registered_scans_complete'
            or Path(receipt['run_root']).resolve(strict=True) != root
            or receipt['raw'] != str(raw)
            or (receipt['raw_bytes'], receipt['raw_mtime_ns']) != (binding['raw_bytes'], binding['raw_mtime_ns'])):
        raise ValueError('Generated scans cannot cross raw inputs or cold-attempt boundaries')
    return backend, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(inspect(args.raw, args.output)['backend'], flush=True)
