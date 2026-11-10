"""Keep visual admission separate from genuine independent LiDAR evidence."""

def assess_partial_coverage(rows, unique_stable_point_counts):
    if len(rows) < 3:
        raise ValueError('Too few visual frames for the adaptive sensor contract')
    names = [row['image_name'] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError('Visual frame identities must be unique')
    if any(int(value) < 0 for value in unique_stable_point_counts.values()):
        raise ValueError('Independent scan evidence cannot have negative counts')
    supported_scans = sum(int(value) > 0 for value in unique_stable_point_counts.values())
    total = sum(int(value) for value in unique_stable_point_counts.values())
    if supported_scans < 3 or total < 10000:
        raise ValueError('Insufficient independent stable LiDAR evidence for sensor fusion')
    unpaired, weak, contributing = [], [], []
    for row in rows:
        if row['lidar_timestamp_ns'] < 0:
            if row['path'] is not None or row['stable_anchor_point_count']:
                raise ValueError('An unpaired frame cannot contain borrowed LiDAR evidence')
            row['sensor_role'] = 'visual_only_no_synchronized_lidar'
            unpaired.append(row['image_name'])
        elif row['stable_anchor_point_count'] < 10:
            row['sensor_role'] = 'visual_only_lidar_abstained_below_10_anchors'
            weak.append(row['image_name'])
        else:
            if row['absolute_delta_ms'] > 60. or row['path'] is None:
                raise ValueError('A contributing frame violates the LiDAR pairing contract')
            row['sensor_role'] = 'visual_and_stable_lidar'
            contributing.append(row['image_name'])
    if not contributing:
        raise ValueError('No frame can contribute genuine stable LiDAR')
    return {'status': 'partial_sensor_coverage_visual_set_preserved',
        'visual_frame_count': len(rows), 'lidar_contributing_frame_count': len(contributing),
        'time_unpaired_frame_count': len(unpaired), 'paired_weak_anchor_frame_count': len(weak),
        'time_unpaired_frames': unpaired, 'paired_weak_anchor_frames': weak,
        'independent_scan_count': len(unique_stable_point_counts),
        'independent_nonempty_stable_scan_count': supported_scans,
        'independent_stable_anchor_point_count': total, 'minimum_total_independent_anchors': 10000,
        'minimum_contributing_frame_anchors': 10, 'maximum_pair_delta_ms': 60.,
        'visual_frames_removed_for_lidar_coverage': 0,
        'missing_lidar_fabricated_or_extrapolated': False,
        'legacy_all_frame_coverage_gate_claimed': False}
