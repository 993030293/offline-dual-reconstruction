"""Compare adapted raw evidence to preserved failed-profile diagnostic arrays."""

import argparse
from pathlib import Path

import numpy as np

from common import atomic_json, read_json
from sensor_coverage import assess_partial_coverage


def check(target, prior, output):
    receipt = read_json(target / 'stable_lidar/stable_lidar_receipt.json')
    selected = read_json(target / 'registered_image_manifest.json')
    rows = receipt['per_frame']
    if [row['image_name'] for row in rows] != [row['image_name'] for row in selected]:
        raise AssertionError('The adaptive raw evidence changed the ordered visual set')
    failed = read_json(prior / 'stable_lidar/PAIRING_FAILURE.json')
    absent = [row['image_name'] for row in rows if row['lidar_timestamp_ns'] < 0]
    if sorted(absent) != sorted(failed['images']):
        raise AssertionError('The adaptive branch changed the frozen 60 ms pairing rule')
    old_roi, new_roi = read_json(prior / 'vehicle_roi_receipt.json'), read_json(target / 'vehicle_roi_receipt.json')
    if old_roi['roi_world_lio_m'] != new_roi['roi_world_lio_m']:
        raise AssertionError('The adaptive ROI changed relative to the recorded diagnostic')
    checked, independent = 0, {}
    for row in rows:
        if row['lidar_timestamp_ns'] < 0:
            if row['path'] is not None or row['stable_anchor_point_count']:
                raise AssertionError('An unpaired frame acquired synthetic LiDAR')
            continue
        if row['absolute_delta_ms'] > 60.:
            raise AssertionError('A recorded LiDAR match exceeded the unchanged time bound')
        with np.load(target / 'stable_lidar' / row['path']) as archive:
            actual = archive['points_world_lio_m']
        old_path = prior / 'diagnostics/supported_time_subset/stable_lidar/scans' / (Path(row['image_name']).stem + '.npz')
        with np.load(old_path) as archive:
            original = archive['points_world_lio_m']
            if int(archive['lidar_timestamp_ns']) != row['lidar_timestamp_ns']:
                raise AssertionError('An image was moved to a different raw scan')
        if not np.array_equal(actual, original) or len(actual) != row['stable_anchor_point_count']:
            raise AssertionError('Stable point membership changed during independent-scan caching')
        independent[row['lidar_timestamp_ns']] = len(actual)
        checked += 1
    if sum(independent.values()) != receipt['independent_stable_anchor_point_count']:
        raise AssertionError('Independent anchor counts were inflated by repeated visual images')
    reevaluated = [dict(row) for row in rows]
    coverage = assess_partial_coverage(reevaluated, independent)
    if [row['sensor_role'] for row in reevaluated] != [row['sensor_role'] for row in rows]:
        raise AssertionError('Final sensor coverage validation changed any visual or LiDAR role')
    atomic_json(output, {'status': 'actual_visual_set_and_raw_lidar_membership_verified',
        'visual_frames_preserved': len(rows), 'time_paired_arrays_compared_exactly': checked,
        'unpaired_frames_match_preserved_legacy_failure': len(absent),
        'ROI_exactly_unchanged': True, 'independent_scan_count': len(independent),
        'independent_point_count_verified': sum(independent.values()),
        'independent_nonempty_stable_scan_count': coverage['independent_nonempty_stable_scan_count'],
        'latest_coverage_checks_reexecuted_on_exact_raw_arrays': True,
        'prior_profile_remains_failed': True, 'prior_failed_arrays_used_only_as_regression_oracle': True,
        'no_training_or_geometry_GT_claim': True})
    print('actual evidence exact array regression', len(rows), checked, len(absent), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.target_root.resolve(strict=True), args.prior_root.resolve(strict=True), args.output)
