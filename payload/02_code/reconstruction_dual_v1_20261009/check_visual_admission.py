"""Check actual optimized additions and partial sensor abstention independently."""

import argparse
import json
import os
from pathlib import Path
import sys

from common import ROOT, atomic_json, read_json
from pose_gap_support import bind
from sensor_coverage import assess_partial_coverage

sys.path.insert(0, str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005'))
from pose_gap_candidates import select_pose_gap_candidates


def check(run, time_index, output):
    config = read_json(run / 'CONFIG.json')
    initial = run / config['case'] / 'optimized/initial'
    rows = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
    manifest = read_json(initial / 'INITIAL_FRAME_MANIFEST.json')
    original, decisions = select_pose_gap_candidates(rows, manifest['policy'], manifest['receipt']['sharpness_min'])
    os.environ['RECON_REGISTERED_LIDAR_TIMESTAMPS'] = str(time_index)
    os.environ['RECON_LIDAR_FRAME_ADMISSION'] = 'PreserveVisualFrames'
    actual, annotated = bind(select_pose_gap_candidates)(rows, manifest['policy'], manifest['receipt']['sharpness_min'])
    if actual != original:
        raise AssertionError('Partial LiDAR availability changed valid visual additions or their order')
    for before, after in zip(decisions, annotated):
        if any(after[key] != value for key, value in before.items()):
            raise AssertionError('The optimized quality or pose recovery decisions changed')
    absent = [row['image_name'] for row in annotated if row.get('lidar_time_support_available') is False]
    os.environ['RECON_LIDAR_FRAME_ADMISSION'] = 'RequireEveryVisualFrame'
    strict, _ = bind(select_pose_gap_candidates)(rows, manifest['policy'], manifest['receipt']['sharpness_min'])
    os.environ['RECON_LIDAR_FRAME_ADMISSION'] = 'PreserveVisualFrames'
    fixture = [
        {'image_name': 'a', 'lidar_timestamp_ns': -1, 'path': None, 'stable_anchor_point_count': 0, 'absolute_delta_ms': None},
        {'image_name': 'b', 'lidar_timestamp_ns': 10, 'path': 'ten.npz', 'stable_anchor_point_count': 0, 'absolute_delta_ms': 0.},
        {'image_name': 'c', 'lidar_timestamp_ns': 20, 'path': 'twenty.npz', 'stable_anchor_point_count': 5000, 'absolute_delta_ms': 2.}]
    coverage = assess_partial_coverage(fixture, {10: 5000, 20: 5000, 30: 5000})
    if coverage['visual_frame_count'] != 3 or coverage['lidar_contributing_frame_count'] != 1:
        raise AssertionError('Missing and weak LiDAR must abstain without dropping visual frames')
    try:
        assess_partial_coverage(fixture, {10: 100000, 20: 100000})
    except ValueError:
        pass
    else:
        raise AssertionError('Repeated images fabricated three independent scans')
    try:
        assess_partial_coverage(fixture, {10: 100000, 20: 0, 30: 0})
    except ValueError:
        pass
    else:
        raise AssertionError('Empty scans were counted as stable evidence')
    fixture[0]['path'] = 'borrowed.npz'
    try:
        assess_partial_coverage(fixture, {10: 5000, 20: 5000, 30: 5000})
    except ValueError:
        pass
    else:
        raise AssertionError('An unpaired visual frame borrowed LiDAR evidence')
    shared = read_json(run / 'shared/RESULT.json')
    atomic_json(output, {'status': 'optimized_visual_admission_preserved_and_sensor_abstention_verified',
        'original_additions': len(original), 'adaptive_additions': len(actual),
        'explicit_legacy_time_filter_additions': len(strict), 'missing_lidar_additions_retained': len(absent),
        'quality_and_visual_selection_rows_exactly_preserved': True,
        'shared_existing_training_frames': shared['registered_frames'], 'new_training_performed': False,
        'unpaired_and_zero_anchor_frames_keep_visual_evidence': True,
        'two_independent_scan_rejection_verified': True, 'borrowed_lidar_rejection_verified': True})
    print('visual recovery candidates preserved', len(actual), 'legacy filter would keep', len(strict), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--time-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.run_root.resolve(strict=True), args.time_index.resolve(strict=True), args.output)
