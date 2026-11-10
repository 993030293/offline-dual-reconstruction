"""Verify the raw-time gate rejects the actual failed pose-gap additions."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from common import ROOT, atomic_json, read_json
from pose_gap_support import bind, nearest_delta

sys.path.insert(0, str(ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005'))
from pose_gap_candidates import select_pose_gap_candidates


def check(root, target):
    config = read_json(root / 'CONFIG.json')
    initial = root / config['case'] / 'optimized/initial'
    rows = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
    manifest = read_json(initial / 'INITIAL_FRAME_MANIFEST.json')
    time_path = target / 'raw_time_index/registered_lidar_header_ns.npy'
    os.environ['RECON_REGISTERED_LIDAR_TIMESTAMPS'] = str(time_path)
    os.environ['RECON_LIDAR_FRAME_ADMISSION'] = 'RequireEveryVisualFrame'
    candidates, original_decisions = select_pose_gap_candidates(rows, manifest['policy'], manifest['receipt']['sharpness_min'])
    retained, decisions = bind(select_pose_gap_candidates)(rows, manifest['policy'], manifest['receipt']['sharpness_min'])
    rejected = sorted(row['image_name'] for row in decisions if row['role'] == 'rejected_no_registered_lidar_time_support')
    prior_failure = read_json(target / 'stable_lidar/PAIRING_FAILURE.json')
    if rejected != sorted(prior_failure['images']):
        raise AssertionError('The pretraining gate did not isolate the actual failed additions')
    selected = read_json(target / 'registered_image_manifest.json')
    stamps = np.load(time_path, allow_pickle=False)
    times = [row['camera_header_ns'] for row in selected]
    delta = nearest_delta(times, stamps)
    oracle = np.abs(np.asarray(times)[:, None] - stamps[None, :]).min(1)
    if not np.array_equal(delta, oracle):
        raise AssertionError('The admission search differs from exhaustive raw scan pairing')
    supported = [row for row, value in zip(selected, delta) if value <= 60_000_000]
    destination = target / 'diagnostics/supported_time_subset'
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / 'registered_image_manifest.json', supported)
    import shutil
    shutil.copyfile(target / 'vehicle_roi_receipt.json', destination / 'vehicle_roi_receipt.json')
    atomic_json(target / 'TIME_ADMISSION_REGRESSION.json', {
        'status': 'pretraining_gate_matches_actual_raw_pairing_failure',
        'previous_pose_gap_candidates': len(candidates), 'supported_pose_gap_candidates': len(retained),
        'rejected_candidates': len(rejected), 'rejected_names': rejected,
        'previous_common_registered_frames': len(selected), 'time_supported_subset': len(supported),
        'raw_timestamp_exhaustive_oracle_verified': True, 'minimum_quality_threshold_changed': False,
        'maximum_pair_delta_ms_changed': False, 'maximum_pair_delta_ms': 60.,
        'new_matching_or_training_executed': False,
        'diagnostic_subset_manifest': str(destination / 'registered_image_manifest.json')})
    print('raw time admission regression', len(candidates), '->', len(retained), 'rejected', len(rejected), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    check(args.run_root.resolve(strict=True), args.output_root.resolve(strict=True))
