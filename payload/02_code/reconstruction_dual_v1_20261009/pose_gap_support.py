"""Annotate LiDAR time availability without discarding valid visual recovery frames."""

import os
from pathlib import Path

import numpy as np


def nearest_delta(times, stamps):
    times, stamps = np.asarray(times, np.int64), np.asarray(stamps, np.int64)
    if stamps.ndim != 1 or len(stamps) < 3 or not (np.diff(stamps) > 0).all():
        raise ValueError('The registered scan time index is invalid')
    right = np.searchsorted(stamps, times)
    left_delta = np.abs(times - stamps[np.clip(right - 1, 0, len(stamps) - 1)])
    right_delta = np.abs(times - stamps[np.clip(right, 0, len(stamps) - 1)])
    return np.minimum(left_delta, right_delta)


def bind(original):
    path = os.environ.get('RECON_REGISTERED_LIDAR_TIMESTAMPS')
    if not path:
        return original
    stamps = np.load(Path(path).resolve(strict=True), allow_pickle=False)
    admission = os.environ.get('RECON_LIDAR_FRAME_ADMISSION', 'PreserveVisualFrames')
    if admission not in ('PreserveVisualFrames', 'RequireEveryVisualFrame'):
        raise ValueError('Unknown explicit LiDAR frame admission policy')

    def select(rows, policy, sharpness_min):
        candidates, decisions = original(rows, policy, sharpness_min)
        names = [row['image_name'] for row in candidates]
        times = [int(Path(name).stem.rsplit('_', 1)[1]) for name in names]
        delta = nearest_delta(times, stamps)
        excluded = {name: int(value) for name, value in zip(names, delta) if value > 60_000_000}
        deltas = dict(zip(names, map(int, delta)))
        for row in decisions:
            if row['image_name'] in deltas:
                row.update(lidar_time_support_available=row['image_name'] not in excluded,
                    registered_lidar_time_delta_ms=deltas[row['image_name']] / 1e6,
                    registered_lidar_maximum_pair_delta_ms=60.,
                    visual_frame_admission_policy=admission)
            if admission == 'RequireEveryVisualFrame' and row['image_name'] in excluded:
                row.update(role='rejected_no_registered_lidar_time_support',
                    registered_lidar_time_delta_ms=excluded[row['image_name']] / 1e6,
                    registered_lidar_maximum_pair_delta_ms=60.,
                    temporal_support_check_before_matching_and_training=True)
        if admission == 'RequireEveryVisualFrame':
            return [row for row in candidates if row['image_name'] not in excluded], decisions
        return candidates, decisions
    return select


def install():
    import pose_gap_candidates
    pose_gap_candidates.select_pose_gap_candidates = bind(pose_gap_candidates.select_pose_gap_candidates)
