"""Check dense frame pairing against an exhaustive timestamp oracle."""

import argparse
from pathlib import Path

import numpy as np

from common import atomic_json
from target_anchors import improved_images, supported_voxels


def check(output):
    rng = np.random.default_rng(27)
    times = np.arange(0, 1_000_000_000, 20_000_000, dtype=np.int64)
    scans = np.unique(np.r_[0, np.arange(0, 1_000_000_000, 100_000_000) + rng.integers(-500000, 500000, 10)])
    best = np.full(len(times), np.iinfo(np.int64).max, np.int64)
    chosen = np.full(len(times), -1, np.int64)
    for stamp in scans:
        indices = improved_images(times, stamp, best, 60_000_000)
        chosen[indices], best[indices] = stamp, np.abs(times[indices] - stamp)
    distances = np.abs(times[:, None] - scans[None, :])
    oracle_indices = distances.argmin(1)
    valid = distances[np.arange(len(times)), oracle_indices] <= 60_000_000
    oracle = np.where(valid, scans[oracle_indices], -1)
    if not np.array_equal(chosen, oracle):
        raise AssertionError('Dense pairing disagrees with the independent exhaustive oracle')
    point = np.array([[1.001, 2.001, 3.001]], np.float32)
    if len(supported_voxels({11: point, 22: point}, minimum=3)):
        raise AssertionError('Two independent scans were accepted as three')
    if len(supported_voxels({11: point, 22: point, 33: point}, minimum=3)) != 1:
        raise AssertionError('Genuine three-scan support was lost')
    atomic_json(output, {'status': 'dense_pairing_and_independent_scan_support_verified',
        'oracle': 'exhaustive nearest timestamp matrix with earliest chronological tie',
        'image_count': len(times), 'scan_count': len(scans), 'paired_images': int(valid.sum()),
        'two_scan_rejection_verified': True, 'three_scan_acceptance_verified': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    check(parser.parse_args().output)
