"""Validate spatial and temporal support for a mesh-derived ground plane."""

from __future__ import annotations

import sys
import json
import hashlib
import time
from pathlib import Path

import numpy as np


TOPIC = "/localization/lidar/points_registered_full"
SAMPLE_EVERY_NTH_FRAME = 10
NEAR_PLANE_DISTANCE_M = 0.02
GRID_CELL_M = 0.05
MINIMUM_NEAR_PLANE_POINTS = 1000
MINIMUM_SUPPORTED_FRAMES = 30
MINIMUM_POINTS_PER_SUPPORTED_FRAME = 20


def validate_generated_scan_source(raw_mcap, receipt_path, expected_run_root):
    """Bind full measured scans to this raw recording and reconstruction attempt."""
    source = Path(raw_mcap).resolve(strict=True)
    root = Path(expected_run_root).resolve(strict=True)
    path = Path(receipt_path).resolve(strict=True)
    if path != (root / 'raw_sensor_backend/RESULT.json').resolve(strict=True):
        raise ValueError('Generated ground scans cannot cross reconstruction attempts')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    binding = json.loads((root / 'SENSOR_BACKEND.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'CONFIG.json').read_text(encoding='utf-8'))
    original_identity = (source.stat().st_size, source.stat().st_mtime_ns)
    for value in (receipt, binding, config):
        if (Path(value['raw']).resolve(strict=True) != source
                or (value['raw_bytes'], value['raw_mtime_ns']) != original_identity):
            raise ValueError('Generated ground scans have a different original recording')
    if (receipt.get('status') != 'fresh_raw_registered_scans_complete'
            or Path(receipt['run_root']).resolve(strict=True) != root
            or binding.get('status') != 'raw_sensor_backend_selected'
            or binding.get('backend') != 'fresh_raw_gyro_translation_local_map'
            or receipt.get('topic') != '/sensing/lidar/points_raw'
            or receipt.get('source_frame') != 'fresh_local_gyro_translation_lio_world'):
        raise ValueError('Generated ground scan status or metric coordinate frame is invalid')
    for value, keys in ((receipt, ('runtime_language_model_used', 'historical_derived_inputs_used',
                                   'SfM_used_for_scan_registration', 'GT_used')),
                        (binding, ('runtime_language_model_used', 'historical_motion_or_geometry_used')),
                        (config, ('historical_derived_inputs_used', 'GT_used'))):
        if any(value.get(key) is not False for key in keys):
            raise ValueError('Generated ground scan provenance lacks explicit machine-only flags')
    rows = receipt['scans']
    if not isinstance(rows, list) or not rows or len(rows) != receipt['raw_scans_read']:
        raise ValueError('Generated ground scan accounting is incomplete')
    previous_stamp = -1
    eligible = 0
    backend = path.parent
    for index, row in enumerate(rows):
        stamp = row['scan_end_ns']
        if (row['index'] != index or type(stamp) is not int or stamp <= previous_stamp
                or type(row['eligible_registered_evidence']) is not bool):
            raise ValueError('Generated ground scans require unique ordered original timestamps')
        previous_stamp = stamp
        if row['eligible_registered_evidence']:
            scan = (backend / row['path']).resolve(strict=True)
            scan.relative_to((backend / 'scans').resolve(strict=True))
            if scan.suffix != '.npz' or row['valid_raw_points'] <= 0:
                raise ValueError('Generated ground scan file or point count is invalid')
            eligible += 1
        elif row['path'] is not None:
            raise ValueError('An excluded motion scan cannot supply ground evidence')
    if (eligible != receipt['eligible_scans']
            or len(rows) - eligible != receipt['rejected_evidence_scans']):
        raise ValueError('Generated ground scan eligibility counters disagree')
    evidence = {'source_kind': 'same_attempt_raw_motion_registered_scans',
        'source_registered_scan_receipt': str(path),
        'source_registered_scan_receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_run_root': str(root), 'source_mcap': str(source),
        'raw_bytes': original_identity[0], 'raw_mtime_ns': original_identity[1],
        'coordinate_frame': receipt['source_frame'], 'raw_scan_count': len(rows),
        'eligible_scan_count': eligible, 'runtime_language_model_used': False,
        'historical_derived_inputs_used': False, 'independent_geometry_accuracy_verified': False}
    return receipt, evidence


def generated_scan_points(receipt_path, row):
    """Load one full measured scan, never the target anchors with ground removed."""
    backend = Path(receipt_path).resolve(strict=True).parent
    scan = (backend / row['path']).resolve(strict=True)
    scan.relative_to((backend / 'scans').resolve(strict=True))
    with np.load(scan, allow_pickle=False) as archive:
        if int(archive['lidar_timestamp_ns']) != row['scan_end_ns']:
            raise ValueError('Generated ground scan has a different original timestamp')
        points = archive['points_world_lio_m'].copy()
    if (points.ndim != 2 or points.shape != (row['valid_raw_points'], 3)
            or points.dtype.kind != 'f' or not np.isfinite(points).all()):
        raise ValueError('Generated ground scan has invalid measured point data')
    return points


def validate_registered_lidar_support(
    raw_mcap: Path,
    normal: np.ndarray,
    offset_m: float,
    roi_minimum: np.ndarray,
    roi_maximum: np.ndarray,
    sparse_ground_z_m: float,
    minimum_xy_extent_m: float,
    minimum_roi_span_fraction: float,
    prior_height_half_window_m: float,
    registered_scan_receipt: Path | None = None,
    expected_run_root: Path | None = None,
    expand_evidence: bool = False,
    sample_every_nth_frame: int = SAMPLE_EVERY_NTH_FRAME,
) -> dict:
    """Check that independently measured low returns span the current metric ROI."""
    source = raw_mcap.resolve(strict=True)
    started = time.monotonic()
    if type(expand_evidence) is not bool or type(sample_every_nth_frame) is not int:
        raise ValueError('Ground evidence sampling policy requires explicit boolean and integer values')
    if sample_every_nth_frame not in (SAMPLE_EVERY_NTH_FRAME, 1):
        raise ValueError('Ground evidence supports only the fixed sampled or full-source pass')
    frame_count = 0
    sampled_frame_count = 0
    supported_frame_count = 0
    near_xy: list[np.ndarray] = []
    source_binding = None

    def near_ground(points):
        valid = np.isfinite(points).all(axis=1)
        valid &= np.all((points >= roi_minimum) & (points <= roi_maximum), axis=1)
        points = points[valid]
        distance = np.abs(points @ normal + offset_m)
        low = np.abs(points[:, 2] - sparse_ground_z_m) <= (
            prior_height_half_window_m + NEAR_PLANE_DISTANCE_M
        )
        return points[(distance <= NEAR_PLANE_DISTANCE_M) & low, :2]

    if registered_scan_receipt is not None:
        if expected_run_root is None:
            raise ValueError('Generated ground scans require the current surface run root')
        registered, source_binding = validate_generated_scan_source(
            source, registered_scan_receipt, expected_run_root)
        for row in registered['scans']:
            frame_count += 1
            if frame_count % sample_every_nth_frame or not row['eligible_registered_evidence']:
                continue
            sampled_frame_count += 1
            xy = near_ground(generated_scan_points(registered_scan_receipt, row))
            supported_frame_count += int(len(xy) >= MINIMUM_POINTS_PER_SUPPORTED_FRAME)
            if len(xy):
                near_xy.append(xy)
    else:
        from rosbags.highlevel import AnyReader
        code_root = Path(__file__).resolve().parents[2] / "code"
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from extract_stable_registered_lidar import decode_xyz, timestamp_ns
        seen_timestamps = set()
        with AnyReader([source]) as reader:
            connections = [item for item in reader.connections if item.topic == TOPIC]
            if len(connections) != 1:
                raise RuntimeError(f"Expected one registered LiDAR connection, found {len(connections)}")
            for connection, _, raw in reader.messages(connections=connections):
                frame_count += 1
                if frame_count % sample_every_nth_frame:
                    continue
                message = reader.deserialize(raw, connection.msgtype)
                if str(message.header.frame_id).lstrip("/") != "odom":
                    raise RuntimeError("Registered LiDAR frame does not match the metric ROI frame")
                stamp = timestamp_ns(message.header.stamp)
                if stamp in seen_timestamps:
                    raise ValueError('Duplicate recorded timestamps cannot supply independent ground evidence')
                seen_timestamps.add(stamp)
                sampled_frame_count += 1
                xy = near_ground(decode_xyz(message))
                supported_frame_count += int(len(xy) >= MINIMUM_POINTS_PER_SUPPORTED_FRAME)
                if len(xy):
                    near_xy.append(xy)

    combined = np.concatenate(near_xy) if near_xy else np.empty((0, 2))
    roi_extent = roi_maximum[:2] - roi_minimum[:2]
    xy_extent = np.ptp(combined, axis=0) if len(combined) else np.zeros(2)
    roi_span = xy_extent / roi_extent
    cells = np.unique(np.floor(combined / GRID_CELL_M).astype(np.int64), axis=0)
    occupied_area_m2 = float(len(cells) * GRID_CELL_M**2)
    required_area_m2 = float(minimum_xy_extent_m**2)
    accepted = bool(
        len(combined) >= MINIMUM_NEAR_PLANE_POINTS
        and supported_frame_count >= MINIMUM_SUPPORTED_FRAMES
        and np.all(xy_extent >= minimum_xy_extent_m)
        and np.all(roi_span >= minimum_roi_span_fraction)
        and occupied_area_m2 >= required_area_m2
    )
    evidence = {
        "status": "accepted" if accepted else "insufficient_support",
        "source_mcap": str(source),
        "topic": TOPIC,
        "coordinate_frame": "odom",
        "sample_every_nth_frame": sample_every_nth_frame,
        "frame_count": frame_count,
        "sampled_frame_count": sampled_frame_count,
        "supported_frame_count": supported_frame_count,
        "minimum_supported_frames": MINIMUM_SUPPORTED_FRAMES,
        "minimum_points_per_supported_frame": MINIMUM_POINTS_PER_SUPPORTED_FRAME,
        "near_plane_point_count": int(len(combined)),
        "minimum_near_plane_points": MINIMUM_NEAR_PLANE_POINTS,
        "near_plane_distance_m": NEAR_PLANE_DISTANCE_M,
        "near_sparse_ground_height_half_window_m": (
            prior_height_half_window_m + NEAR_PLANE_DISTANCE_M
        ),
        "xy_extent_m": xy_extent.tolist(),
        "minimum_xy_extent_m": minimum_xy_extent_m,
        "roi_span_fraction": roi_span.tolist(),
        "minimum_roi_span_fraction": minimum_roi_span_fraction,
        "grid_cell_m": GRID_CELL_M,
        "occupied_area_m2": occupied_area_m2,
        "minimum_occupied_area_m2": required_area_m2,
        "elapsed_seconds": time.monotonic() - started,
    }
    if source_binding is not None:
        evidence.update(source_binding)
        evidence['topic'] = '/sensing/lidar/points_raw'
    if not accepted:
        if expand_evidence and sample_every_nth_frame == SAMPLE_EVERY_NTH_FRAME:
            try:
                full = validate_registered_lidar_support(
                    source, normal, offset_m, roi_minimum, roi_maximum, sparse_ground_z_m,
                    minimum_xy_extent_m, minimum_roi_span_fraction, prior_height_half_window_m,
                    registered_scan_receipt, expected_run_root,
                    expand_evidence=False, sample_every_nth_frame=1)
            except RuntimeError as full_error:
                raise RuntimeError(
                    f'Ground support failed sampled and full-source evidence passes: '
                    f'sampled={evidence}; full_source={full_error}') from full_error
            full['support_sampling_attempts'] = [evidence, dict(full)]
            full['automatic_same_source_evidence_expansion'] = True
            full['total_support_validation_seconds'] = time.monotonic() - started
            return full
        raise RuntimeError(f"Registered LiDAR ground support is insufficient: {evidence}")
    return evidence
