"""Freeze an explicitly supplied camera-to-body calibration numeric subset."""

import hashlib
from pathlib import Path

import numpy as np

from common import atomic_json, read_json


def load_calibration(source):
    source = Path(source).resolve(strict=True)
    declared = read_json(source)
    entry = declared['T_base_link_camera_optical']
    if entry['source_frame'] != 'camera_optical' or entry['target_frame'] != 'base_link':
        raise ValueError('Explicit calibration must map camera_optical into base_link')
    values = np.asarray(entry['pose_xyz_xyzw'], dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Explicit calibration must contain seven finite rigid-pose values')
    if abs(np.linalg.norm(values[3:]) - 1.) > 1e-9:
        raise ValueError('Explicit calibration quaternion must be normalized')
    subset = {'source_frame': entry['source_frame'], 'target_frame': entry['target_frame'],
              'pose_xyz_xyzw': values.tolist()}
    identity = {'path': str(source), 'bytes': source.stat().st_size,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'consumed_field': 'T_base_link_camera_optical', 'numeric_subset': subset}
    return identity


def freeze_binding(root, case, identity, resume=False):
    frozen = root / 'EXPLICIT_BODY_CAMERA_CALIBRATION.json'
    subset = {'schema': 'explicit_camera_body_calibration_subset_v1',
              'source_identity': identity,
              'T_base_link_camera_optical': identity['numeric_subset'],
              'historical_trajectory_or_SfM_fields_consumed': False,
              'independent_cross_recording_rig_validation': 'unconfirmed'}
    bindings = {case: {'path': str(frozen), 'explicit_external_calibration': True,
                       'absolute_metric_accuracy_claim': False}}
    if resume:
        if read_json(frozen) != subset or read_json(root / 'CASE_CALIBRATION_BINDINGS.json') != bindings:
            raise ValueError('Frozen explicit calibration binding changed')
    else:
        atomic_json(frozen, subset)
        atomic_json(root / 'CASE_CALIBRATION_BINDINGS.json', bindings)
