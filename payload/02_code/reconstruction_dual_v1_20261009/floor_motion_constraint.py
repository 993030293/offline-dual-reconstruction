"""Bind measured low planes to one raw-motion frame without object labels."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate'
_spec = importlib.util.spec_from_file_location('raw_measured_floor_estimator',
    PACKAGE / 'code/estimate_sparse_floor.py')
_estimator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_estimator)
estimate_sparse_floor = _estimator.estimate_sparse_floor
FLOOR_SEARCH_RADIUS_M = _estimator.FLOOR_SEARCH_RADIUS_M
MAXIMUM_INLIER_RESIDUAL_M = _estimator.MAXIMUM_INLIER_RESIDUAL_M
MAXIMUM_PLANE_TILT_DEGREES = _estimator.MAXIMUM_PLANE_TILT_DEGREES

PROTOCOL = 'raw_measured_low_plane_motion_constraint_v1'
MINIMUM_REFERENCE_SCANS = 3
QUALITY_REJECTION_PREFIXES = ('Too few metric sparse floor candidates:',
    'Metric sparse floor has too few occupied XY cells', 'Sparse floor lacks',
    'Sparse floor is not gravity aligned:', 'Sparse floor implies implausible camera height:')


def minimum_normal_rotation(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.shape != (3,) or target.shape != (3,) or not np.isfinite(source).all()
            or not np.isfinite(target).all() or np.linalg.norm(source) <= 0
            or np.linalg.norm(target) <= 0):
        raise ValueError('Measured plane normals are missing or nonfinite')
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine, cosine = np.linalg.norm(cross), float(np.clip(np.dot(source, target), -1., 1.))
    if sine < 1e-12:
        if cosine <= 0:
            raise ValueError('Opposite plane normals have no unique minimum rotation')
        return Rotation.identity()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine))


def plane_translation_equation(normal, local_offset, world_offset):
    # A sensor-relative plane becomes n.dot(t) = d_local - d_world.
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite([local_offset, world_offset]).all():
        raise ValueError('Measured plane translation equation is invalid')
    if not np.isclose(np.linalg.norm(normal), 1., atol=1e-10, rtol=0):
        raise ValueError('The measured plane reference normal is not a unit vector')
    return normal.copy(), float(local_offset - world_offset)


def measured_plane(points):
    try:
        return estimate_sparse_floor(points, np.zeros(2), 0., FLOOR_SEARCH_RADIUS_M), None
    except RuntimeError as error:
        if not str(error).startswith(QUALITY_REJECTION_PREFIXES):
            raise
        return None, str(error)


class FloorMotionConstraint:
    """Use current scans only; an unobservable plane is an explicit abstention."""

    def __init__(self):
        self.reference_normal = None
        self.reference_offset = None
        self.bootstrap = []
        self.correction = Rotation.identity()
        self.rows = []
        self.corrections = []
        self.supported = []

    def propose(self, gyro_rotated_points, point_dt_s, previous_velocity, gyro_end, translation, stamp):
        if self.rows and stamp <= self.rows[-1]['timestamp_ns']:
            raise ValueError('Measured floor scans must have unique increasing clocks')
        points = np.asarray(gyro_rotated_points, float)
        dt, velocity, translation = np.asarray(point_dt_s, float), np.asarray(previous_velocity, float), np.asarray(translation, float)
        if (points.ndim != 2 or points.shape[1] != 3 or dt.shape != (len(points),)
                or velocity.shape != (3,) or translation.shape != (3,)
                or not all(np.isfinite(value).all() for value in (points, dt, velocity, translation))):
            raise ValueError('The raw measured floor input is invalid')
        predicted = self.correction.apply(points) + dt[:, None] * velocity
        floor, rejection = measured_plane(predicted)
        row = {'timestamp_ns': int(stamp), 'status': 'abstain_floor_measurement',
               'measurement': floor, 'reason': rejection, 'motion_accepted': False,
               'reference_from_independent_actual_scans': len(self.bootstrap)}
        constraint = None
        if floor is not None:
            normal, offset = np.asarray(floor['normal']), float(floor['offset_m'])
            if self.reference_normal is None:
                world_offset = offset - float(np.dot(normal, translation))
                self.bootstrap.append((normal.copy(), world_offset, floor['inlier_cell_count']))
                row['reference_from_independent_actual_scans'] = len(self.bootstrap)
                row.update(status='initializing_measured_plane_reference', reason=None)
                if len(self.bootstrap) >= MINIMUM_REFERENCE_SCANS:
                    weights = np.asarray([value[2] for value in self.bootstrap], float)
                    normals = np.asarray([value[0] for value in self.bootstrap])
                    offsets = np.asarray([value[1] for value in self.bootstrap])
                    proposed = np.average(normals, axis=0, weights=weights)
                    proposed /= np.linalg.norm(proposed)
                    center = float(np.average(offsets, weights=weights))
                    if (np.any(normals @ proposed < np.cos(np.deg2rad(MAXIMUM_PLANE_TILT_DEGREES)))
                            or np.max(np.abs(offsets - center)) > MAXIMUM_INLIER_RESIDUAL_M):
                        self.bootstrap = [self.bootstrap[-1]]
                        row.update(reason='independent_measured_planes_do_not_share_one_reference',
                                   reference_from_independent_actual_scans=1)
                    else:
                        self.reference_normal, self.reference_offset = proposed, center
            if self.reference_normal is not None:
                delta = minimum_normal_rotation(normal, self.reference_normal)
                angle = float(np.rad2deg(delta.magnitude()))
                if angle > MAXIMUM_PLANE_TILT_DEGREES:
                    row.update(status='abstain_plane_rotation_conflict', reason='original_plane_tilt_limit',
                               correction_increment_degrees=angle)
                else:
                    candidate = delta * self.correction
                    refined, refined_rejection = measured_plane(candidate.apply(points) + dt[:, None] * velocity)
                    if refined is None:
                        row.update(reason=refined_rejection)
                    else:
                        n, target = plane_translation_equation(self.reference_normal,
                            float(refined['offset_m']), self.reference_offset)
                        self.correction = candidate
                        constraint = {'normal': n, 'target_m': target,
                            'independent_inlier_cells': int(refined['inlier_cell_count'])}
                        row.update(status='measured_plane_constraint_ready', reason=None,
                            correction_increment_degrees=angle, refined_measurement=refined,
                            target_normal_translation_m=target,
                            independent_inlier_cells=int(refined['inlier_cell_count']))
        self.rows.append(row)
        self.corrections.append(self.correction.as_quat())
        self.supported.append(constraint is not None)
        return self.correction * gyro_end, constraint

    def accepted_translation(self, constraint, translation):
        if constraint is None:
            self.rows[-1]['motion_accepted'] = False
            return False
        residual = float(abs(np.dot(constraint['normal'], translation) - constraint['target_m']))
        passed = np.isfinite(residual) and residual <= MAXIMUM_INLIER_RESIDUAL_M
        self.rows[-1].update(actual_plane_translation_constraint_residual_m=residual, motion_accepted=bool(passed))
        return bool(passed)

    def receipt(self):
        return {'protocol': PROTOCOL, 'status': 'raw_measured_floor_motion_complete',
            'reference_normal': None if self.reference_normal is None else self.reference_normal.tolist(),
            'reference_offset_m': self.reference_offset,
            'minimum_independent_reference_scans': MINIMUM_REFERENCE_SCANS,
            'actual_scan_count': len(self.rows), 'actual_supported_scan_count': int(sum(self.supported)),
            'actual_accepted_plane_motion_count': sum(bool(row['motion_accepted']) for row in self.rows),
            'per_scan': self.rows, 'original_floor_quality_rules_preserved': True,
            'scan_rotation_correction_constant_within_each_scan': True,
            'ground_reference_is_one_plane_assumption': True,
            'object_dimensions_labels_or_historical_planes_used': False,
            'future_sequence_measurements_or_GT_used': False,
            'runtime_language_model_used': False, 'independent_geometry_accuracy_verified': False}
