"""Recover known physical metric failures with bounded offline numerical steps."""

import importlib.util
from pathlib import Path
import time
import traceback

import numpy as np
import pycolmap

from metric_point_policy import ROOT, native_point_policy
from native_positive_depth import add_positive_depth_residuals
from positive_depth_point_initialization import initialize_observed_point

RECOVERY_PROTOCOL = 'sensor_position_warmup_active_depth_joint_BA_v3'

# The metric entry already caches a different module named common.
_RECEIPT_SPEC = importlib.util.spec_from_file_location('metric_recovery_receipt_publisher',
    ROOT/'02_code/gaussian_full_raw_ab_20261008/atomic_json_windows.py')
_RECEIPT_PUBLISHER = importlib.util.module_from_spec(_RECEIPT_SPEC)
_RECEIPT_SPEC.loader.exec_module(_RECEIPT_PUBLISHER)


def _write_json(helpers, path, value, *, update=False):
    path = Path(path)
    if path.exists() and not update:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _RECEIPT_PUBLISHER.atomic_json(path, helpers.serializable(value))


def maximum_camera_radius(model):
    centers = np.asarray([image.projection_center() for image in model.images.values() if image.has_pose])
    if not len(centers) or not np.isfinite(centers).all():
        raise ValueError('A metric extent requires finite registered camera centers')
    return float(np.linalg.norm(centers, axis=1).max())


def _solve(model, rows, helpers, checks, output, loss, active_points, near_m, threads):
    output.mkdir(exist_ok=False)
    options = helpers.ba_options(60)
    options.ceres.solver_options.max_solver_time_in_seconds = 180
    # The installed COLMAP entry blocks Python callbacks in multiple threads.
    # Its complete single-thread entry retains native metric denormalization.
    options.ceres.solver_options.num_threads = 1 if active_points else threads
    config = helpers.ba_config(model)
    visual = int(config.num_residuals(model))
    priors = helpers.make_priors(model, rows, .15)
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    prior_options.prior_position_fallback_stddev = .15
    prior_options.alignment_ransac = pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2,
        max_num_trials=10000, random_seed=0)
    prior_options.ceres.prior_position_loss_function_type = loss
    prior_options.ceres.prior_position_loss_scale = 2.7954834829151074
    adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(
        options, prior_options, config, priors, model)
    keepalive, depth_rows = add_positive_depth_residuals(adjuster, model, active_points, near_m)
    _write_json(helpers, output/'PHYSICAL_DEPTH_FACTORS.json', {'rows': depth_rows,
        'physical_residual_count': len(depth_rows), 'position_prior_count': len(priors),
        'same_actual_native_pose_and_point_blocks': True})
    native = adjuster.solve()
    solution = helpers.solver_receipt(native)
    solution['actual_threads_used'] = int(native.ceres_summary.num_threads_used)
    destination = output/'unaccepted_model'
    destination.mkdir()
    model.write(destination)
    factors = (solution['num_residuals'] == visual+3*len(priors)+len(depth_rows)
        and solution['num_residual_blocks'] == visual//2+len(priors)+len(depth_rows))
    if active_points and solution['actual_threads_used'] != 1:
        raise ValueError('Python physical residuals require the checked single-thread native entry')
    validity = checks.projection_validity(model)
    _write_json(helpers, output/'PROJECTION_VALIDITY.json', validity)
    result = {'solution': solution, 'factor_count_verified': factors,
        'active_position_priors': len(priors), 'expected_visual_residuals': visual,
        'separate_physical_depth_residuals': len(depth_rows),
        'invalid_projection_count': validity['invalid_count'],
        'native_complete_metric_coordinate_lifecycle_used': True, 'model': str(destination)}
    _write_json(helpers, output/'RESULT.json', result)
    if not solution['usable'] or not factors:
        raise ValueError('Numerical recovery lacks usable optimization or actual native factor activation')
    return result, validity


def recover_metric_geometry(source, prior_rows, reference_p95, output, helpers, checks, metrics_fn, threads):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if not np.isfinite(reference_p95):
        raise ValueError('Physical recovery requires the genuine common-observation visual reference')
    policy_path, policy, renderer, near_m = native_point_policy()
    model = pycolmap.Reconstruction(source)
    identity = checks.observation_identity(model)
    intrinsics = {key: camera.params.copy() for key, camera in model.cameras.items()}
    image_names = {image.name for image in model.images.values() if image.has_pose}
    point_ids = set(model.points3D)
    contract = {'protocol': RECOVERY_PROTOCOL, 'runtime': 'offline_numeric_code_no_LLM_or_VLM',
        'prior_sigma_m': .15, 'prior_covariance_is_assumed_weight_not_sensor_estimate': True,
        'warmup_prior_loss': 'TRIVIAL', 'final_prior_loss': 'CAUCHY',
        'iterations_per_solver': 60, 'seconds_per_solver': 180, 'maximum_active_rounds': 3,
        'physical_depth_scale_from_installed_renderer_m': near_m, 'native_renderer_source': str(renderer),
        'point_policy_source': str(policy_path), 'original_camera_radius_guard_m': policy['maximum_camera_radius_m'],
        'original_common_observation_p95_guard_px': .05,
        'new_physical_objective_explicit_not_claimed_as_original_solver_unchanged': True,
        'position_and_physical_residuals_accounted_separately': True,
        'fixed_intrinsics_and_genuine_observations_preserved': True,
        'point_selection': 'current_actual_invalid_projection_tracks',
        'initialization_seed': 'same_attempt_aligned_registered_geometry_before_rejected_BA',
        'positive_seed_methods': ['positive_mean_observed_ray', 'genuine_multi_view_bearing_intersection',
            'positive_constrained_multi_view_bearing_seed'],
        'maximum_seed_and_true_pixel_iterations_each': 100,
        'inverse_depth_constraint': 'same_positive_minimum_as_enforced_variable_bound',
        'true_pixel_objective_scaling': 'positive_scalar_with_same_absolute_pixel_tolerance',
        'final_native_pixel_formula_equivalence_required': True,
        'early_stop': 'all_original_projection_common_pixel_and_metric_extent_guards_pass',
        'no_historical_frame_point_object_labels_or_manual_result_adoption': True,
        'native_complete_metric_coordinate_lifecycle_required': True, 'GT_used': False}
    _write_json(helpers, output/'CONTRACT.json', contract)

    def check_invariants():
        if (checks.observation_identity(model) != identity or set(model.points3D) != point_ids
                or {image.name for image in model.images.values() if image.has_pose} != image_names
                or set(model.cameras) != set(intrinsics)
                or any(not np.array_equal(camera.params, intrinsics[key])
                    for key, camera in model.cameras.items())):
            raise ValueError('Physical recovery changed genuine observations, registration, points, or calibration')

    phase = 'prior_warm_start'
    try:
        _write_json(helpers, output/'STATE.json', {'status': 'running', 'phase': phase}, update=True)
        warm, validity = _solve(model, prior_rows, helpers, checks, output/phase,
            pycolmap.LossFunctionType.TRIVIAL, set(), near_m, threads)
        check_invariants()
        active, rounds = set(), []
        for index in range(3):
            phase = 'physical_joint_BA_'+str(index+1)
            folder = output/phase
            invalid = {int(row['point3D_id']) for row in validity['invalid']}
            if index and invalid and invalid <= active:
                raise ValueError('Existing physical constraints still leave invalid genuine projections')
            active.update(invalid)
            decisions = []
            _write_json(helpers, output/'STATE.json', {'status': 'running', 'phase': phase,
                'active_points': len(active), 'new_violating_points': len(invalid)}, update=True)
            for key in sorted(invalid):
                xyz, decision = initialize_observed_point(model, key, near_m)
                decisions.append(decision)
                _write_json(helpers, output/(phase+'_POINT_INITIALIZATION.json'), {'rows': decisions}, update=True)
                if not decision['optimizer_success'] or not decision['finite_positive_depth']:
                    raise ValueError('A genuine violating track has no completed positive pixel initialization')
                model.points3D[key].xyz = xyz
            row, validity = _solve(model, prior_rows, helpers, checks, folder,
                pycolmap.LossFunctionType.CAUCHY, active, near_m, threads)
            check_invariants()
            radius = maximum_camera_radius(model)
            metrics = metrics_fn(model) if validity['valid'] else None
            delta = float(metrics['reprojection_px']['p95']-reference_p95) if metrics else None
            row['original_admission_guards_at_this_round'] = {
                'all_actual_projections_finite_and_positive': validity['valid'],
                'common_observation_p95_delta_px': delta,
                'common_observation_p95_delta_guard_px': .05,
                'metric_camera_radius_m': radius,
                'metric_camera_radius_guard_m': policy['maximum_camera_radius_m'],
                'passed': bool(validity['valid'] and delta <= .05 and radius <= policy['maximum_camera_radius_m'])}
            _write_json(helpers, output/(phase+'_ADMISSION_GUARDS.json'), row['original_admission_guards_at_this_round'])
            rounds.append(row)
            if row['original_admission_guards_at_this_round']['passed']:
                break
        if not validity['valid']:
            raise ValueError('Bounded physical recovery did not restore all genuine observed projections')
        model.update_point_3d_errors()
        metrics = metrics_fn(model)
        delta = float(metrics['reprojection_px']['p95']-reference_p95)
        radius = maximum_camera_radius(model)
        if delta > .05 or radius > policy['maximum_camera_radius_m']:
            raise ValueError('Numerical recovery fails the original common-observation or metric extent guard')
        check_invariants()
        accepted = output/'accepted_model'
        accepted.mkdir()
        model.write(accepted)
        final = rounds[-1]
        result = {'status': 'completed_original_metric_guards', 'protocol': RECOVERY_PROTOCOL,
            'model': str(accepted), 'solver': final['solution'], 'warmup': warm, 'rounds': rounds,
            'metrics': metrics, 'common_observation_p95_delta_px': delta,
            'maximum_metric_camera_radius_m': radius, 'active_position_priors': len(prior_rows),
            'physical_depth_residuals': final['separate_physical_depth_residuals'],
            'factor_count_verified': True, 'common_observations_preserved': True,
            'native_complete_metric_coordinate_lifecycle_used': True,
            'elapsed_seconds': time.monotonic()-start, 'counted_as_full_cold_delivery': False}
        _write_json(helpers, output/'RESULT.json', result)
        _write_json(helpers, output/'STATE.json', {'status': result['status'], 'phase': phase}, update=True)
        return model, result
    except BaseException:
        failure = {'status': 'failed_offline_metric_geometry_recovery', 'phase': phase,
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic()-start,
            'original_failure_not_relabelled': True, 'counted_as_full_cold_delivery': False}
        _write_json(helpers, output/'FAILURE.json', failure)
        _write_json(helpers, output/'STATE.json', {'status': failure['status'], 'phase': phase}, update=True)
        raise
