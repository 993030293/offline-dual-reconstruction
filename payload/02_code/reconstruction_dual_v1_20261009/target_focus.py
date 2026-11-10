"""Fit directed camera rays with the original full-camera forward requirement."""

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

REQUIRED_FORWARD_FRACTION = 0.95
AXIS_HUBER_CUTOFF_M = 0.35


def numerical_constraint_correction(centers, directions, focus, selected, near_m, tolerance):
    """Find one bounded roundoff correction in the original complete halfspace set."""
    geometry_scale = max(1., float(np.abs(centers).max()), float(np.abs(focus).max()))
    maximum_shift = np.sqrt(np.finfo(float).eps) * geometry_scale
    depths = np.einsum('ij,ij->i', focus[None] - centers, directions)
    deficit = near_m - depths[selected]
    scale = max(tolerance, float(deficit.max()))
    if scale > maximum_shift:
        raise ValueError('A geometric focus violation exceeds bounded numerical correction')
    matrix = np.vstack((np.column_stack((-directions[selected], np.zeros(len(selected)))),
        np.column_stack((np.eye(3), -np.ones(3))),
        np.column_stack((-np.eye(3), -np.ones(3)))))
    limits = np.r_[-(deficit + tolerance) / scale, np.zeros(6)]
    correction = linprog([0., 0., 0., 1.], A_ub=matrix, b_ub=limits,
        bounds=[(None, None)] * 3 + [(0., maximum_shift / scale)],
        method='highs', options={'maxiter': 200})
    if not correction.success:
        raise ValueError('The bounded original-halfspace numerical correction is not feasible')
    shift = correction.x[:3] * scale
    if not np.isfinite(shift).all() or np.abs(shift).max() > maximum_shift:
        raise ValueError('The focus correction exceeds its machine-derived numerical scale')
    candidate = focus + shift
    actual_depths = np.einsum('ij,ij->i', candidate[None] - centers, directions)
    if np.any(actual_depths[selected] < near_m - tolerance):
        raise ValueError('The corrected focus still violates the unchanged near-plane guard')
    return candidate, {'used': True, 'method': 'scaled_minimum_infinity_norm_original_halfspace_correction',
        'maximum_numerical_shift_m': maximum_shift, 'actual_maximum_focus_component_shift_m': float(np.abs(shift).max()),
        'original_guard_tolerance_m': float(tolerance), 'original_maximum_near_plane_deficit_m': float(deficit.max()),
        'original_violating_selected_camera_count': int(np.count_nonzero(deficit > tolerance)),
        'actual_minimum_selected_depth_m': float(actual_depths[selected].min()),
        'LP_success': bool(correction.success), 'LP_message': str(correction.message),
        'LP_iterations': int(correction.nit), 'all_selected_halfspaces_retained': True,
        'original_guard_tolerance_relaxed': False, 'camera_denominator_reduced': False,
        'camera_inputs_or_directions_modified': False, 'manual_focus_coordinates_used': False}


def forward_focus(centers, directions, near_m):
    centers = np.asarray(centers, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if (centers.ndim != 2 or centers.shape[1] != 3 or directions.shape != centers.shape
            or len(centers) < 3 or not np.isfinite(centers).all()
            or not np.isfinite(directions).all() or not np.isfinite(near_m) or near_m <= 0):
        raise ValueError('Forward focus requires finite camera centers, directions, and a positive near plane')
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., rtol=0, atol=1e-9):
        raise ValueError('Forward focus requires actual unit CV optical directions')
    gram = np.sum(np.eye(3)[None] - directions[:, :, None] * directions[:, None, :], axis=0)
    if np.linalg.eigvalsh(gram).min() <= np.finfo(float).eps * max(1., float(np.trace(gram))):
        raise ValueError('Camera rays do not identify a finite three-dimensional focus')

    def objective(focus):
        delta = focus[None] - centers
        depths = np.einsum('ij,ij->i', delta, directions)
        residual = delta - np.maximum(depths, 0)[:, None] * directions
        distances = np.linalg.norm(residual, axis=1)
        weights = np.minimum(1., AXIS_HUBER_CUTOFF_M / np.maximum(distances, 1e-12))
        losses = np.where(distances <= AXIS_HUBER_CUTOFF_M, .5 * distances**2,
                          AXIS_HUBER_CUTOFF_M * (distances - .5 * AXIS_HUBER_CUTOFF_M))
        return float(losses.sum()), np.sum(weights[:, None] * residual, axis=0)

    unconstrained = minimize(objective, centers.mean(axis=0), jac=True, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
    if not unconstrained.success or not np.isfinite(unconstrained.x).all():
        raise ValueError('Directed robust focus did not produce a completed finite estimate')
    depths = np.einsum('ij,ij->i', unconstrained.x[None] - centers, directions)
    required = int(np.ceil(REQUIRED_FORWARD_FRACTION * len(centers)))
    # Geometry breaks ties independently of image identity or file iteration order.
    order = np.lexsort((directions[:, 2], directions[:, 1], directions[:, 0],
                        centers[:, 2], centers[:, 1], centers[:, 0], -depths))
    selected = order[:required]
    matrix = directions[selected]
    bounds = np.einsum('ij,ij->i', matrix, centers[selected]) + near_m
    fit, constrained = unconstrained, False
    if np.any(depths[selected] < near_m):
        feasible = linprog(np.zeros(3), A_ub=-matrix, b_ub=-bounds,
                           bounds=[(None, None)] * 3, method='highs')
        if not feasible.success:
            raise ValueError('The original camera fraction has no forward near-plane intersection')
        fit = minimize(objective, feasible.x, jac=True, method='SLSQP',
                       constraints=[LinearConstraint(matrix, bounds, np.inf)],
                       options={'maxiter': 200, 'ftol': 1e-10})
        constrained = True
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Forward constrained focus did not complete its bounded numerical solve')
    focus = fit.x.copy()
    delta = focus[None] - centers
    depths = np.einsum('ij,ij->i', delta, directions)
    tolerance = 100 * np.finfo(float).eps * max(1., float(np.abs(centers).max()), float(np.abs(fit.x).max()))
    before = {'camera_count': len(centers), 'required_positive_count': required,
        'renderer_near_plane_m': float(near_m), 'original_positive_count': int(np.count_nonzero(depths > 0)),
        'original_minimum_selected_depth_m': float(depths[selected].min()),
        'original_maximum_near_plane_deficit_m': float((near_m - depths[selected]).max()),
        'original_guard_tolerance_m': float(tolerance), 'solver_success': bool(fit.success)}
    correction = {'used': False}
    try:
        if np.any(depths[selected] < near_m - tolerance):
            focus, correction = numerical_constraint_correction(centers, directions, focus, selected, near_m, tolerance)
            delta = focus[None] - centers
            depths = np.einsum('ij,ij->i', delta, directions)
        if np.any(depths[selected] < near_m - tolerance) or np.count_nonzero(depths > 0) < required:
            raise ValueError('Forward focus violates its original complete-camera support requirement')
    except ValueError as error:
        error.focus_constraint_evidence = before
        raise
    residuals = np.linalg.norm(delta - np.maximum(depths, 0)[:, None] * directions, axis=1)
    witness = {'method': 'directed_ray_huber_with_original_camera_fraction_and_renderer_near_plane',
               'camera_count': len(centers), 'required_forward_fraction': REQUIRED_FORWARD_FRACTION,
               'required_positive_count': required, 'actual_positive_count': int(np.count_nonzero(depths > 0)),
               'renderer_near_plane_m': float(near_m), 'minimum_constrained_forward_depth_m': float(depths[selected].min()),
               'constrained_camera_count': len(selected), 'constrained_refinement_used': constrained,
               'unconstrained_positive_count': int(np.count_nonzero(np.einsum('ij,ij->i', unconstrained.x[None] - centers, directions) > 0)),
               'solver_success': bool(fit.success), 'solver_message': str(fit.message),
               'solver_iterations': int(fit.nit), 'objective': objective(focus)[0],
               'solver_objective_before_numerical_correction': float(fit.fun),
               'numerical_constraint_correction': correction,
               'original_candidate_constraint_evidence': before,
               'camera_directions_flipped': False, 'camera_inputs_modified': False,
               'camera_denominator_reduced': False, 'manual_target_coordinates_used': False}
    return focus, residuals, int(fit.nit), witness
