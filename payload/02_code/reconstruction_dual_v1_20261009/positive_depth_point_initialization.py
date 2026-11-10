"""Initialize genuine OPENCV tracks with true pixels and positive inverse depth."""

import numpy as np
import pycolmap
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def verify_native_pixel_formula(model, point_id, coordinates, formula_cost):
    """Compare calibration on identical vectors before world-coordinate readback."""
    entries = model.points3D[point_id].track.elements
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(entries), 3) or not np.isfinite(coordinates).all():
        raise ValueError('The native formula check requires finite camera vectors for every actual observation')
    errors = []
    for entry, xyz in zip(entries, coordinates):
        image = model.images[entry.image_id]
        projected = model.cameras[image.camera_id].img_from_cam(xyz)
        if projected is None or not np.isfinite(projected).all():
            raise ValueError('The native formula check requires finite positive projections')
        errors.append(float(np.linalg.norm(projected-image.points2D[entry.point2D_idx].xy)))
    errors = np.asarray(errors)
    native_cost = float(np.where(errors <= 1, .5*errors**2, errors-.5).sum())
    if not np.isclose(formula_cost, native_cost, atol=1e-8, rtol=1e-12):
        raise ValueError('The OPENCV formula differs from actual native projections')
    return native_cost


def check_pixel_gradient(variables, analytic, real_cost, complex_cost):
    """Retain the same derivative bound when real differences lose precision."""
    limit = 1e-4*max(1, float(np.linalg.norm(analytic)))
    checks = []
    for step_scale in (np.sqrt(np.finfo(float).eps), np.cbrt(np.finfo(float).eps)):
        numerical = np.empty(3)
        for axis in range(3):
            step = step_scale*max(1, abs(variables[axis]))
            delta = np.zeros(3)
            delta[axis] = step
            numerical[axis] = (real_cost(variables+delta)-real_cost(variables-delta))/(2*step)
        error = float(np.linalg.norm(analytic-numerical))
        checks.append({'method': 'real_central_difference', 'central_difference_step_scale': float(step_scale),
                       'absolute_gradient_error': error, 'unchanged_gradient_error_limit': limit})
        if error <= limit:
            return error, checks
    # Complex arithmetic keeps the perturbation separate from a large real
    # constant. The caller supplies the same pixel formula and Huber branches.
    step_scale = np.finfo(float).eps**2
    numerical = np.empty(3)
    for axis in range(3):
        step = step_scale*max(1, abs(variables[axis]))
        shifted = np.asarray(variables, dtype=complex).copy()
        shifted[axis] += 1j*step
        numerical[axis] = np.imag(complex_cost(shifted))/step
    error = float(np.linalg.norm(analytic-numerical))
    checks.append({'method': 'complex_step_same_pixel_formula_and_huber_branches',
                   'complex_step_scale': float(step_scale), 'absolute_gradient_error': error,
                   'unchanged_gradient_error_limit': limit})
    if not np.isfinite(error) or error > limit:
        raise ValueError('The analytic pixel derivative failed its numerical check')
    return error, checks


def initialize_observed_point(model, point_id, margin):
    point = model.points3D[point_id]
    normals, offsets, centers, rays = [], [], [], []
    rotations, translations, camera_params, observed_pixels = [], [], [], []
    for entry in point.track.elements:
        image = model.images[entry.image_id]
        observation = image.points2D[entry.point2D_idx]
        if not image.has_pose or observation.point3D_id != point_id:
            raise ValueError('A point initializer requires genuine registered observations')
        pose = image.cam_from_world()
        camera = model.cameras[image.camera_id]
        if camera.model != pycolmap.CameraModelId.OPENCV:
            raise ValueError('Positive pixel initialization requires the declared OPENCV calibration')
        normalized = camera.cam_from_img(observation.xy)
        ray = pose.rotation.matrix().T @ np.r_[normalized, 1.0]
        ray /= np.linalg.norm(ray)
        normals.append(pose.rotation.matrix()[2])
        offsets.append(float(pose.translation[2]))
        centers.append(image.projection_center())
        rays.append(ray)
        rotations.append(pose.rotation.matrix())
        translations.append(pose.translation.copy())
        camera_params.append(camera.params.copy())
        observed_pixels.append(observation.xy.copy())
    matrix, shift, centers = np.asarray(normals), np.asarray(offsets), np.asarray(centers)
    if (not np.isfinite(margin) or margin <= 0 or len(centers) < 2
            or not all(np.isfinite(array).all() for array in (matrix, shift, centers))):
        raise ValueError('Positive pixel initialization requires finite multi-view geometry and depth')
    feasible = linprog(np.zeros(3), A_ub=-matrix, b_ub=shift-margin,
        bounds=[(None, None)]*3, method='highs')
    if not feasible.success:
        raise ValueError('No positive-depth intersection exists for this actual fixed-camera track')
    rotations, translations = np.asarray(rotations), np.asarray(translations)
    params, observed = np.asarray(camera_params), np.asarray(observed_pixels)
    rays = np.asarray(rays)
    direction = rays.sum(axis=0)
    direction_norm = np.linalg.norm(direction)
    direction = direction/direction_norm if direction_norm > np.finfo(float).eps else np.zeros(3)
    coefficients = matrix @ direction
    seed_method, seed_iterations = 'positive_mean_observed_ray', 0
    if np.all(coefficients > 0):
        center = centers.mean(axis=0)
        minimum_range = float(np.max((margin-matrix @ center-shift)/coefficients))
        distance = max(float(np.median(np.linalg.norm(point.xyz-centers, axis=1))),
            minimum_range+margin/float(coefficients.min()))
        seed = center+distance*direction
    else:
        # Opposing views can have a bounded feasible intersection but no ray
        # pointing forward in every camera. Bearing distances choose a seed;
        # only the subsequent true-pixel solve can initialize the final point.
        projectors = np.eye(3)[None, :, :]-rays[:, :, None]*rays[:, None, :]
        gram = projectors.sum(axis=0)
        linear = np.einsum('nij,nj->i', projectors, centers)
        seed = np.linalg.lstsq(gram, linear, rcond=None)[0]
        seed_method = 'genuine_multi_view_bearing_intersection'
        if np.any(matrix @ seed+shift < margin):
            def seed_objective(xyz):
                residual = np.einsum('nij,nj->ni', projectors, xyz-centers)
                return .5*np.square(residual).sum()
            prepared = minimize(seed_objective, feasible.x, jac=lambda xyz: gram @ xyz-linear,
                constraints=[LinearConstraint(matrix, margin-shift, np.inf)],
                method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12})
            if not prepared.success or not np.isfinite(prepared.x).all():
                raise ValueError('The genuine bearing intersection has no completed feasible seed')
            seed, seed_iterations = prepared.x, int(prepared.nit)
            seed_method = 'positive_constrained_multi_view_bearing_seed'
    anchor_center, anchor_rotation = centers[0], rotations[0]
    anchor_coordinates = anchor_rotation @ (seed-anchor_center)
    variable_seed = np.r_[anchor_coordinates[:2]/anchor_coordinates[2], 1/anchor_coordinates[2]]
    relative_rotations = rotations @ anchor_rotation.T
    relative_translations = np.einsum('nij,j->ni', rotations, anchor_center)+translations
    coordinate_jacobian = np.stack((relative_rotations[:, :, 0], relative_rotations[:, :, 1],
        relative_translations), axis=2)
    coordinate_constant = relative_rotations[:, :, 2]
    inverse_depth_min = np.sqrt(np.finfo(float).eps)/max(1, float(np.max(np.linalg.norm(centers, axis=1))))
    depth_constraints = np.column_stack((relative_rotations[:, 2, 0], relative_rotations[:, 2, 1],
        relative_translations[:, 2]-margin))
    constraints = [LinearConstraint(depth_constraints, -relative_rotations[:, 2, 2], np.inf)]
    # SLSQP clips variable bounds at every evaluation. Representing the tiny
    # inverse-depth limit as a general inequality allowed steps across zero.
    bounds = Bounds([-np.inf, -np.inf, inverse_depth_min], np.inf)

    def pixel_errors(variables):
        coordinates = np.einsum('nij,j->ni', coordinate_jacobian, variables)+coordinate_constant
        z = coordinates[:, 2]
        x, y = (coordinates[:, :2]/z[:, None]).T
        fx, fy, cx, cy, k1, k2, p1, p2 = params.T
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2
        xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
        yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
        errors = np.column_stack((fx*xd+cx, fy*yd+cy))-observed
        return errors, x, y, z, r2, radial

    def objective_and_gradient(variables):
        # The smooth projection formula supports derivatives; final admission
        # still requires native finite projections with strictly positive depth.
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            errors, x, y, z, r2, radial = pixel_errors(variables)
            fx, fy, cx, cy, k1, k2, p1, p2 = params.T
            lengths = np.linalg.norm(errors, axis=1)
            cost = np.where(lengths <= 1, .5*lengths*lengths, lengths-.5).sum()
            derivative_r = k1+2*k2*r2
            drdx, drdy = 2*x*derivative_r, 2*y*derivative_r
            distortion = np.empty((len(z), 2, 2))
            distortion[:, 0, 0] = fx*(radial+x*drdx+2*p1*y+6*p2*x)
            distortion[:, 0, 1] = fx*(x*drdy+2*p1*x+2*p2*y)
            distortion[:, 1, 0] = fy*(y*drdx+2*p1*x+2*p2*y)
            distortion[:, 1, 1] = fy*(radial+y*drdy+6*p1*y+2*p2*x)
            perspective = np.zeros((len(z), 2, 3))
            perspective[:, 0, 0] = 1/z
            perspective[:, 1, 1] = 1/z
            perspective[:, 0, 2] = -x/z
            perspective[:, 1, 2] = -y/z
            world_jacobian = distortion @ perspective @ coordinate_jacobian
            gradient = np.einsum('npi,np->i', world_jacobian, errors/np.maximum(lengths, 1)[:, None])
            if not np.isfinite(cost) or not np.isfinite(gradient).all():
                raise ValueError('The actual pixel objective or derivative is nonfinite')
            return float(cost), gradient

    native_errors = []
    for entry in point.track.elements:
        image = model.images[entry.image_id]
        pose = image.cam_from_world()
        projected = model.cameras[image.camera_id].img_from_cam(
            pose.rotation.matrix() @ seed+pose.translation)
        native_errors.append(np.linalg.norm(projected-image.points2D[entry.point2D_idx].xy))
    if not np.isfinite(native_errors).all():
        raise ValueError('The observed mean-ray seed is not a valid native projection')
    native_world_seed_cost = np.where(np.asarray(native_errors) <= 1,
        .5*np.square(native_errors), np.asarray(native_errors)-.5).sum()
    seed_cost, analytic = objective_and_gradient(variable_seed)
    # OPENCV is homogeneous in camera coordinates. Compare the formula on
    # identical chart vectors; transforming a near-plane world seed separately
    # adds cancellation amplified by distortion. Final world pixels still use
    # the independent native readback and the original acceptance tolerances.
    native_seed_coordinates = np.einsum('nij,j->ni', coordinate_jacobian, variable_seed)+coordinate_constant
    native_cost = verify_native_pixel_formula(model, point_id, native_seed_coordinates, seed_cost)
    seed_errors = pixel_errors(variable_seed)[0]
    large = np.linalg.norm(seed_errors, axis=1) > 1
    def complex_cost(variables):
        errors = pixel_errors(variables)[0]
        squares = np.sum(errors*errors, axis=1)
        return .5*squares[~large].sum()+np.sqrt(squares[large]).sum()-.5*np.count_nonzero(large)
    derivative_error, derivative_checks = check_pixel_gradient(variable_seed, analytic,
        lambda variables: objective_and_gradient(variables)[0], complex_cost)
    # Scaling both the objective and its absolute tolerance keeps the same
    # pixel objective and stopping tolerance while conditioning the QP step.
    objective_scale = max(1., abs(seed_cost))
    def conditioned_objective(variables):
        value, gradient = objective_and_gradient(variables)
        return value/objective_scale, gradient/objective_scale
    solution = minimize(conditioned_objective, variable_seed, jac=True, constraints=constraints,
        bounds=bounds, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12/objective_scale})
    if not np.isfinite(solution.x).all() or solution.x[2] < inverse_depth_min:
        raise ValueError('The observed pixel solution violates its finite inverse-depth variable bound')
    final_cost, _ = objective_and_gradient(solution.x)
    xyz = anchor_center+anchor_rotation.T @ np.r_[solution.x[:2], 1]/solution.x[2]
    depths = matrix @ xyz+shift
    final_errors = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        entry = point.track.elements[index]
        image = model.images[entry.image_id]
        projection = model.cameras[image.camera_id].img_from_cam(rotation @ xyz+translation)
        if projection is None or not np.isfinite(projection).all():
            raise ValueError('The completed point has no finite positive native pixel projection')
        final_errors.append(float(np.linalg.norm(projection-observed[index])))
    final_errors = np.asarray(final_errors)
    native_final_cost = float(np.where(final_errors <= 1,
        .5*np.square(final_errors), final_errors-.5).sum())
    if not np.isclose(final_cost, native_final_cost, atol=1e-7, rtol=1e-10):
        raise ValueError('The completed point objective differs from actual native pixel projections')
    row = {'point3D_id': int(point_id), 'track_length': len(centers),
        'optimizer_success': bool(solution.success), 'optimizer_status': int(solution.status),
        'optimizer_message': str(solution.message), 'iterations': int(solution.nit),
        'minimum_depth_m': float(depths.min()), 'objective_value': final_cost,
        'objective': 'actual_OPENCV_HUBER_pixel_residuals_scale_1',
        'native_formula_equivalence_checked': True, 'analytic_derivative_error': derivative_error,
        'analytic_derivative_checks': derivative_checks,
        'actual_physical_seed_pixel_cost': seed_cost, 'seed_maximum_pixel_error': float(max(native_errors)),
        'native_world_seed_pixel_cost': float(native_world_seed_cost),
        'native_same_camera_vector_seed_pixel_cost': float(native_cost),
        'seed_world_to_chart_pixel_cost_difference': float(native_world_seed_cost-native_cost),
        'native_seed_calibration_checked_on_identical_camera_vectors': True,
        'positive_depth_seed_method': seed_method, 'seed_preparation_iterations': seed_iterations,
        'coordinate_chart': 'first_actual_observation_bearing_and_inverse_depth',
        'inverse_depth_minimum_is_numerical_stabilization_not_final_quality_gate': True,
        'inverse_depth_minimum': float(inverse_depth_min), 'optimized_chart_variables': solution.x.tolist(),
        'inverse_depth_variable_bound_checked': True,
        'solver_objective_scale': objective_scale, 'absolute_true_pixel_solver_tolerance': 1e-12,
        'final_native_pixel_cost': native_final_cost, 'final_native_formula_equivalence_checked': True,
        'old_xyz': point.xyz.tolist(), 'initialized_xyz': xyz.tolist(),
        'finite_positive_depth': bool(np.isfinite(xyz).all() and np.isfinite(depths).all()
            and np.all(depths > 0))}
    return xyz, row
