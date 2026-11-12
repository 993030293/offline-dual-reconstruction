"""Optimize actual free camera and point blocks before freezing metric comparison."""

import argparse
import importlib.util
from pathlib import Path
import time
import traceback

import numpy as np
import pycolmap

from camera_initialization_roles import observation_decisions, point_support_roles
from camera_support_recovery import model_identity
from camera_visual_recovery import fixed_camera_invariants
from common import ROOT, atomic_json, read_json


NATIVE_JOINT_BA_ITERATIONS = 60
NATIVE_JOINT_BA_SOFT_SECONDS = 180
# Native solver teardown, cold model serialization, filtering, and receipt publication
# happen after the solver's soft limit and require their own bounded watchdog window.
NATIVE_JOINT_STAGE_HARD_SECONDS = 4 * NATIVE_JOINT_BA_SOFT_SECONDS


def load_helpers():
    path = ROOT / '02_code/box2_lio_guided_sfm_v1_20261002/common.py'
    spec = importlib.util.spec_from_file_location('dual_native_visual_joint_helpers', path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def joint_problem(model, helpers, iterations=NATIVE_JOINT_BA_ITERATIONS,
        solver_seconds=NATIVE_JOINT_BA_SOFT_SECONDS):
    config = helpers.ba_config(model)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
    options = helpers.ba_options(iterations)
    options.ceres.solver_options.max_solver_time_in_seconds = solver_seconds
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, model)
    variables = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        pose = model.frames[image.frame_id].rig_from_world.params
        if (not adjuster.problem.has_parameter_block(pose)
                or adjuster.problem.parameter_block_size(pose) != 7
                or adjuster.problem.is_parameter_block_constant(pose)):
            raise ValueError('An intended camera is absent or fixed in the actual native joint problem')
        variables.append({'image_name': image.name, 'variable_native_pose_block': True})
    residuals = int(config.num_residuals(model))
    if residuals != 2*model.compute_num_observations() or not variables:
        raise ValueError('The actual joint problem lacks the complete visual observation factors')
    return adjuster, variables, residuals


def run(source, initial, role_path, output):
    source, initial, role_path = (Path(value).resolve(strict=True) for value in (source, initial, role_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = model_identity(source)
    started = time.monotonic()
    try:
        seeded = read_json(role_path)
        if (not role_path.samefile(source.parent / 'RESULT.json')
                or not source.samefile(seeded['model']) or seeded['output_identity'] != identity):
            raise ValueError('The joint conditioning source or its typed role receipt changed')
        model = pycolmap.Reconstruction(source)
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        before = pycolmap.Reconstruction(model)
        original_names = {image.name for image in model.images.values() if image.has_pose}
        roles = point_support_roles(original_names, initial_names, seeded)
        helpers = load_helpers()
        adjuster, variables, residuals = joint_problem(model, helpers)
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': identity,
            'initialization_roles': seeded['initialization_roles'],
            'native_joint_BA_iterations': NATIVE_JOINT_BA_ITERATIONS,
            'native_joint_BA_soft_seconds': NATIVE_JOINT_BA_SOFT_SECONDS,
            'native_joint_stage_hard_seconds': NATIVE_JOINT_STAGE_HARD_SECONDS,
            'fixed_intrinsics': True,
            'native_all_map_geometry_filter_before_metric_input_freeze': True,
            'previous_failure_models_or_point_lists_read': False,
            'final_common_metric_reference_and_guards_unchanged': True,
            'training_admission_granted': False})
        atomic_json(output / 'ACTUAL_FREE_CAMERA_BLOCKS.json', {'rows': variables,
            'expected_visual_residuals': residuals, 'fixed_intrinsics': True})
        solution = helpers.solver_receipt(adjuster.solve())
        before_filter = output / 'unaccepted_native_joint_model_before_filter'
        before_filter.mkdir()
        model.write(before_filter)
        atomic_json(output / 'NATIVE_SOLVER.json', helpers.serializable(solution))
        if not solution['usable'] or solution['num_residuals'] != residuals:
            raise ValueError('Actual native joint optimization lacks usable complete visual factor activation')
        mapper = pycolmap.IncrementalMapperOptions()
        points_before, observations_before = model.num_points3D(), model.compute_num_observations()
        filtered = pycolmap.ObservationManager(model).filter_all_points3D(
            mapper.filter_max_reproj_error, mapper.filter_min_tri_angle)
        observations = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
        decisions = observation_decisions(observations, roles['required_names'], roles['optional_names'])
        atomic_json(output / 'POINT_SUPPORT_DECISIONS.json', {**decisions,
            'roles': roles, 'observations_after_filtering': observations})
        if decisions['unsupported_required_names']:
            raise ValueError('Native joint filtering leaves required cameras without actual observations: '
                + ', '.join(decisions['unsupported_required_names']))
        for image in list(model.images.values()):
            if image.has_pose and image.name in decisions['rejected_optional_names']:
                model.deregister_frame(image.frame_id)
        rejected = set(decisions['rejected_optional_names']) & original_names
        if original_names-rejected != {image.name for image in model.images.values() if image.has_pose}:
            raise ValueError('Native joint filtering changed an undeclared camera denominator')
        # A converged input may remain stationary; actual free blocks and factors prove joint participation.
        retained_before = pycolmap.Reconstruction(before)
        for image in list(retained_before.images.values()):
            if image.has_pose and image.name in rejected:
                retained_before.deregister_frame(image.frame_id)
        drift = fixed_camera_invariants(retained_before, model, preserve_poses=False)
        if not np.isfinite(drift) or model_identity(source) != identity:
            raise ValueError('Native joint conditioning changed its source or produced nonfinite camera poses')
        models = output / 'models'
        models.mkdir()
        model.write(models)
        result = {'status': 'native_joint_visual_conditioned_support_ready_metric_BA_REQUIRED',
            'source_model': str(source), 'source_identity': identity, 'model': str(models),
            'output_identity': model_identity(models), 'initialization_roles': seeded['initialization_roles'],
            'required_initial_names_all_registered': True, 'registered_frames': model.num_reg_images(),
            'points': model.num_points3D(), 'observations': model.compute_num_observations(),
            'points_before_native_filter': points_before, 'observations_before_native_filter': observations_before,
            'native_filtered_observations': int(filtered),
            'native_filter_max_error_px': mapper.filter_max_reproj_error,
            'native_filter_minimum_angle_degrees': mapper.filter_min_tri_angle,
            'actual_variable_camera_blocks': len(variables), 'actual_pose_maximum_component_delta': drift,
            'required_below_original_PnP_count': {name: observations[name] for name in roles['required_names']
                if observations[name] < mapper.abs_pose_min_num_inliers},
            'rejected_optional_bridge_names': sorted(rejected), 'elapsed_seconds': time.monotonic()-started,
            'full_metric_evaluation_still_required': True, 'training_admission_granted': False,
            'runtime_language_model_used': False, 'manual_geometry_adoption_allowed': False}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_actual_joint_visual_conditioning',
            'traceback': traceback.format_exc(), 'source_model_unchanged': model_identity(source) == identity})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'camera-role-result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_model, args.initial_frames, args.camera_role_result, args.output)
    print(result['status'], result['registered_frames'], flush=True)
