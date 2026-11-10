"""Rebuild genuine point tracks with known camera poses before joint optimization."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback

import pycolmap

from camera_initialization_roles import NATIVE_POINT_REBUILD_INPUT_STATUS, point_support_roles
from camera_required_relocalization import low_support_targets, run as relocalize_required
from camera_support_recovery import model_identity
from camera_visual_recovery import fixed_camera_invariants, required_native_support, required_registered_support
from common import atomic_json, read_json
from native_command import run_native


def file_identity(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def triangulation_options():
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads, options.random_seed = 8, 0
    options.fix_existing_frames = True
    options.mapper.fix_existing_frames = True
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_refine_sensor_from_rig = False
    options.max_runtime_seconds = 280
    options.ba_global_max_num_iterations = 60
    options.ba_global_max_refinements = 1
    if not options.check():
        raise ValueError('The installed native point triangulator rejects its options')
    return options


def run(source, initial, database_source, images, role_path, output):
    source, initial, database_source, images, role_path = (Path(value).resolve(strict=True)
        for value in (source, initial, database_source, images, role_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity, database_identity = model_identity(source), file_identity(database_source)
    started = time.monotonic()
    deadline = started + 360

    def remaining_budget():
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError('The complete point rebuilding and relocalization budget was exhausted')
        return remaining

    try:
        seeded = read_json(role_path)
        if (not role_path.samefile(source.parent / 'RESULT.json')
                or not source.samefile(seeded['model']) or seeded['output_identity'] != identity):
            raise ValueError('The point rebuild role receipt differs from the actual visual source')
        original = pycolmap.Reconstruction(source)
        registered = {image.name for image in original.images.values() if image.has_pose}
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        roles = point_support_roles(registered, initial_names, seeded, allow_pending_point_rebuild=True)
        if seeded['status'] == NATIVE_POINT_REBUILD_INPUT_STATUS:
            required_registered_support(original, roles['required_names'])
        database = output / 'database.db'
        with closing(sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True)) as incoming:
            with closing(sqlite3.connect(database)) as destination:
                incoming.backup(destination)
        models = output / 'models'
        options = triangulation_options()
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': identity,
            'source_database_identity': database_identity, 'fixed_actual_visual_camera_poses': True,
            'fixed_camera_scope': 'each_point_triangulation_input_only_relocalized_initializers_may_change',
            'all_points_rebuilt_from_actual_matches': True, 'clear_points': True,
            'joint_camera_optimization_claimed_by_triangulation_interface': False,
            'one_required_low_support_relocalization_and_rebuild_allowed': True,
            'bounded_native_feature_recovery_when_normal_registration_is_incomplete': True,
            'bounded_native_feature_recovery_when_registered_required_support_is_incomplete': True,
            'recovery_geometry_filtered_native_tracks_incrementally_retriangulated': True,
            'complete_execution_budget_seconds': 360,
            'final_required_native_minimum_unchanged': pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers,
            'options': json.loads(json.dumps(options.todict(), default=str)),
            'initialization_roles': seeded['initialization_roles'], 'training_admission_granted': False})
        command = [sys.executable, str(Path(__file__).resolve()), '--triangulate-only',
            '--source-model', source, '--database', database, '--images', images, '--output', models]
        seconds = run_native(command, output, 'triangulation', remaining_budget())
        actual = pycolmap.Reconstruction(models)
        first_delta = fixed_camera_invariants(original, actual)
        relocalized = None
        triangulation_input = original
        if low_support_targets(actual, roles['required_names']):
            relocalized = relocalize_required(models, roles['required_names'], database, images,
                output / 'required_support_relocalization', remaining_budget())
            triangulation_input = pycolmap.Reconstruction(relocalized['model'])
            mode = relocalized.get('point_rebuilding_mode', 'clear_all_points')
            if mode not in ('clear_all_points', 'native_incremental_preserve_geometry_filtered_tracks'):
                raise ValueError('Required relocalization returned an unknown point rebuilding role')
            database = Path(relocalized['database'])
            models = output / 'relocalized_point_models'
            retry_command = [sys.executable, str(Path(__file__).resolve()), '--triangulate-only',
                '--source-model', relocalized['model'], '--database', database,
                '--images', images, '--output', models]
            if mode == 'native_incremental_preserve_geometry_filtered_tracks':
                retry_command.append('--preserve-native-tracks')
            seconds += run_native(retry_command, output, 'relocalized_triangulation', remaining_budget())
            actual = pycolmap.Reconstruction(models)
        observations = required_native_support(actual, roles['required_names'])
        delta = max(first_delta, fixed_camera_invariants(triangulation_input, actual))
        source_delta = fixed_camera_invariants(original, actual, preserve_poses=False)
        if (set(observations) != registered or model_identity(source) != identity
                or file_identity(database_source) != database_identity):
            raise ValueError('Point rebuilding changed actual camera names or its source inputs')
        result = {'status': 'native_point_rebuilt_visual_support_ready_BA_REQUIRED',
            'source_model': str(source), 'source_identity': identity, 'model': str(models),
            'output_identity': model_identity(models), 'images': str(images), 'database': str(database),
            'initialization_roles': seeded['initialization_roles'], 'required_initial_names_all_registered': True,
            'registered_frames': actual.num_reg_images(), 'points': actual.num_points3D(),
            'observations': actual.compute_num_observations(), 'fixed_pose_maximum_component_delta': delta,
            'fixed_pose_check_scope': 'first_and_last_point_triangulation_inputs',
            'overall_source_to_output_pose_maximum_component_delta': source_delta,
            'full_required_native_point_support_verified': True,
            'required_native_minimum_unchanged': pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers,
            'minimum_required_support_after_rebuilding': min(observations[name] for name in roles['required_names']),
            'required_low_support_relocalization_completed': relocalized is not None,
            'relocalized_point_rebuilding_mode': mode if relocalized is not None else None,
            'required_support_relocalization': relocalized,
            'point_rebuilding_seconds': seconds, 'elapsed_seconds': time.monotonic()-started,
            'explicit_joint_camera_BA_still_required': True, 'training_admission_granted': False,
            'runtime_language_model_used': False}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_actual_point_rebuilding',
            'traceback': traceback.format_exc(), 'source_model_unchanged': model_identity(source) == identity})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'database', 'images', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--initial-frames', type=Path)
    parser.add_argument('--camera-role-result', type=Path)
    parser.add_argument('--triangulate-only', action='store_true')
    parser.add_argument('--preserve-native-tracks', action='store_true')
    args = parser.parse_args()
    if args.triangulate_only:
        args.output.mkdir(parents=True, exist_ok=False)
        model = pycolmap.Reconstruction(args.source_model)
        if args.preserve_native_tracks:
            from camera_required_feature_recovery import verify_feature_binding
            verify_feature_binding(model, args.database)
            native = pycolmap.IncrementalMapperOptions()
            pycolmap.ObservationManager(model).filter_all_points3D(
                native.filter_max_reproj_error, native.filter_min_tri_angle)
        pycolmap.triangulate_points(model, args.database, args.images,
            args.output, clear_points=not args.preserve_native_tracks,
            options=triangulation_options(), refine_intrinsics=False)
    else:
        if args.preserve_native_tracks:
            parser.error('Native track preservation is only a bounded triangulation worker role')
        if args.initial_frames is None or args.camera_role_result is None:
            parser.error('The bounded point rebuild requires actual frame and camera role inputs')
        result = run(args.source_model, args.initial_frames, args.database, args.images,
            args.camera_role_result, args.output)
        print(result['status'], result['registered_frames'], flush=True)
