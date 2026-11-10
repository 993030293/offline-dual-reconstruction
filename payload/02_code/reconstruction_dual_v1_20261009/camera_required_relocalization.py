"""Relocalize required low-support cameras without freezing their stale poses."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from camera_support_recovery import model_identity
from camera_required_feature_recovery import run as recover_required_features, verify_feature_binding
from camera_visual_recovery import fixed_camera_invariants, required_registered_support
from common import COLMAP, atomic_json
from native_command import run_native


def low_support_targets(model, required):
    """Select by actual same-run support, never by object or image identity."""
    required = set(required)
    observations = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
    if not required or not required <= observations.keys():
        raise ValueError('Required relocalization cannot change the registered camera denominator')
    for image in model.images.values():
        if image.has_pose and not np.isfinite(image.cam_from_world().matrix()).all():
            raise ValueError('Required relocalization cannot use a nonfinite camera pose')
    minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
    return {name: observations[name] for name in sorted(required) if observations[name] < minimum}


def run(source, required, database_source, images, output, maximum_seconds):
    source, database_source, images = (Path(value).resolve(strict=True)
        for value in (source, database_source, images))
    output = Path(output).resolve()
    if maximum_seconds <= 10:
        raise TimeoutError('No bounded native relocalization budget remains')
    output.mkdir(parents=True, exist_ok=False)
    identity = model_identity(source)
    started = time.monotonic()
    try:
        original = pycolmap.Reconstruction(source)
        targets = low_support_targets(original, required)
        if not targets:
            raise ValueError('Required native relocalization needs an actual low-support target')
        names = {image.name for image in original.images.values() if image.has_pose}
        anchors = pycolmap.Reconstruction(original)
        for image in list(anchors.images.values()):
            if image.has_pose and image.name in targets:
                anchors.deregister_frame(image.frame_id)
        if anchors.num_reg_images() < 2:
            raise ValueError('Required relocalization lacks sufficient actual fixed visual anchors')
        if any(not (images / name).is_file() for name in names):
            raise FileNotFoundError('A relocalization camera lacks its actual same-run image')
        seed = output / 'trusted_anchors'
        seed.mkdir()
        anchors.write(seed)
        database = output / 'database.db'
        with closing(sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True)) as incoming:
            if not names <= {row[0] for row in incoming.execute('SELECT name FROM images')}:
                raise ValueError('A relocalization camera lacks actual database features')
            with closing(sqlite3.connect(database)) as destination:
                incoming.backup(destination)
        allowed = output / 'all_original_images.txt'
        allowed.write_text('\n'.join(sorted(names)) + '\n', encoding='utf-8')
        models = output / 'models'
        models.mkdir()
        remaining = maximum_seconds - (time.monotonic() - started)
        if remaining <= 10:
            raise TimeoutError('Native relocalization preparation exhausted its remaining budget')
        command = [COLMAP, 'mapper', '--database_path', database, '--image_path', images,
            '--input_path', seed, '--output_path', models, '--Mapper.image_list_path', allowed,
            '--Mapper.fix_existing_frames', '1', '--Mapper.num_threads', '8', '--Mapper.random_seed', '0',
            '--default_random_seed', '0', '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0', '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.ba_refine_sensor_from_rig', '0', '--Mapper.max_runtime_seconds',
            str(max(1, min(570, int(remaining) - 10))),
            '--Mapper.ba_global_max_num_iterations', '1', '--Mapper.ba_global_max_refinements', '1']
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': identity,
            'required_names': sorted(required), 'all_original_registered_names': sorted(names),
            'targets_computed_from_actual_required_support': targets,
            'original_native_minimum': pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers,
            'targets_have_no_fixed_pose_anchor': True, 'all_other_registered_anchors_fixed': True,
            'native_pose_and_geometry_gates_unchanged': True, 'maximum_seconds': maximum_seconds,
            'one_bounded_relocalization_only': True, 'command': list(map(str, command)),
            'one_feature_refinement_if_registration_or_native_support_incomplete': True,
            'ordinary_relocalization_tracks_filtered_and_bound_before_incremental_rebuild': True,
            'full_point_rebuild_and_final_support_still_required': True,
            'training_admission_granted': False, 'runtime_language_model_used': False})
        seconds = run_native(command, output, 'mapping', remaining)
        actual = pycolmap.Reconstruction(models)
        actual_names = {image.name for image in actual.images.values() if image.has_pose}
        if actual_names-names:
            raise ValueError('Required native relocalization registered an unexpected camera')
        if actual_names != names:
            atomic_json(output / 'MAPPING_READBACK.json', {'status': 'native_command_completed_required_registration_incomplete',
                'native_command_returned_success': True, 'missing_original_names': sorted(names-actual_names),
                'mapping_seconds': seconds, 'training_admission_granted': False})
            recovered = recover_required_features(source, required, database_source, images,
                output / 'native_feature_recovery', maximum_seconds-(time.monotonic()-started), targets)
            recovered['ordinary_mapping_readback'] = str(output / 'MAPPING_READBACK.json')
            recovered['ordinary_mapping_seconds'] = seconds
            atomic_json(output / 'RESULT.json', recovered)
            return recovered
        fixed_delta = fixed_camera_invariants(anchors, actual)
        if model_identity(source) != identity:
            raise ValueError('Native required relocalization changed its source or camera denominator')
        drift = fixed_camera_invariants(original, actual, preserve_poses=False)
        bindings = verify_feature_binding(actual, database)
        native = pycolmap.IncrementalMapperOptions()
        removed = pycolmap.ObservationManager(actual).filter_all_points3D(
            native.filter_max_reproj_error, native.filter_min_tri_angle)
        # A fresh clear-points pass can lose valid tracks recovered by the mapper.
        actual.write(models)
        remaining_targets = low_support_targets(actual, required)
        if remaining_targets:
            atomic_json(output / 'MAPPING_READBACK.json', {
                'status': 'native_command_completed_registration_complete_native_support_incomplete',
                'native_command_returned_success': True, 'missing_original_names': [],
                'registered_frames': actual.num_reg_images(),
                'required_names': sorted(required),
                'below_native_minimum': remaining_targets,
                'native_filtered_observations': int(removed),
                'actual_database_keypoint_bindings_checked': bindings,
                'original_native_minimum': pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers,
                'mapping_seconds': seconds, 'training_admission_granted': False})
            # Use the current mapper's geometry and database, including newly weak cameras.
            recovered = recover_required_features(models, required, database, images,
                output / 'native_feature_recovery', maximum_seconds-(time.monotonic()-started), remaining_targets)
            if model_identity(source) != identity:
                raise ValueError('Native feature refinement changed the original relocalization source')
            recovered.update(ordinary_mapping_readback=str(output / 'MAPPING_READBACK.json'),
                ordinary_mapping_seconds=seconds, ordinary_relocalization_source_model=str(source),
                ordinary_relocalization_source_identity=identity,
                ordinary_relocalization_targets_before=targets,
                ordinary_remaining_anchor_maximum_component_delta=fixed_delta,
                ordinary_actual_pose_maximum_component_delta=drift)
            atomic_json(output / 'RESULT.json', recovered)
            return recovered
        observations = required_registered_support(actual, required)
        result = {'status': 'native_required_camera_relocalized_point_rebuild_REQUIRED',
            'source_model': str(source), 'source_identity': identity, 'model': str(models),
            'output_identity': model_identity(models), 'database': str(database), 'images': str(images),
            'registered_frames': actual.num_reg_images(), 'targets_before': targets,
            'target_support_before_mandatory_rebuild': {name: observations[name] for name in targets},
            'remaining_anchor_maximum_component_delta': fixed_delta,
            'actual_pose_maximum_component_delta': drift, 'mapping_seconds': seconds,
            'native_filtered_observations': int(removed),
            'actual_database_keypoint_bindings_checked': bindings,
            'point_rebuilding_mode': 'native_incremental_preserve_geometry_filtered_tracks',
            'elapsed_seconds': time.monotonic()-started, 'full_required_point_support_pending': True,
            'training_admission_granted': False, 'runtime_language_model_used': False}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_required_native_relocalization',
            'traceback': traceback.format_exc(), 'source_model_unchanged': model_identity(source) == identity})
        raise
