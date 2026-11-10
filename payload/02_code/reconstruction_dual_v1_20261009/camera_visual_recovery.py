"""Restore actual visual registration from trusted same-run anchors and matches."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from camera_initialization_roles import NATIVE_POINT_REBUILD_INPUT_STATUS
from camera_support_recovery import model_identity
from common import COLMAP, atomic_json, read_json
from native_command import run_native


def qualified_coarse_names(initial):
    rows = (Path(initial) / 'frame_decisions.jsonl').read_text(encoding='utf-8').splitlines()
    return {row['image_name'] for row in map(json.loads, rows) if row['coarse_pose_valid']
        and not row['excluded_role']
        and row['decision'] in ('eligible_not_selected', 'selected_initial_sfm_pending')}


def reestimate_unbound_anchors(model, qualified):
    """Keep cameras without qualified priors required, without fixing stale poses."""
    names = sorted(image.name for image in model.images.values()
        if image.has_pose and image.name not in qualified)
    for image in list(model.images.values()):
        if image.has_pose and image.name in names:
            model.deregister_frame(image.frame_id)
    return names


def fixed_camera_invariants(before, after, preserve_poses=True):
    for key, camera in before.cameras.items():
        actual = after.cameras[key]
        if (camera.model != actual.model or camera.width != actual.width or camera.height != actual.height
                or not np.array_equal(camera.params, actual.params)):
            raise ValueError('Native visual recovery changed fixed optical calibration')
    delta = 0.0
    for image in before.images.values():
        if not image.has_pose:
            continue
        actual = after.images[image.image_id]
        if not actual.has_pose or actual.name != image.name:
            raise ValueError('Native visual recovery lost an actual trusted camera')
        delta = max(delta, float(np.max(np.abs(image.cam_from_world().matrix()-actual.cam_from_world().matrix()))))
    if preserve_poses and delta > 1e-9:
        raise ValueError('Native visual recovery moved fixed trusted camera anchors')
    return delta


def required_native_support(model, required):
    observations = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
    minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
    missing = sorted(set(required)-observations.keys())
    below = {name: observations[name] for name in sorted(set(required) & observations.keys())
        if observations[name] < minimum}
    if missing or below:
        raise ValueError('Native visual recovery lacks required genuine camera support: '
            + json.dumps({'missing': missing, 'below_native_minimum': below}))
    return observations


def required_registered_support(model, required):
    """Keep filtered registration provisional until the mandatory all-point rebuild."""
    required = set(required)
    observations = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
    missing = sorted(required-observations.keys())
    zero = sorted(name for name in required & observations.keys() if observations[name] == 0)
    if missing or zero:
        raise ValueError('Native registration lacks required nonzero visual support: '
            + json.dumps({'missing': missing, 'zero_observations': zero}))
    for image in model.images.values():
        if image.name in required and not np.isfinite(image.cam_from_world().matrix()).all():
            raise ValueError('A required native registration pose is nonfinite')
    return observations


def run(source, initial, database_source, images, output, bridge=None):
    source, initial, database_source, images = (Path(value).resolve(strict=True)
        for value in (source, initial, database_source, images))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = model_identity(source)
    started = time.monotonic()
    deadline = started + 600

    def remaining():
        value = deadline-time.monotonic()
        if value <= 1:
            raise TimeoutError('Native visual recovery exhausted its original shared execution budget')
        return value

    try:
        original = pycolmap.Reconstruction(source)
        original_names = {image.name for image in original.images.values() if image.has_pose}
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        required = original_names | initial_names
        optional = set()
        anchors = pycolmap.Reconstruction(original)
        if bridge is not None:
            bridge = Path(bridge).resolve(strict=True)
            contract, result = read_json(bridge / 'CONTRACT.json'), read_json(bridge / 'RESULT.json')
            if (result['status'] != 'raw_quality_bridges_matched_geometry_pending'
                    or result['source_identity'] != identity or contract['source_identity'] != identity
                    or not database_source.samefile(result['database']) or not images.samefile(result['images'])):
                raise ValueError('Native visual recovery bridge lacks actual same-run source binding')
            optional = {row['image_name'] for row in contract['additions']}
            anchors = pycolmap.Reconstruction(bridge / 'trusted_source')
            if optional & required:
                raise ValueError('A raw quality witness cannot replace a required camera')
        reestimated = reestimate_unbound_anchors(anchors, qualified_coarse_names(initial))
        if anchors.num_reg_images() < 2:
            raise ValueError('Native visual recovery requires actual connected visual anchors')
        allowed_names = required | optional
        if any(not (images / name).is_file() for name in allowed_names):
            raise FileNotFoundError('A required genuine native visual image is absent')
        mapper_source = output / 'trusted_anchors'
        mapper_source.mkdir()
        anchors.write(mapper_source)
        database = output / 'database.db'
        with closing(sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True)) as incoming:
            if not allowed_names <= {row[0] for row in incoming.execute('SELECT name FROM images')}:
                raise ValueError('A required native visual image lacks actual extracted features')
            with closing(sqlite3.connect(database)) as destination:
                incoming.backup(destination)
        allowed = output / 'required_and_bridge_images.txt'
        allowed.write_text('\n'.join(sorted(allowed_names))+'\n', encoding='utf-8')
        models = output / 'models'
        models.mkdir()
        command = [COLMAP, 'mapper', '--database_path', database, '--image_path', images,
            '--input_path', mapper_source, '--output_path', models, '--Mapper.image_list_path', allowed,
            '--Mapper.fix_existing_frames', '1', '--Mapper.num_threads', '8', '--Mapper.random_seed', '0',
            '--default_random_seed', '0', '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0', '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.ba_refine_sensor_from_rig', '0', '--Mapper.max_runtime_seconds', '570',
            '--Mapper.ba_global_max_num_iterations', '1', '--Mapper.ba_global_max_refinements', '1']
        roles = {'original_registered_names': sorted(original_names), 'required_names': sorted(required),
            'optional_bridge_names': sorted(optional)}
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': identity,
            'initialization_roles': roles, 'reestimated_unbound_anchor_names': reestimated,
            'qualified_coarse_poses_used_only_to_identify_unbound_anchors': True,
            'coarse_poses_inserted_as_verified_registration': False,
            'native_registration_local_BA_and_triangulation_gates_unchanged': True,
            'native_required_point_support_checked_after_mandatory_all_point_rebuild': True,
            'initializer_global_BA_iterations': 1, 'full_joint_and_metric_BA_still_required': True,
            'command': list(map(str, command)), 'training_admission_granted': False})
        seconds = run_native(command, output, 'mapping', remaining())
        actual = pycolmap.Reconstruction(models)
        delta = fixed_camera_invariants(anchors, actual)
        registered = {image.name for image in actual.images.values() if image.has_pose}
        if registered - allowed_names:
            raise ValueError('Native visual recovery registered an undeclared camera')
        missing = sorted(required-registered)
        atomic_json(output / 'MAPPING_READBACK.json', {
            'status': 'native_mapper_completed_registration_readback',
            'native_registered_frames': len(registered), 'missing_required_names': missing,
            'trusted_pose_maximum_component_delta': delta,
            'ordinary_native_model': str(models), 'ordinary_native_database': str(database),
            'remaining_original_execution_budget_seconds': remaining(),
            'training_admission_granted': False})
        feature_recovery = None
        if missing:
            # Import lazily because feature recovery shares this module's geometry gates.
            from camera_required_feature_recovery import run as recover_required_features
            feature_recovery = recover_required_features(models, required, database, images,
                output / 'missing_required_feature_recovery', remaining(),
                {name: 0 for name in missing}, target_scope='missing_required')
            database = Path(feature_recovery['database'])
            actual = pycolmap.Reconstruction(feature_recovery['model'])
            # The downstream role receipt belongs to this stage, while helper evidence remains private.
            models = output / 'feature_recovered_models'
            models.mkdir()
            actual.write(models)
            remaining()
        observations = required_registered_support(actual, required)
        delta = fixed_camera_invariants(anchors, actual)
        if model_identity(source) != identity:
            raise ValueError('Native visual recovery changed its source model')
        if set(observations) - allowed_names:
            raise ValueError('Native visual recovery registered an undeclared camera')
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        result = {'status': NATIVE_POINT_REBUILD_INPUT_STATUS,
            'source_model': str(source), 'source_identity': identity, 'model': str(models),
            'output_identity': model_identity(models), 'database': str(database), 'images': str(images),
            'initialization_roles': roles, 'required_initial_names_all_registered': True,
            'registered_frames': actual.num_reg_images(), 'points': actual.num_points3D(),
            'observations': actual.compute_num_observations(), 'trusted_pose_maximum_component_delta': delta,
            'full_native_point_support_pending': True, 'required_native_minimum_unchanged': minimum,
            'before_rebuild_below_native_minimum': {
                name: observations[name] for name in sorted(required) if observations[name] < minimum},
            'reestimated_unbound_anchor_names': reestimated, 'mapping_seconds': seconds,
            'missing_required_feature_recovery': feature_recovery,
            'maximum_execution_seconds_shared_by_mapping_and_feature_recovery': 600,
            'elapsed_seconds': time.monotonic()-started, 'training_admission_granted': False,
            'runtime_language_model_used': False, 'manual_geometry_adoption_allowed': False}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_native_visual_recovery',
            'traceback': traceback.format_exc(), 'source_model_unchanged': model_identity(source) == identity})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'database', 'images', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bridge', type=Path)
    args = parser.parse_args()
    result = run(args.source_model, args.initial_frames, args.database, args.images, args.output, args.bridge)
    print(result['status'], result['registered_frames'], flush=True)
