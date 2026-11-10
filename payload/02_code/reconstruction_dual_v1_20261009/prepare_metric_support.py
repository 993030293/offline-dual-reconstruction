"""Apply the existing training point and renderer support policies before BA."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import pycolmap

from common import atomic_json, read_json
from camera_support_recovery import model_identity
from camera_initialization_roles import point_support_roles, observation_decisions
from metric_point_policy import native_point_policy, point_policy_removals


def prepare(source, initial, output, scope, camera_role_result=None):
    model = pycolmap.Reconstruction(source)
    original_names = {image.name for image in model.images.values() if image.has_pose}
    seeded = None
    if camera_role_result is not None:
        role_path = Path(camera_role_result).resolve(strict=True)
        if not role_path.samefile(Path(source).parent / 'RESULT.json'):
            raise ValueError('Point support roles must come from this actual source model stage')
        seeded = read_json(role_path)
        if (not Path(seeded['model']).samefile(source)
                or seeded['output_identity'] != model_identity(source)):
            raise ValueError('The triangulated source or its typed support receipt changed')
    initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
    roles = point_support_roles(original_names, initial_names, seeded)
    initialization_roles = seeded['initialization_roles'] if seeded is not None else {
        'original_registered_names': sorted(original_names), 'required_names': roles['required_names'],
        'optional_bridge_names': roles['optional_names']}
    required = set(roles['required_names'])
    required_poses = {image.name: image.cam_from_world().matrix().copy() for image in model.images.values()
        if image.has_pose and image.name in required}
    before_points, before_observations = model.num_points3D(), model.compute_num_observations()
    policy_path, policy, camera_source, near_m = native_point_policy()
    raw = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
    lookup = {row['image_name']: row for row in raw if row['coarse_pose_valid'] and not row['excluded_role']
        and row['decision'] in ('eligible_not_selected', 'selected_initial_sfm_pending')}
    names = sorted(original_names & lookup.keys())
    alignment = pycolmap.align_reconstruction_to_locations(model, names,
        np.asarray([lookup[name]['C_W_m'] for name in names]), 3,
        pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0))
    if alignment is None:
        raise ValueError('No raw metric scale for the native renderer point policy')
    removals = point_policy_removals(model, alignment.scale,
        policy['minimum_point_track_length'], near_m, scope)
    for key in sorted(removals):
        model.delete_point3D(key)
    observations = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
    decisions = observation_decisions(observations, roles['required_names'], roles['optional_names'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'POINT_SUPPORT_DECISIONS.json', {'roles': roles, 'observations_after_filtering': observations,
        **decisions, 'minimum_point_track_length': policy['minimum_point_track_length'], 'native_near_m': near_m,
        'all_point_filter_thresholds_unchanged': True, 'not_a_completed_camera_pose_acceptance': True})
    if model.num_points3D() < policy['minimum_retained_sparse_points']:
        raise ValueError('Existing support filtering left insufficient actual sparse points')
    if decisions['unsupported_required_names']:
        raise ValueError('Existing support filtering leaves required cameras without observations: '
            + ', '.join(decisions['unsupported_required_names']))
    rejected_optional = set(decisions['rejected_optional_names']) & original_names
    for image in list(model.images.values()):
        if image.has_pose and image.name in rejected_optional:
            model.deregister_frame(image.frame_id)
    if any(image.has_pose and image.num_points3D == 0 for image in model.images.values()):
        raise ValueError('Existing support filtering leaves a registered camera without observations')
    if original_names - rejected_optional != {image.name for image in model.images.values() if image.has_pose}:
        raise ValueError('Point support preparation changed an undeclared camera denominator')
    if any(not np.array_equal(image.cam_from_world().matrix(), required_poses[image.name])
            for image in model.images.values() if image.has_pose and image.name in required):
        raise ValueError('Point support preparation changed a required camera pose')
    destination = output / 'models'
    destination.mkdir()
    model.write(destination)
    value = {'status': 'existing_point_policies_applied_BA_required', 'source_model': str(source),
        'source_identity': model_identity(source), 'model': str(destination), 'output_identity': model_identity(destination),
        'policy_path': str(policy_path), 'minimum_track': policy['minimum_point_track_length'],
        'pre_BA_scope': scope, 'training_track_gate_remains_required_at_export': True,
        'native_renderer_source': str(camera_source), 'native_near_m': near_m, 'metric_scale_for_near_check': float(alignment.scale),
        'before_points': before_points, 'before_observations': before_observations,
        'after_points': model.num_points3D(), 'after_observations': model.compute_num_observations(),
        'removed_points': list(removals.values()), 'registered_frames': model.num_reg_images(),
        'minimum_camera_observations_after': min(image.num_points3D for image in model.images.values() if image.has_pose),
        'all_original_cameras_and_poses_retained': not rejected_optional, 'source_map_preserved': True,
        'all_required_cameras_and_poses_retained': True, 'required_registered_frames': len(required),
        'camera_role_scope': roles['scope'], 'rejected_optional_bridge_names': sorted(rejected_optional),
        'initialization_roles': initialization_roles,
        'required_initial_names_all_registered': initial_names <= set(observations),
        'point_support_decisions': str(output / 'POINT_SUPPORT_DECISIONS.json'),
        'observation_identity_must_still_be_unchanged_during_subsequent_BA': True,
        'BA_projection_or_factor_guard_relaxed': False, 'not_a_completed_camera_pose_acceptance': True}
    atomic_json(output / 'RESULT.json', value)
    print(value['status'], before_points, model.num_points3D(), model.num_reg_images(), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scope', choices=('full-training-policy', 'renderer-only'), required=True)
    parser.add_argument('--camera-role-result', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('A point preparation attempt requires a unique output directory')
    try:
        prepare(args.source_model.resolve(strict=True), args.initial_frames.resolve(strict=True), output, args.scope,
            args.camera_role_result)
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'FAILURE.json', {'status': 'failed_pre_BA_point_preparation',
            'traceback': traceback.format_exc()})
        raise
