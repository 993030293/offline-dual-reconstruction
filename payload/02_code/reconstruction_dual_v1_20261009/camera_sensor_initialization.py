"""Use qualified coarse poses only as initialization for genuine triangulation and BA."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from common import atomic_json, read_json
from camera_support_recovery import model_identity, coarse_consistency
from camera_initialization_roles import initialization_roles, observation_decisions, SENSOR_INITIALIZATION_PROTOCOL
from metric_point_policy import native_point_policy, point_policy_removals, qualified_seed_observations


def run(source, initial, bridge, output, verification='pnp-support', pnp_recovery_input=None):
    source, initial, bridge = (Path(value).resolve(strict=True) for value in (source, initial, bridge))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        original = pycolmap.Reconstruction(source)
        model = pycolmap.Reconstruction(original)
        rows = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
        lookup = {row['image_name']: row for row in rows if row['coarse_pose_valid'] and not row['excluded_role']
            and row['decision'] in ('eligible_not_selected', 'selected_initial_sfm_pending')}
        names = sorted(image.name for image in original.images.values() if image.has_pose and image.name in lookup)
        alignment = pycolmap.align_reconstruction_to_locations(original, names,
            np.asarray([lookup[name]['C_W_m'] for name in names]), 3,
            pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0))
        if alignment is None:
            raise ValueError('No fresh metric transform for coarse initialization')
        bridge_contract = read_json(bridge / 'CONTRACT.json')
        additions = {row['image_name'] for row in bridge_contract['additions']}
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        policy_path, policy, renderer_source, near_m = native_point_policy()
        seed_removals = point_policy_removals(original, alignment.scale,
            policy['minimum_point_track_length'], near_m)
        seed_support = qualified_seed_observations(original, seed_removals)
        conflicted, partial = set(), None
        if pnp_recovery_input is not None:
            partial = read_json(pnp_recovery_input)
            if (partial['status'] != 'same_run_partial_PnP_recovery_input_BA_required'
                    or Path(partial['model']).resolve(strict=True) != source
                    or partial['output_identity'] != model_identity(source)
                    or partial['initial_manifest_identity'] != hashlib.sha256(
                        (initial / 'INITIAL_FRAME_MANIFEST.json').read_bytes()).hexdigest()
                    or partial['training_admission_granted'] is not False):
                raise ValueError('Coarse seed roles require the same-run typed partial PnP input')
            conflicts = coarse_consistency(original, initial)
            conflicted = {name for name in partial['unverified_registered_names'] if conflicts.get(name, 0) > .6}
        roles = initialization_roles({image.name: image.num_points3D for image in original.images.values()
            if image.has_pose}, initial_names, additions, lookup.keys(), seed_support, conflicted)
        if partial and set(partial['verified_PnP_names']) & set(roles['coarse_initializer_names']):
            raise ValueError('Coarse initialization would undo a genuine verified PnP update')
        initialize = set(roles['coarse_initializer_names'])
        if set(bridge_contract.get('required_initial_recovery_names', [])) != set(roles['missing_initial_names']):
            raise ValueError('The bridge omitted or changed required initially selected recovery frames')
        database = output / 'database.db'
        with sqlite3.connect('file:' + (bridge / 'database.db').as_posix() + '?mode=ro', uri=True) as incoming:
            with sqlite3.connect(database) as destination:
                incoming.backup(destination)
        with pycolmap.Database.open(database) as db:
            frames = {int(data.id): frame for frame in db.read_all_frames() for data in frame.data_ids}
            images = {image.name: image for image in db.read_all_images()}
            for name in sorted(initialize):
                image = images[name]
                frame = frames[image.image_id]
                if image.image_id in model.images:
                    if model.images[image.image_id].has_pose:
                        model.deregister_frame(frame.frame_id)
                else:
                    model.add_frame(frame)
                    local = pycolmap.Image(name=name, keypoints=np.asarray(db.read_keypoints(image.image_id)[:, :2], np.float64),
                        camera_id=image.camera_id, image_id=image.image_id)
                    local.frame_id = frame.frame_id
                    model.add_image(local)
                raw = lookup[name]
                center = alignment.rotation.matrix().T @ (np.asarray(raw['C_W_m']) - alignment.translation) / alignment.scale
                camera_rotation = (alignment.rotation.matrix().T @ np.asarray(raw['R_WC'])).T
                pose = pycolmap.Rigid3d(pycolmap.Rotation3d(camera_rotation), -camera_rotation @ center)
                model.frames[frame.frame_id].set_cam_from_world(image.camera_id, pose)
                model.register_frame(frame.frame_id)
        seed = output / 'UNVERIFIED_COARSE_INITIALIZATION'
        seed.mkdir()
        model.write(seed)
        source_roles = dict(Counter(lookup[name].get('source_role', 'unconfirmed') for name in initialize))
        producers = dict(Counter(lookup[name].get('pose_producer', 'unconfirmed') for name in initialize))
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': model_identity(source),
            'initial_frames': str(initial), 'bridge': str(bridge), 'coarse_initialization_names': sorted(initialize),
            'initialization_protocol': SENSOR_INITIALIZATION_PROTOCOL,
            'initialization_roles': roles,
            'same_run_partial_PnP_input': str(pnp_recovery_input) if pnp_recovery_input else None,
            'verified_PnP_updates_retained': partial['verified_PnP_names'] if partial else [],
            'observed_visual_poses_retained_as_initializers': not (
                roles['degenerate_observed_original_names'] or roles['unverified_conflicted_original_names']),
            'supported_visual_poses_retained_as_initializers': not roles['unverified_conflicted_original_names'],
            'qualified_seed_observations': seed_support,
            'qualification_policy_path': str(policy_path), 'native_renderer_source': str(renderer_source),
            'seed_qualification_minimum_point_track': policy['minimum_point_track_length'],
            'seed_qualification_native_near_m': near_m,
            'zero_physical_support_initializer_requires_real_triangulation_and_joint_BA': True,
            'coarse_pose_role': 'unverified_initializer_only_not_a_visual_registration_or_deliverable',
            'all_original_frames_required': True, 'native_triangulation_parameters_unchanged': True,
            'recorded_raw_position_and_orientation_used': 'recorded_raw_odometry_coarse_estimate' in source_roles,
            'fresh_generated_position_and_orientation_used': 'fresh_raw_LiDAR_IMU_coarse_estimate' in source_roles,
            'coarse_initializer_source_role_counts': source_roles, 'coarse_initializer_producer_counts': producers,
            'coarse_source_accuracy_independently_verified': False, 'GT_used': False,
            'metric_BA_must_really_optimize_and_verify_every_factor_before_training': True,
            'pure_similarity_recovery_of_coarse_initialized_cameras_allowed': False})
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads, options.random_seed = 8, 0
        options.fix_existing_frames, options.mapper.fix_existing_frames = True, True
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.ba_refine_sensor_from_rig = False
        options.max_runtime_seconds = 280
        destination = output / 'models'
        destination.mkdir()
        pycolmap.triangulate_points(model, database, bridge / 'images', destination,
            clear_points=False, options=options, refine_intrinsics=False)
        model = pycolmap.Reconstruction(destination)
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        support = [{'image_name': image.name, 'observations': image.num_points3D,
            'passed_native_support_count': image.num_points3D >= minimum}
            for image in model.images.values() if image.has_pose and image.name in initialize]
        atomic_json(output / 'SUPPORT.json', {'minimum_observations': minimum, 'rows': support,
            'not_a_completed_visual_pose_verification': True})
        original_names = {image.name for image in original.images.values() if image.has_pose}
        if not original_names <= {image.name for image in model.images.values() if image.has_pose}:
            raise ValueError('Coarse-initialized triangulation lost an original camera')
        observed = {image.name: image.num_points3D for image in model.images.values() if image.has_pose}
        decisions = observation_decisions(observed, roles['required_names'], additions)
        atomic_json(output / 'OBSERVATION_DECISIONS.json', decisions)
        for image in list(model.images.values()):
            if image.has_pose and image.name in decisions['rejected_optional_names']:
                model.deregister_frame(image.frame_id)
        if decisions['unsupported_required_names']:
            raise ValueError('Joint BA initialization leaves required cameras without genuine visual observations: '
                + ', '.join(decisions['unsupported_required_names']))
        failed = [image.name for image in model.images.values() if image.has_pose
            and image.name in roles['required_names'] and image.num_points3D < minimum]
        if failed and verification == 'pnp-support':
            raise ValueError('Genuine triangulation still cannot support original cameras: ' + ', '.join(failed))
        rejected_additions = list(decisions['rejected_optional_names'])
        for image in list(model.images.values()):
            if verification == 'pnp-support' and image.name in additions and image.num_points3D < minimum:
                rejected_additions.append(image.name)
                model.deregister_frame(image.frame_id)
        filtered = output / 'supported_models_BA_REQUIRED'
        filtered.mkdir()
        model.write(filtered)
        value = {'status': 'coarse_initialized_visual_support_ready_BA_REQUIRED', 'source_model': str(source),
            'source_identity': model_identity(source), 'model': str(filtered), 'output_identity': model_identity(filtered),
            'registered_frames': model.num_reg_images(), 'coarse_initialization_names': sorted(initialize - set(rejected_additions)),
            'initialization_protocol': SENSOR_INITIALIZATION_PROTOCOL,
            'initialization_roles': roles,
            'observed_visual_poses_retained_as_initializers': not (
                roles['degenerate_observed_original_names'] or roles['unverified_conflicted_original_names']),
            'supported_visual_poses_retained_as_initializers': not roles['unverified_conflicted_original_names'],
            'required_initial_names_all_registered': set(roles['required_names']) <= {
                image.name for image in model.images.values() if image.has_pose},
            'verification_route': verification, 'pnp_support_gate_passed': not failed,
            'pnp_support_gate_failed_original_names': failed,
            'rejected_support_additions': rejected_additions, 'database': str(database), 'images': str(bridge / 'images'),
            'coarse_only_pose_is_not_accepted': True, 'native_metric_BA_required': True,
            'coarse_consistency_after_m': coarse_consistency(model, initial), 'elapsed_seconds': time.monotonic() - began}
        atomic_json(output / 'RESULT.json', value)
        return value
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_sensor_initialized_camera_support',
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - began})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'bridge', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verification', choices=('pnp-support', 'joint-ba'), default='pnp-support')
    parser.add_argument('--pnp-recovery-input', type=Path)
    args = parser.parse_args()
    value = run(args.source_model, args.initial_frames, args.bridge, args.output, args.verification,
                args.pnp_recovery_input)
    print(value['status'], value['registered_frames'], flush=True)
