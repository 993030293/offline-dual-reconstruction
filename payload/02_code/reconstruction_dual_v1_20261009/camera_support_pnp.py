"""Repair camera support with genuine PnP while keeping trusted geometry fixed."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from camera_support_recovery import model_identity, coarse_consistency
from common import atomic_json, read_json
from camera_initialization_roles import VISUAL_RECOVERY_PROTOCOL, visual_recovery_targets
from camera_pnp_recovery_input import preserve_verified_updates

PAIR_BASE = 2147483647


def correspondences(connection, image_id, trusted):
    votes = defaultdict(Counter)
    pairs = sorted({min(image_id, key)*PAIR_BASE + max(image_id, key)
                    for key in trusted if key != image_id})
    limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    if limit <= 0:
        raise ValueError('Native pair queries require a positive SQLite parameter limit')
    def matched_rows():
        for offset in range(0, len(pairs), limit):
            values = pairs[offset:offset+limit]
            sql = ('SELECT pair_id,rows,data FROM two_view_geometries WHERE pair_id IN ('
                + ','.join('?' for _ in values) + ') AND rows>0 ORDER BY pair_id')
            yield from connection.execute(sql, values)
    for pair_id, count, payload in matched_rows():
        first, second = divmod(pair_id, PAIR_BASE)
        neighbor, column = (second, 0) if first == image_id else (first, 1)
        if neighbor not in trusted or not count:
            continue
        matches = np.frombuffer(payload, np.uint32).reshape(count, 2)
        reference = trusted[neighbor]
        for feature, reference_feature in matches[:, [column, 1-column]]:
            point_id = reference[int(reference_feature)]
            if point_id >= 0:
                votes[int(feature)][int(point_id)] += 1
    choices = []
    for feature, counts in votes.items():
        ranked = counts.most_common(2)
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            choices.append((ranked[0][1], feature, ranked[0][0]))
    unique, seen = [], set()
    for weight, feature, point_id in sorted(choices, reverse=True):
        if point_id not in seen:
            unique.append((feature, point_id, weight))
            seen.add(point_id)
    return unique


def estimate_camera(connection, image, model, trusted, mapper, lookup, alignment, *, rejection_cache=None):
    if image.name not in lookup:
        decision = {'image_name': image.name, 'image_id': image.image_id,
            'before_observations': image.num_points3D, 'correspondences': 0,
            'status': 'abstain_missing_qualified_coarse_pose', 'scope': 'coarse_source_qualification'}
        return decision, [decision], None, []
    timestamp = int(Path(image.name).stem.rsplit('_', 1)[1])
    local = {key: value for key, value in trusted.items() if abs(
        int(Path(model.images[key].name).stem.rsplit('_', 1)[1]) - timestamp) <= 800_000_000}
    attempts = []
    for scope, anchors in (('existing_augmentation_temporal_window', local), ('full_trusted_map', trusted)):
        choices = correspondences(connection, image.image_id, anchors)
        decision = {'image_name': image.name, 'image_id': image.image_id,
            'before_observations': image.num_points3D, 'correspondences': len(choices), 'scope': scope}
        xy = np.asarray([image.points2D[feature].xy for feature, _, _ in choices], np.float64).reshape(-1, 2)
        xyz = np.asarray([model.points3D[point_id].xyz for _, point_id, _ in choices], np.float64).reshape(-1, 3)
        camera = model.cameras[image.camera_id]
        # Cache only identical numerical inputs and rejected estimates in this run.
        signature = (tuple((feature, point_id) for feature, point_id, _ in choices), xy.tobytes(), xyz.tobytes(),
            str(camera.model), camera.width, camera.height, camera.params.tobytes(),
            np.asarray(lookup[image.name]['C_W_m'], np.float64).tobytes(), alignment.params.tobytes(),
            mapper.abs_pose_min_num_inliers, mapper.abs_pose_max_error, mapper.abs_pose_min_inlier_ratio)
        cache_key = (int(image.image_id), scope)
        previous = rejection_cache.get(cache_key) if rejection_cache is not None else None
        if previous is not None and previous['signature'] == signature:
            cached = dict(previous['decision'])
            cached.update(before_observations=image.num_points3D,
                unchanged_effective_PnP_input_cached=True, native_pose_estimation_executed=False)
            attempts.append(cached)
            continue
        decision['native_pose_estimation_executed'] = len(choices) >= mapper.abs_pose_min_num_inliers
        estimate = None
        if len(choices) >= mapper.abs_pose_min_num_inliers:
            estimation = pycolmap.AbsolutePoseEstimationOptions()
            estimation.ransac.max_error = mapper.abs_pose_max_error
            estimation.ransac.min_inlier_ratio = mapper.abs_pose_min_inlier_ratio
            estimation.ransac.random_seed = 0
            estimate = pycolmap.estimate_and_refine_absolute_pose(xy, xyz,
                model.cameras[image.camera_id], estimation, pycolmap.AbsolutePoseRefinementOptions())
        if estimate is None:
            decision['status'] = 'abstain_no_genuine_PnP'
        else:
            pose = estimate['cam_from_world']
            mask = np.asarray(estimate['inlier_mask'], bool)
            camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
            valid = mask & np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0)
            projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
            error_px = np.linalg.norm(projected - xy, axis=1)
            valid &= np.isfinite(projected).all(axis=1) & (error_px <= mapper.abs_pose_max_error)
            center = pose.inverse().translation
            metric_center = alignment.scale * (alignment.rotation.matrix() @ center) + alignment.translation
            delta = float(np.linalg.norm(metric_center - lookup[image.name]['C_W_m']))
            decision.update(pnp_inliers=int(np.sum(mask)), positive_valid_observations=int(np.sum(valid)),
                verified_inlier_ratio=float(np.mean(valid)), coarse_center_delta_m=delta,
                reprojection_p95_px=float(np.quantile(error_px[valid], .95)) if np.any(valid) else None)
            if (np.sum(valid) < mapper.abs_pose_min_num_inliers or np.mean(valid) < mapper.abs_pose_min_inlier_ratio
                    or delta > .6 or not np.isfinite(pose.matrix()).all()):
                decision['status'] = 'abstain_PnP_geometry_or_coarse_conflict'
            else:
                decision['status'] = 'genuine_PnP_accepted'
                attempts.append(decision)
                return decision, attempts, pose, [choices[index] for index in np.flatnonzero(valid)]
        attempts.append(decision)
        if rejection_cache is not None:
            rejection_cache[cache_key] = {'signature': signature, 'decision': dict(decision)}
    return attempts[-1], attempts, None, []


def run(source, initial, database, output, trusted_model=None):
    source, initial, database = (Path(value).resolve(strict=True) for value in (source, initial, database))
    output = Path(output).resolve()
    identity = model_identity(source)
    manifest_identity = hashlib.sha256((initial / 'INITIAL_FRAME_MANIFEST.json').read_bytes()).hexdigest()
    trusted_model = Path(trusted_model).resolve(strict=True) if trusted_model else None
    trusted_identity = model_identity(trusted_model) if trusted_model else None
    if (output / 'RESULT.json').is_file():
        value = read_json(output / 'RESULT.json')
        if (value['status'] != 'supported_camera_model_ready' or value['source_identity'] != identity
                or value.get('visual_recovery_protocol') != VISUAL_RECOVERY_PROTOCOL
                or value.get('initial_manifest_identity') != manifest_identity
                or value.get('trusted_model_identity') != trusted_identity
                or model_identity(Path(value['model'])) != value['output_identity']):
            raise ValueError('Fixed-map PnP cache identity or completion changed')
        return value
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        original = pycolmap.Reconstruction(source)
        base = pycolmap.Reconstruction(trusted_model) if trusted_model else pycolmap.Reconstruction(original)
        model = pycolmap.Reconstruction(base)
        mapper = pycolmap.IncrementalMapperOptions()
        errors = coarse_consistency(original, initial)
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        targets = visual_recovery_targets({image.name: image.num_points3D for image in original.images.values()
            if image.has_pose}, initial_names, errors, mapper.abs_pose_min_num_inliers, .6)
        registered_suspects = set(targets['registered_suspect_names'])
        suspects = [image for image in original.images.values() if image.has_pose and image.name in registered_suspects]
        suspect_ids = {image.image_id for image in suspects}
        for image in suspects:
            if image.image_id in model.images:
                if model.images[image.image_id].has_pose:
                    model.deregister_frame(image.frame_id)
            else:
                old_frame = original.frames[image.frame_id]
                if image.frame_id not in model.frames:
                    frame = pycolmap.Frame()
                    frame.frame_id, frame.rig_id = old_frame.frame_id, old_frame.rig_id
                    for data_id in old_frame.data_ids:
                        frame.add_data_id(data_id)
                    model.add_frame(frame)
                local = pycolmap.Image(name=image.name, keypoints=np.asarray([point.xy for point in image.points2D]),
                    camera_id=image.camera_id, image_id=image.image_id)
                local.frame_id = image.frame_id
                model.add_image(local)
        with pycolmap.Database.open(database) as db:
            database_images = {image.name: image for image in db.read_all_images()}
            database_frames = {int(data.id): frame for frame in db.read_all_frames() for data in frame.data_ids}
            for name in targets['missing_initial_names']:
                if name not in database_images:
                    raise ValueError('A missing initial camera has no actual feature database image')
                stored = database_images[name]
                frame = database_frames[stored.image_id]
                if stored.camera_id not in model.cameras or frame.rig_id not in model.rigs:
                    raise ValueError('A missing initial camera does not use the actual fixed camera rig')
                if frame.frame_id not in model.frames:
                    model.add_frame(frame)
                keypoints = np.asarray(db.read_keypoints(stored.image_id)[:, :2], np.float64)
                if stored.image_id in model.images:
                    local = model.images[stored.image_id]
                    if local.has_pose or local.name != name or local.frame_id != frame.frame_id or local.num_points3D:
                        raise ValueError('An allegedly missing initial camera has inconsistent model membership')
                    local.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in keypoints])
                else:
                    local = pycolmap.Image(name=name, keypoints=keypoints,
                        camera_id=stored.camera_id, image_id=stored.image_id)
                    local.frame_id = frame.frame_id
                    model.add_image(local)
                suspects.append(model.images[stored.image_id])
                suspect_ids.add(stored.image_id)
        trusted = {image.image_id: np.asarray([int(point.point3D_id) if point.has_point3D()
            and model.points3D[point.point3D_id].track.length() >= 3 else -1 for point in image.points2D], np.int64)
            for image in model.images.values() if image.has_pose}
        rows = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
        lookup = {row['image_name']: row for row in rows if row['coarse_pose_valid'] and not row['excluded_role']
            and row['decision'] in ('eligible_not_selected', 'selected_initial_sfm_pending')}
        names = sorted(image.name for image in original.images.values() if image.has_pose and image.name in lookup)
        locations = np.asarray([lookup[name]['C_W_m'] for name in names])
        alignment = pycolmap.align_reconstruction_to_locations(original, names, locations, 3,
            pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0))
        if alignment is None:
            raise ValueError('No raw coarse center alignment for PnP consistency checks')
        contract = {'source_model': str(source), 'source_identity': identity, 'database': str(database),
            'trusted_model': str(trusted_model) if trusted_model else None, 'trusted_model_identity': trusted_identity,
            'same_run_verified_visual_matches_only': True, 'coarse_pose_inserted_as_visual_pose': False,
            'trusted_poses_intrinsics_and_points_fixed': True, 'all_visual_frames_retained': True,
            'minimum_pnp_inliers': mapper.abs_pose_min_num_inliers,
            'minimum_pnp_inlier_ratio': mapper.abs_pose_min_inlier_ratio,
            'maximum_pnp_error_px': mapper.abs_pose_max_error,
            'threshold_source': 'installed native IncrementalMapperOptions PnP defaults',
            'minimum_trusted_point_track': 3, 'maximum_coarse_center_conflict_m': .6,
            'coarse_gate_source': 'unchanged metric alignment RANSAC',
            'suspect_names': sorted(image.name for image in suspects), 'GT_used': False,
            'visual_recovery_protocol': VISUAL_RECOVERY_PROTOCOL, 'initial_manifest_identity': manifest_identity,
            'registered_suspect_names': targets['registered_suspect_names'],
            'required_missing_initial_names': targets['missing_initial_names'],
            'local_correspondence_window_seconds': .8, 'window_source': 'existing six-bin temporal augmentation',
            'verified_PnP_frames_can_propagate_existing_point_correspondences': True,
            'indexed_native_pair_ids_only_for_actual_trusted_neighbors': True,
            'unchanged_exact_numeric_PnP_rejections_cached_only_in_this_run': True,
            'cached_rejections_never_grant_registration_or_training': True,
            'all_per_frame_attempts_retained': True}
        atomic_json(output / 'CONTRACT.json', contract)
        decisions, attempts = {}, []
        with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as connection:
            pending = {image.image_id: image for image in suspects}
            rejection_cache = {}
            for pass_index in range(len(suspects)):
                accepted = []
                for key, image in sorted(pending.items()):
                    decision, history, pose, observations = estimate_camera(connection, image, model,
                        trusted, mapper, lookup, alignment, rejection_cache=rejection_cache)
                    decision['pass_index'] = pass_index
                    attempts.extend({**row, 'pass_index': pass_index} for row in history)
                    decisions[image.name] = decision
                    if pose is not None:
                        model.frames[image.frame_id].set_cam_from_world(image.camera_id, pose)
                        model.register_frame(image.frame_id)
                        for feature, point_id, _ in observations:
                            model.add_observation(point_id, pycolmap.TrackElement(key, feature))
                        trusted[key] = np.asarray([int(point.point3D_id) if point.has_point3D()
                            else -1 for point in model.images[key].points2D], np.int64)
                        accepted.append(key)
                for key in accepted:
                    del pending[key]
                atomic_json(output / 'FRAME_DECISIONS.json', {'rows': list(decisions.values()),
                    'attempts': attempts, 'all_frames_required': True})
                if not pending or not accepted:
                    break
            rejected = [image.name for image in pending.values()]
        if rejected:
            partial_path = None
            if trusted_model is None and any(row['status'] == 'genuine_PnP_accepted' for row in decisions.values()):
                partial = preserve_verified_updates(original, model, list(decisions.values()))
                destination = output / 'recovery_input_model'
                destination.mkdir()
                model.write(destination)
                images = output / 'images'
                images.mkdir()
                source_images = source.parent / 'images'
                if not source_images.is_dir():
                    source_images = initial / 'images'
                for image in model.images.values():
                    if image.has_pose:
                        incoming = source_images / image.name
                        if not incoming.is_file():
                            incoming = initial / 'images' / image.name
                        os.link(incoming, images / image.name)
                partial.update(status='same_run_partial_PnP_recovery_input_BA_required',
                    source_model=str(source), source_identity=identity, model=str(destination),
                    output_identity=model_identity(destination), images=str(images),
                    initial_manifest_identity=manifest_identity,
                    registered_frames=model.num_reg_images())
                partial_path = output / 'RECOVERY_INPUT.json'
                atomic_json(partial_path, partial)
            atomic_json(output / 'UNVERIFIED_FRAMES.json', {'status': 'genuine_PnP_support_incomplete',
                'source_identity': identity, 'names': sorted(rejected),
                'frame_decisions': str(output / 'FRAME_DECISIONS.json'),
                'partial_recovery_input': str(partial_path) if partial_path else None})
            raise ValueError('Genuine fixed-map PnP could not verify every suspect: ' + ', '.join(rejected))
        expected_names = set(targets['required_names'])
        expected_names |= {image.name for image in base.images.values() if image.has_pose}
        if {image.name for image in model.images.values() if image.has_pose} != expected_names:
            raise ValueError('Fixed-map PnP changed the registration denominator')
        maximum = max((float(np.max(np.abs(image.cam_from_world().matrix()
            - model.images[image.image_id].cam_from_world().matrix()))) for image in original.images.values()
            if image.has_pose and image.image_id not in suspect_ids), default=0)
        if maximum > 1e-9 or any(not np.array_equal(camera.params, model.cameras[key].params)
                for key, camera in original.cameras.items()):
            raise ValueError('Fixed-map PnP changed a trusted camera')
        if any(not np.array_equal(point.xyz, model.points3D[key].xyz) for key, point in base.points3D.items()
                if key in model.points3D):
            raise ValueError('Fixed-map PnP moved a trusted 3D point')
        destination = output / 'models'
        destination.mkdir()
        model.update_point_3d_errors()
        model.write(destination)
        receipt = {'status': 'supported_camera_model_ready', 'method': 'fixed_trusted_map_genuine_PnP',
            'source_model': str(source), 'source_identity': identity, 'model': str(destination),
            'trusted_model_identity': trusted_identity,
            'output_identity': model_identity(destination), 'registered_frames': model.num_reg_images(),
            'relocalized_camera_count': len(suspects), 'minimum_camera_observations': min(image.num_points3D
                for image in model.images.values() if image.has_pose),
            'trusted_pose_maximum_component_delta': maximum, 'trusted_3D_points_unchanged': True,
            'frame_decisions': list(decisions.values()), 'coarse_residuals_after_m': coarse_consistency(model, initial),
            'visual_recovery_protocol': VISUAL_RECOVERY_PROTOCOL, 'initial_manifest_identity': manifest_identity,
            'required_initial_names_all_registered': initial_names <= expected_names,
            'required_missing_initial_names_recovered': targets['missing_initial_names'],
            'elapsed_seconds': time.monotonic() - started, 'camera_drop_or_coarse_pose_fabrication': False,
            'independent_geometry_accuracy_verified': False}
        atomic_json(output / 'RESULT.json', receipt)
        return receipt
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_fixed_map_PnP_recovery',
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--initial-frames', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-model', type=Path)
    args = parser.parse_args()
    value = run(args.source_model, args.initial_frames, args.database, args.output, args.trusted_model)
    print(value['status'], value['registered_frames'], value['relocalized_camera_count'], flush=True)
