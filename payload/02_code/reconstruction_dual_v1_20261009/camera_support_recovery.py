"""Re-localize weak registered cameras using genuine cached visual correspondences."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import shutil
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from common import ROOT, COLMAP, atomic_json, read_json
from native_command import run_native
from camera_initialization_roles import VISUAL_RECOVERY_PROTOCOL, visual_recovery_targets


def model_identity(source):
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('cameras.bin', 'images.bin', 'points3D.bin')}


def coarse_consistency(model, initial):
    rows = [read_json_line(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
    lookup = {row['image_name']: row for row in rows if row['coarse_pose_valid'] and not row['excluded_role']
        and row['decision'] in ('eligible_not_selected', 'selected_initial_sfm_pending')}
    names = sorted(image.name for image in model.images.values() if image.has_pose and image.name in lookup)
    centers = np.asarray([lookup[name]['C_W_m'] for name in names])
    options = pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0)
    aligned = pycolmap.Reconstruction(model)
    transform = pycolmap.align_reconstruction_to_locations(aligned, names, centers, 3, options)
    if transform is None:
        raise ValueError('Fresh metric alignment is unavailable for camera support diagnostics')
    aligned.transform(transform)
    errors = {image.name: float(np.linalg.norm(image.projection_center() - lookup[image.name]['C_W_m']))
        for image in aligned.images.values() if image.has_pose and image.name in lookup}
    return errors


def read_json_line(line):
    import json
    return json.loads(line)


def run(source, initial, output, matching_only=False):
    source, initial, output = source.resolve(strict=True), initial.resolve(strict=True), output.resolve()
    identity = model_identity(source)
    manifest_identity = hashlib.sha256((initial / 'INITIAL_FRAME_MANIFEST.json').read_bytes()).hexdigest()
    if (output / 'RESULT.json').is_file():
        receipt = read_json(output / 'RESULT.json')
        expected = 'verified_missing_matches_ready' if matching_only else 'supported_camera_model_ready'
        if (receipt['status'] != expected or receipt['source_identity'] != identity
                or receipt.get('visual_recovery_protocol') != VISUAL_RECOVERY_PROTOCOL
                or receipt.get('initial_manifest_identity') != manifest_identity
                or (not matching_only and model_identity(Path(receipt['model'])) != receipt['output_identity'])):
            raise ValueError('A camera support recovery cache changed or did not complete')
        return receipt
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        original = pycolmap.Reconstruction(source)
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        errors = coarse_consistency(original, initial)
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        targets = visual_recovery_targets({image.name: image.num_points3D for image in original.images.values()
            if image.has_pose}, initial_names, errors, minimum, .6)
        suspect_names = set(targets['registered_suspect_names'])
        suspects = [image for image in original.images.values() if image.has_pose and image.name in suspect_names]
        names = {image.name for image in original.images.values() if image.has_pose}
        details = [{'image_name': image.name, 'observations': image.num_points3D,
            'coarse_center_delta_m': errors.get(image.name), 'image_id': image.image_id,
            'weak_support': image.num_points3D < minimum} for image in suspects]
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': identity,
            'initial_frames': str(initial), 'original_camera_count': len(names), 'suspects': details,
            'minimum_observations': minimum, 'minimum_source': 'installed IncrementalMapperOptions.abs_pose_min_num_inliers',
            'coarse_conflict_m': .6, 'coarse_conflict_source': 'unchanged metric alignment RANSAC maximum error',
            'all_visual_frames_retained': True, 'GT_used': False, 'coarse_pose_inserted_as_visual_pose': False,
            'visual_recovery_protocol': VISUAL_RECOVERY_PROTOCOL, 'initial_manifest_identity': manifest_identity,
            'required_missing_initial_names': targets['missing_initial_names'],
            'all_recovery_target_names': targets['target_names'],
            'recovery': 'genuine missing matches plus warm native mapper with trusted existing frames fixed'})
        target_model = output / 'models'
        target_model.mkdir()
        timings = {}
        if targets['target_names'] or matching_only:
            previous = pycolmap.Reconstruction(original)
            for image in suspects:
                previous.deregister_frame(image.frame_id)
            private = output / 'private_supported_source'
            private.mkdir()
            previous.write(private)
            database_source = None
            for parent in (source.parent, source.parent.parent, source.parent.parent.parent):
                if (parent / 'database.db').is_file():
                    database_source = parent / 'database.db'
                    break
            if database_source is None:
                raise FileNotFoundError('The same-run camera recovery requires its real matching database')
            database = output / 'database.db'
            with closing(sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True)) as incoming:
                with closing(sqlite3.connect(database)) as destination:
                    incoming.backup(destination)
                images = dict(incoming.execute('SELECT image_id,name FROM images'))
                attempted = {row[0] for row in incoming.execute('SELECT pair_id FROM matches')}
            target_names = set(targets['target_names'])
            if not set(targets['required_names']) <= set(images.values()):
                raise ValueError('A required initial or original camera is absent from the actual feature database')
            ids = sorted(images)
            pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                if (images[a] in target_names or images[b] in target_names) and 2147483647*a+b not in attempted]
            pair_file = output / 'support_pairs.txt'
            pair_file.write_text(''.join(images[a] + ' ' + images[b] + '\n' for a, b in pairs), encoding='utf-8')
            image_root = source.parent / 'images'
            if not image_root.is_dir():
                image_root = initial / 'images'
            if any(not (image_root / name).is_file() for name in targets['required_names']):
                raise FileNotFoundError('Camera recovery image denominator changed')

            def execute(label, command, gpu=False):
                budget = 1800 if label == 'matching' else 600
                timings[label] = run_native(command, output, label, budget, gpu)

            if pairs:
                execute('matching', [COLMAP, 'matches_importer', '--database_path', database,
                    '--match_list_path', pair_file, '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1',
                    '--FeatureMatching.gpu_index', '0', '--FeatureMatching.num_threads', '8',
                    '--TwoViewGeometry.random_seed', '0', '--default_random_seed', '0'], True)
            if matching_only:
                receipt = {'status': 'verified_missing_matches_ready', 'source_model': str(source),
                    'source_identity': identity, 'database': str(database), 'images': str(image_root),
                    'registered_frames': len(names), 'relocalized_camera_count': 0,
                    'requested_matching_pairs': len(pairs), 'timings': timings,
                    'visual_recovery_protocol': VISUAL_RECOVERY_PROTOCOL, 'initial_manifest_identity': manifest_identity,
                    'required_missing_initial_names': targets['missing_initial_names'],
                    'all_recovery_target_names': targets['target_names'],
                    'elapsed_seconds': time.monotonic() - started, 'camera_pose_verification_pending': True}
                atomic_json(output / 'RESULT.json', receipt)
                return receipt
            allowed = output / 'mapper_required_images.txt'
            allowed.write_text('\n'.join(targets['required_names']) + '\n', encoding='utf-8')
            execute('mapping', [COLMAP, 'mapper', '--database_path', database, '--image_path', image_root,
                '--input_path', private, '--output_path', target_model, '--Mapper.fix_existing_frames', '1',
                '--Mapper.image_list_path', allowed,
                '--Mapper.num_threads', '8', '--Mapper.random_seed', '0', '--default_random_seed', '0',
                '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_principal_point', '0',
                '--Mapper.ba_refine_extra_params', '0', '--Mapper.ba_refine_sensor_from_rig', '0',
                '--Mapper.max_runtime_seconds', '590'])
        else:
            original.write(target_model)
        repaired = pycolmap.Reconstruction(target_model)
        repaired_names = {image.name for image in repaired.images.values() if image.has_pose}
        if repaired_names != set(targets['required_names']):
            raise ValueError('Camera support recovery did not retain every original and required initial camera')
        remaining = [{'name': image.name, 'observations': image.num_points3D}
            for image in repaired.images.values() if image.has_pose and image.num_points3D < minimum]
        if remaining:
            atomic_json(output / 'UNRESOLVED_CAMERAS.json', {'status': 'weak_visual_support_remains', 'rows': remaining})
            raise ValueError('Registered cameras still lack the native minimum visual support')
        suspect_names = {image.name for image in suspects}
        maximum_pose_change = max((float(np.max(np.abs(image.cam_from_world().matrix()
            - repaired.images[image.image_id].cam_from_world().matrix())))
            for image in original.images.values() if image.has_pose and image.name not in suspect_names), default=0)
        if maximum_pose_change > 1e-9 or any(not np.array_equal(camera.params, repaired.cameras[key].params)
                for key, camera in original.cameras.items()):
            raise ValueError('Camera recovery moved a trusted pose or changed calibrated intrinsics')
        receipt = {'status': 'supported_camera_model_ready', 'source_model': str(source), 'source_identity': identity,
            'model': str(target_model), 'output_identity': model_identity(target_model),
            'registered_frames': len(repaired_names), 'relocalized_camera_count': len(targets['target_names']),
            'visual_recovery_protocol': VISUAL_RECOVERY_PROTOCOL, 'initial_manifest_identity': manifest_identity,
            'required_missing_initial_names': targets['missing_initial_names'],
            'minimum_visual_observations': minimum, 'trusted_pose_maximum_component_delta': maximum_pose_change,
            'coarse_residuals_after_m': coarse_consistency(repaired, initial),
            'timings': timings, 'elapsed_seconds': time.monotonic() - started,
            'camera_drop_or_coarse_pose_fabrication': False, 'independent_geometry_accuracy_verified': False}
        atomic_json(output / 'RESULT.json', receipt)
        return receipt
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_camera_support_recovery',
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--initial-frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matching-only', action='store_true')
    args = parser.parse_args()
    result = run(args.source_model, args.initial_frames, args.output, args.matching_only)
    print(result['status'], result['relocalized_camera_count'], flush=True)
