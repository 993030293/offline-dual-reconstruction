"""Recover real weak cameras with bounded feature refinement and native tracks."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
import pycolmap

from camera_joint_conditioning import load_helpers
from camera_support_recovery import model_identity
from camera_visual_recovery import fixed_camera_invariants, required_native_support, required_registered_support
from common import COLMAP, atomic_json
from native_command import run_native


def database_identity(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_feature_binding(model, database):
    """Keep native feature indices bound to the actual database, including replaced arrays."""
    count = 0
    with closing(sqlite3.connect('file:' + Path(database).as_posix() + '?mode=ro', uri=True)) as db:
        images = {key: (name, camera_id) for key, name, camera_id in
            db.execute('SELECT image_id,name,camera_id FROM images')}
        for image in model.images.values():
            if not image.has_pose:
                continue
            if images.get(image.image_id) != (image.name, image.camera_id):
                raise ValueError('Recovered native image identity differs from the actual feature database')
            row = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image.image_id,)).fetchone()
            if row is None:
                raise ValueError('A recovered native image lacks actual feature coordinates')
            actual = np.frombuffer(row[2], dtype=np.float32).reshape(row[0], row[1])[:, :2]
            pixels = np.asarray([point.xy for point in image.points2D]).reshape(-1, 2)
            if pixels.shape != actual.shape or not np.array_equal(pixels, actual.astype(np.float64)):
                raise ValueError('Recovered native feature coordinates or indices differ from the actual database')
            count += len(pixels)
    return count


def register_with_existing_geometry(mapper, options, image_id):
    """Use available 2D-to-3D support before structure-less pose estimation."""
    attempts = []
    for method, register in (
            ('native_PnP', mapper.register_next_image),
            ('native_structureless', mapper.register_next_structure_less_image)):
        ok = bool(register(options, image_id))
        attempts.append({'method': method, 'native_registered': ok})
        if ok:
            return True, method, attempts
    return False, None, attempts


def run(source, required, database_source, images, output, maximum_seconds, targets,
        target_scope='below_native_minimum'):
    source, database_source, images = (Path(value).resolve(strict=True)
        for value in (source, database_source, images))
    output = Path(output).resolve()
    if maximum_seconds <= 60:
        raise TimeoutError('Required feature recovery lacks its original final triangulation reserve')
    output.mkdir(parents=True, exist_ok=False)
    started, source_identity, input_database_identity = time.monotonic(), model_identity(source), database_identity(database_source)
    deadline = started + maximum_seconds
    timings, rows = {}, []

    def remaining(reserve=0):
        value = deadline-time.monotonic()-reserve
        if value <= 1:
            raise TimeoutError('Bounded required feature recovery exhausted its shared parent budget')
        return value

    def native(label, command):
        timings[label] = run_native(command, output, label, remaining(60), True,
            lease_filename=label+'_GPU_LEASE.json', lease_label='required_native_feature_'+label)

    try:
        original = pycolmap.Reconstruction(source)
        original_names = {image.name for image in original.images.values() if image.has_pose}
        required, targets = set(required), dict(targets)
        minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
        actual_support = {image.name: image.num_points3D for image in original.images.values() if image.has_pose}
        if target_scope == 'below_native_minimum':
            names = original_names
            expected = {name: actual_support[name] for name in sorted(required & names)
                if actual_support[name] < minimum}
            valid_scope = required <= names
        elif target_scope == 'missing_required':
            names = original_names | required
            expected = {name: 0 for name in sorted(required-original_names)}
            valid_scope = True
        else:
            raise ValueError('Required feature recovery received an unknown target scope')
        if not required or not valid_scope or not targets or targets != expected:
            raise ValueError('Required feature targets must equal actual same-run weak required camera support')
        anchors = pycolmap.Reconstruction(original)
        for image in list(anchors.images.values()):
            if image.has_pose and image.name in targets:
                anchors.deregister_frame(image.frame_id)
        if anchors.num_reg_images() < 2:
            raise ValueError('Required feature recovery lacks genuine fixed native anchors')
        if any(not (images / name).is_file() for name in names):
            raise FileNotFoundError('A required feature recovery image is absent')
        seed = output / 'trusted_anchors'
        seed.mkdir()
        anchors.write(seed)
        database = output / 'database.db'
        with closing(sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True)) as incoming:
            with closing(sqlite3.connect(database)) as destination:
                incoming.backup(destination)
        with closing(sqlite3.connect(database)) as db:
            ids = {name: key for key, name in db.execute('SELECT image_id,name FROM images')}
            if not names <= ids.keys():
                raise ValueError('Required refinement lacks actual database image identities')
            cameras = [row[0] for row in db.execute('SELECT camera_id FROM cameras')]
            if len(cameras) != 1:
                raise ValueError('Required refinement needs the supported single calibrated camera')
            pairs = sorted({min(ids[name], other)*2147483647+max(ids[name], other)
                for name in targets for other in ids.values() if ids[name] != other})
            counts = {name: db.execute('SELECT rows FROM keypoints WHERE image_id=?', (ids[name],)).fetchone()[0]
                for name in targets}
            limit = db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            with db:
                for table in ('matches', 'two_view_geometries'):
                    for offset in range(0, len(pairs), limit):
                        batch = pairs[offset:offset+limit]
                        db.execute('DELETE FROM '+table+' WHERE pair_id IN ('+','.join('?' for _ in batch)+')', batch)
                for name in targets:
                    db.execute('DELETE FROM keypoints WHERE image_id=?', (ids[name],))
                    db.execute('DELETE FROM descriptors WHERE image_id=?', (ids[name],))
        feature_list, pair_list = output / 'new_features.txt', output / 'new_pairs.txt'
        feature_list.write_text('\n'.join(sorted(targets))+'\n', encoding='utf-8')
        ordered = sorted(names)
        pair_list.write_text(''.join(a+' '+b+'\n' for index, a in enumerate(ordered)
            for b in ordered[index+1:] if a in targets or b in targets), encoding='utf-8')
        peak = pycolmap.SiftExtractionOptions().peak_threshold
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': source_identity,
            'source_database_identity': input_database_identity, 'actual_targets': targets,
            'target_scope': target_scope, 'original_registered_names': sorted(original_names),
            'initially_missing_required_names': sorted(required-original_names),
            'original_feature_counts': counts, 'original_detector_peak_threshold': peak,
            'one_detector_refinement_peak_threshold': peak/2, 'maximum_features_unchanged': 8192,
            'invalidated_pairs_including_unregistered_database_images': len(pairs),
            'native_geometry_and_final_support_gates_unchanged': True,
            'native_registration_policy': 'existing_geometry_PnP_first_structureless_if_PnP_rejected_v1',
            'maximum_seconds_within_original_parent_budget': maximum_seconds,
            'clear_points_after_native_recovery': False,
            'training_admission_granted': False, 'runtime_language_model_used': False})
        native('features', [COLMAP, 'feature_extractor', '--database_path', database, '--image_path', images,
            '--image_list_path', feature_list, '--ImageReader.existing_camera_id', cameras[0],
            '--FeatureExtraction.use_gpu', '1', '--FeatureExtraction.gpu_index', '0',
            '--FeatureExtraction.num_threads', '8', '--FeatureExtraction.max_image_size', '1600',
            '--SiftExtraction.max_num_features', '8192', '--SiftExtraction.peak_threshold', str(peak/2),
            '--default_random_seed', '0'])
        native('matching', [COLMAP, 'matches_importer', '--database_path', database,
            '--match_list_path', pair_list, '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1',
            '--FeatureMatching.gpu_index', '0', '--FeatureMatching.num_threads', '8',
            '--TwoViewGeometry.random_seed', '0', '--default_random_seed', '0'])
        with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as db:
            after_counts = {name: db.execute('SELECT rows FROM keypoints WHERE image_id=?', (ids[name],)).fetchone()[0]
                for name in targets}
        atomic_json(output / 'FEATURE_COUNTS.json', {'before': counts, 'after': after_counts})
        # Serialization omits deregistered images; the SDK must reload their new database arrays.
        model = pycolmap.Reconstruction(seed)
        pipeline, cache_options = pycolmap.IncrementalPipelineOptions(), pycolmap.DatabaseCacheOptions()
        cache_options.min_num_matches, cache_options.image_names = pipeline.min_num_matches, names
        cache_options.load_all_images = True
        with pycolmap.Database.open(database) as db:
            cache = pycolmap.DatabaseCache.create(db, cache_options)
        mapper = pycolmap.IncrementalMapper(cache)
        mapper.begin_reconstruction(model)
        options, tri = pycolmap.IncrementalMapperOptions(), pycolmap.IncrementalTriangulatorOptions()
        options.fix_existing_frames = True
        options.abs_pose_refine_focal_length = options.abs_pose_refine_extra_params = False
        options.num_threads, options.random_seed, tri.random_seed = 8, 0, 0
        ba = load_helpers().ba_options(pipeline.ba_local_max_num_iterations)
        actual = {image.name: image.image_id for image in model.images.values()}
        pending = sorted(targets, key=lambda name: (-targets[name], name))
        try:
            for pass_index in range(options.max_reg_trials):
                accepted = []
                for name in pending:
                    ba.ceres.solver_options.max_solver_time_in_seconds = min(30., remaining(60))
                    image_id = actual[name]
                    ok, method, attempts = register_with_existing_geometry(mapper, options, image_id)
                    row = {'name': name, 'pass_index': pass_index, 'native_registered': bool(ok),
                           'native_registration_method': method, 'native_registration_attempts': attempts}
                    if ok:
                        mapper.triangulate_image(tri, image_id)
                        mapper.iterative_local_refinement(pipeline.ba_local_max_refinements,
                            pipeline.ba_local_max_refinement_change, options, ba, tri, image_id)
                        accepted.append(name)
                        row['native_point_support_after_local_refinement'] = model.images[image_id].num_points3D
                    rows.append(row)
                    atomic_json(output / 'DECISIONS.json', {'rows': rows})
                pending = [name for name in pending if name not in accepted]
                if not pending or not accepted:
                    break
        finally:
            mapper.end_reconstruction(False)
        actual_names = {image.name for image in model.images.values() if image.has_pose}
        if actual_names != names:
            raise ValueError('Required native feature recovery lacks cameras: '+','.join(sorted(names-actual_names)))
        removed = pycolmap.ObservationManager(model).filter_all_points3D(
            options.filter_max_reproj_error, options.filter_min_tri_angle)
        if target_scope == 'missing_required':
            required_native_support(model, targets)
            observations = required_registered_support(model, required)
        else:
            observations = required_native_support(model, required)
        fixed_delta = fixed_camera_invariants(anchors, model)
        drift = fixed_camera_invariants(original, model, preserve_poses=False)
        checked = verify_feature_binding(model, database)
        models = output / 'models'
        models.mkdir()
        model.write(models)
        if model_identity(source) != source_identity or database_identity(database_source) != input_database_identity:
            raise ValueError('Required native feature recovery changed its immutable source')
        result = {'status': 'native_required_feature_relocalized_incremental_triangulation_REQUIRED',
            'source_model': str(source), 'source_identity': source_identity, 'model': str(models),
            'output_identity': model_identity(models), 'database': str(database), 'images': str(images),
            'targets_before': targets, 'registered_frames': model.num_reg_images(),
            'target_scope': target_scope, 'initially_missing_required_names': sorted(required-original_names),
            'target_support_before_mandatory_rebuild': {name: observations[name] for name in targets},
            'remaining_anchor_maximum_component_delta': fixed_delta, 'actual_pose_maximum_component_delta': drift,
            'native_filtered_observations': int(removed), 'actual_database_keypoint_bindings_checked': checked,
            'point_rebuilding_mode': 'native_incremental_preserve_geometry_filtered_tracks',
            'full_required_point_support_pending': True, 'training_admission_granted': False,
            'timings': timings, 'elapsed_seconds': time.monotonic()-started, 'runtime_language_model_used': False}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_required_native_feature_recovery',
            'traceback': traceback.format_exc(), 'decisions': rows,
            'source_model_unchanged': model_identity(source) == source_identity,
            'source_database_unchanged': database_identity(database_source) == input_database_identity})
        raise
