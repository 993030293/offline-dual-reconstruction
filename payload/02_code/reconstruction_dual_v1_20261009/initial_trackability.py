"""Check visual trackability before freezing the initial raw-frame selection."""

from collections import defaultdict
from contextlib import closing, contextmanager
import importlib
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np
import pycolmap

from common import COLMAP, atomic_json, read_json
from native_command import run_native
from initial_connectivity import matching_components, component_recovery_pairs, main_component
from unique_feature_tracks import unambiguous_support
from initial_three_view_geometry import evaluate_geometry, PROTOCOL as GEOMETRY_PROTOCOL
from initial_quality_pairs import quality_pairs
from initial_connection_search import recover_connections, spanning_parents, PROTOCOL as CONNECTION_PROTOCOL

PAIR_BASE = 2147483647
WINDOW_NS = 800_000_000


def temporal_pairs(rows, image_ids, attempted):
    ordered = sorted(rows, key=lambda row: (row['camera_header_ns'], row['camera_frame'], row['image_name']))
    if len({row['image_name'] for row in ordered}) != len(ordered) or len({
            row['camera_header_ns'] for row in ordered}) != len(ordered):
        raise ValueError('Trackability support must use distinct original images and timestamps')
    if any(row['image_name'] not in image_ids for row in ordered):
        raise ValueError('A nominated trackability image has no actual native features')
    pairs = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if second['camera_header_ns'] - first['camera_header_ns'] > WINDOW_NS:
                break
            a, b = sorted((image_ids[first['image_name']], image_ids[second['image_name']]))
            if PAIR_BASE * a + b not in attempted:
                pairs.append((first['image_name'], second['image_name']))
    return pairs


def evaluate_trackability(connection, names, minimum_features, minimum_other_views, *, include_feature_ids=False):
    names = set(names)
    images = {name: int(key) for key, name in connection.execute('SELECT image_id,name FROM images') if name in names}
    if set(images) != names or minimum_features < 1 or minimum_other_views < 1:
        raise ValueError('Trackability subjects or installed support thresholds are invalid')
    counts = dict(connection.execute('SELECT image_id,rows FROM keypoints'))
    if not images.keys() or not set(images.values()) <= counts.keys():
        raise ValueError('Trackability requires real feature extraction for every subject')
    indices = {key: index for index, key in enumerate(sorted(images.values()))}
    support, consistency = unambiguous_support(connection, indices, counts, minimum_other_views + 1)
    retained = set(indices)
    while retained:
        mask = sum(1 << indices[key] for key in retained)
        failed = {key for key in retained if sum((bits & mask).bit_count() >= minimum_other_views
            for bits in support[key].values()) < minimum_features}
        if not failed:
            break
        retained -= failed
    mask = sum(1 << indices[key] for key in retained)
    decisions = {name: {'status': 'qualified_visual_trackability' if key in retained
            else 'abstain_no_three_view_feature_support', 'native_keypoint_count': counts[key],
        'supported_distinct_feature_count': sum((bits & mask).bit_count() >= minimum_other_views
            for bits in support[key].values()), 'minimum_distinct_features': minimum_features,
        'minimum_other_qualified_views_per_feature': minimum_other_views,
        'feature_track_consistency': consistency}
        for name, key in sorted(images.items())}
    if include_feature_ids:
        for name, key in images.items():
            decisions[name]['supported_feature_indices'] = sorted(feature for feature, bits in support[key].items()
                if (bits & mask).bit_count() >= minimum_other_views)
    return decisions


def select(native, args, choose, rows, policy, sharpness_min, baseline):
    root = Path(args.output) / 'trackability'
    root.mkdir(parents=True, exist_ok=False)
    images = root / 'images'
    images.mkdir()
    database = root / 'database.db'
    began = time.monotonic()
    minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
    connection_minimum = pycolmap.GlobalPipelineOptions().min_num_matches
    policy_path = native.ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json'
    other_views = read_json(policy_path)['point_filter']['minimum_point_track_length'] - 1
    if other_views != 2:
        raise ValueError('The installed three-view point-support policy changed')
    info = read_json(Path(args.output) / 'RAW_SENSOR_CONTRACT.json')['camera_info']
    k = np.asarray(info['camera_matrix']).reshape(3, 3)
    distortion = info['distortion']
    if info['distortion_model'] != 'plumb_bob' or len(distortion) != 5 or distortion[4] != 0:
        raise ValueError('Trackability requires the existing fixed OPENCV camera contract')
    params = [k[0, 0], k[1, 1], k[0, 2] + .5, k[1, 2] + .5, *distortion[:4]]
    contract = {'protocol': 'pre_manifest_bounded_native_connections_geometric_feature_tracks_v8', 'raw': str(args.raw.resolve(strict=True)),
        'minimum_distinct_features': minimum, 'minimum_other_qualified_views_per_feature': other_views,
        'threshold_source': 'installed PnP defaults and existing three-view point policy',
        'temporal_window_ns': WINDOW_NS, 'camera_params': list(map(float, params)),
        'maximum_candidates': 2 * policy['maximum_exported_frames'],
        'maximum_seconds': policy['maximum_wall_seconds'], 'SfM_model_or_final_poses_used': False,
        'LLM_or_manual_selected_frames_used': False, 'GT_used': False,
        'minimum_connectivity_pair_inliers': connection_minimum,
        'connectivity_threshold_source': 'installed GlobalPipelineOptions.min_num_matches, not absolute camera PnP',
        'connectivity_search_protocol': CONNECTION_PROTOCOL,
        'feature_track_consistency': 'one_keypoint_per_image_in_connected_component',
        'ambiguous_feature_components_excluded_before_frame_selection_freeze': True,
        'local_three_view_geometry_protocol': GEOMETRY_PROTOCOL,
        'local_relative_geometry_used_only_for_selection_quality': True,
        'actual_seed_and_third_camera_roles_verified_with_native_pose_gates': True,
        'coarse_baseline_pair_extension_used_for_quality_and_final_SfM_graph': True,
        'original_strongest_eight_geometry_preserved_before_baseline_extension': True,
        'maximum_geometry_neighbor_union': 12, 'maximum_geometry_pair_attempts': 50,
        'existing_coarse_SfM_pair_proposals_used_before_freeze': True,
        'temporal_and_coarse_spatial_pair_proposals_require_actual_native_matching': True,
        'native_PnP_triangulation_and_reprojection_thresholds_unchanged': True,
        'connectivity_rule': 'bounded_ranked_native_graph_pairs_then_unique_largest_component',
        'connectivity_checked_before_manifest_freeze': True,
        'initial_manifest_frozen_after_this_check': True, 'original_scalar_quality_and_coarse_gates_unchanged': True}
    atomic_json(root / 'CONTRACT.json', contract)
    camera = pycolmap.Camera(model='OPENCV', width=info['width'], height=info['height'], params=params)
    exported, rejected, history, rejection_reasons = set(), set(), [], {}
    geometry_cache = {}
    index = native.McapIndex(args.raw)

    def allowed(row):
        return row['image_name'] not in rejected

    while True:
        selected, eligible = choose(rows, policy, sharpness_min, baseline, additional_quality_gate=allowed)
        names = {row['image_name'] for row in selected}
        new = {row['camera_frame']: row for row in selected if row['image_name'] not in exported}
        if not selected or len(exported) + len(new) > contract['maximum_candidates']:
            raise ValueError('Initial trackability cannot find a qualified shortlist within its frozen candidate budget')
        pass_root = root / ('pass_' + str(len(history) + 1).zfill(3))
        pass_root.mkdir()
        for frame, message in index.iter_selected_messages(baseline.CAMERA_TOPIC, new):
            stamp, optical, pixels = native.decode_bayer(message['payload'])
            row = new[frame]
            if int(stamp) != row['camera_header_ns'] or optical != info['frame_id']:
                raise ValueError('Trackability export changed its original camera measurement')
            ok, encoded = cv2.imencode('.jpg', native.demosaic_rggb8_to_bgr(pixels),
                [cv2.IMWRITE_JPEG_QUALITY, policy['jpeg_quality']])
            if not ok:
                raise RuntimeError('Trackability candidate JPEG encoding failed')
            with (images / row['image_name']).open('xb') as stream:
                stream.write(encoded.tobytes())
            exported.add(row['image_name'])
        if not names <= exported:
            raise ValueError('The nominated raw trackability images were not all exported')
        timings = {}

        def execute(label, command):
            remaining = contract['maximum_seconds'] - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError('Initial trackability exhausted its frozen wall-time budget')
            timings[label] = run_native(command, pass_root, label, min(600, remaining), True,
                lease_filename=label + '_GPU_LEASE.json', lease_label='initial_trackability_' + label)

        if new:
            image_list = pass_root / 'new_images.txt'
            image_list.write_text('\n'.join(sorted(row['image_name'] for row in new.values())) + '\n', encoding='utf-8')
            command = [COLMAP, 'feature_extractor', '--database_path', database, '--image_path', images,
                '--image_list_path', image_list, '--FeatureExtraction.use_gpu', '1', '--FeatureExtraction.gpu_index', '0',
                '--FeatureExtraction.num_threads', '8', '--FeatureExtraction.max_image_size', '1600',
                '--SiftExtraction.max_num_features', '8192', '--default_random_seed', '0']
            if len(history):
                with closing(sqlite3.connect(database)) as connection:
                    cameras = list(connection.execute('SELECT camera_id FROM cameras'))
                if len(cameras) != 1:
                    raise ValueError('Trackability feature extraction changed its shared optical camera')
                command.extend(['--ImageReader.existing_camera_id', cameras[0][0]])
            else:
                command.extend(['--ImageReader.camera_model', 'OPENCV', '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_params', ','.join(map(str, params))])
            execute('features', command)
        with closing(sqlite3.connect(database)) as connection:
            image_ids = {name: int(key) for key, name in connection.execute('SELECT image_id,name FROM images')}
            attempted = {int(pair) for pair, in connection.execute('SELECT pair_id FROM matches')}
        temporal = temporal_pairs(selected, image_ids, attempted)
        pairs, pair_trace = quality_pairs(native.candidate_pairs, selected, image_ids, attempted, temporal)
        atomic_json(pass_root / 'PAIR_PROPOSALS.json', pair_trace)
        if pairs:
            pair_file = pass_root / 'quality_pairs.txt'
            pair_file.write_text(''.join(a + ' ' + b + '\n' for a, b in pairs), encoding='utf-8')
            execute('matching', [COLMAP, 'matches_importer', '--database_path', database,
                '--match_list_path', pair_file, '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1',
                '--FeatureMatching.gpu_index', '0', '--FeatureMatching.num_threads', '8',
                '--TwoViewGeometry.random_seed', '0', '--default_random_seed', '0'])
        with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as connection:
            decisions = evaluate_trackability(connection, names, minimum, other_views, include_feature_ids=True)
            geometry_trace = {}
            geometry = evaluate_geometry(connection, selected, decisions, camera, cache=geometry_cache,
                trace=geometry_trace, coarse_policy=policy)
        atomic_json(pass_root / 'LOCAL_GEOMETRY.json', {'protocol': GEOMETRY_PROTOCOL,
            'rows': geometry, 'cache_trace': geometry_trace, 'initial_manifest_not_yet_frozen': True,
            'full_SfM_model_or_final_poses_used': False})
        if time.monotonic() - began > contract['maximum_seconds']:
            raise TimeoutError('Initial geometric quality checking exhausted its frozen wall-time budget')
        failed = {name for name, value in decisions.items() if value['status'] != 'qualified_visual_trackability'}
        connectivity = {'status': 'pending_local_feature_support', 'cross_component_pairs_requested': 0}
        if not failed:
            def match_connections(recovery_pairs, label):
                pair_file = pass_root / (label + '_pairs.txt')
                pair_file.write_text(''.join(a + ' ' + b + '\n' for a, b in recovery_pairs), encoding='utf-8')
                execute(label, [COLMAP, 'matches_importer', '--database_path', database,
                    '--match_list_path', pair_file, '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1',
                    '--FeatureMatching.gpu_index', '0', '--FeatureMatching.num_threads', '8',
                    '--TwoViewGeometry.random_seed', '0', '--default_random_seed', '0'])
            groups, connection_trace = recover_connections(database, selected, policy, connection_minimum,
                match_connections)
            with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as connection:
                dominant = main_component(groups)
                core_decisions = evaluate_trackability(connection, dominant, minimum, other_views, include_feature_ids=True)
                core_selected = [row for row in selected if row['image_name'] in dominant]
                core_geometry_trace = {}
                core_geometry = evaluate_geometry(connection, core_selected, core_decisions, camera,
                    cache=geometry_cache, trace=core_geometry_trace, coarse_policy=policy)
            atomic_json(pass_root / 'CONNECTED_LOCAL_GEOMETRY.json', {'protocol': GEOMETRY_PROTOCOL,
                'rows': core_geometry, 'cache_trace': core_geometry_trace, 'initial_manifest_not_yet_frozen': True,
                'full_SfM_model_or_final_poses_used': False})
            if time.monotonic() - began > contract['maximum_seconds']:
                raise TimeoutError('Connected initial geometry checking exhausted its frozen wall-time budget')
            for name in names:
                if name not in dominant:
                    decisions[name]['status'] = 'abstain_disconnected_visual_component'
                else:
                    decisions[name] = core_decisions[name]
            failed = {name for name, value in decisions.items() if value['status'] != 'qualified_visual_trackability'}
            connectivity = {'status': 'refill_disconnected_or_unsupported_views' if failed else 'one_supported_component',
                'cross_component_pairs_requested': connection_trace['actual_requested_pairs'],
                'before_component_sizes': connection_trace['before_component_sizes'],
                'after_component_sizes': [len(group) for group in groups],
                'selected_main_component_names': sorted(dominant),
                'minimum_pair_inliers': connection_minimum,
                'all_cross_component_pairs_attempted': connection_trace['all_cross_component_pairs_attempted'],
                'bounded_actual_search': connection_trace,
                'classification_or_final_SfM_used': False}
            atomic_json(pass_root / 'CONNECTIVITY.json', connectivity)
        for row in selected:
            row['initial_trackability'] = decisions[row['image_name']]
        history.append({'pass': len(history) + 1, 'selected_names': sorted(names), 'new_images': len(new),
            'matching_pairs': len(pairs), 'failed_names': sorted(failed), 'timings': timings,
            'actual_quality_pair_proposals': pair_trace,
            'connectivity': connectivity})
        atomic_json(pass_root / 'DECISIONS.json', {'rows': decisions, 'initial_selection_not_yet_frozen': True})
        atomic_json(root / 'STATE.json', {'status': 'refill_quality_candidates' if failed else 'qualified_shortlist_ready',
            'history': history, 'exported_candidates': len(exported), 'elapsed_seconds': time.monotonic() - began})
        print('initial_trackability', len(history), 'candidates', len(exported), 'failed', len(failed), flush=True)
        if not failed:
            with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as connection:
                parent_map = spanning_parents(connection, names, connection_minimum)
            for row in selected:
                row['initial_connectivity_parent_image_name'] = parent_map[row['image_name']]
            break
        if not failed - rejected:
            raise RuntimeError('Initial trackability refill made no progress')
        rejection_reasons.update({name: decisions[name]['status'] for name in failed})
        rejected.update(failed)
    for row in rows:
        if row['image_name'] in rejected:
            row['decision'] = 'rejected_initial_visual_trackability'
            row['initial_trackability_rejection_reason'] = rejection_reasons[row['image_name']]
    value = {'status': 'initial_raw_trackability_qualified_before_manifest', **contract,
        'selected_count': len(selected), 'checked_candidates': len(exported), 'rejected_names': sorted(rejected),
        'passes': len(history), 'elapsed_seconds': time.monotonic() - began,
        'final_shortlist_connected_by_actual_native_matches': True,
        'rejected_reasons': rejection_reasons,
        'actual_three_view_SfM_and_BA_still_required': True, 'cross_attempt_cache_used': False}
    atomic_json(Path(args.output) / 'INITIAL_TRACKABILITY_RECEIPT.json', value)
    return selected, eligible


@contextmanager
def trackability_selection(native, args):
    if args.mode != 'optimized':
        yield
        return
    spatial = importlib.import_module('spatial_novelty')
    original_context = spatial.spatial_novelty_execution

    @contextmanager
    def compose(enabled=True):
        # The existing optimizer validates and compiles the original selector first.
        with original_context(enabled) as trace:
            entry = importlib.import_module('run_initial_pipeline')
            original = native.selection_engine.choose_batched
            entry_original = entry.choose_batched
            if original is not entry_original:
                raise ValueError('The optimized initial selector aliases have different ownership')

            def choose(rows, policy, sharpness_min, baseline):
                return select(native, args, original, rows, policy, sharpness_min, baseline)

            try:
                native.selection_engine.choose_batched = entry.choose_batched = choose
                yield trace
            finally:
                native.selection_engine.choose_batched, entry.choose_batched = original, entry_original

    try:
        spatial.spatial_novelty_execution = compose
        yield
    finally:
        spatial.spatial_novelty_execution = original_context
