"""Require measured local three-view geometry before freezing raw frame selection."""

from itertools import combinations
import hashlib
import json

import cv2
import numpy as np
import pycolmap
from coarse_baseline_diversity import checked_centers, diverse_neighbors

PAIR_BASE = 2147483647
PROTOCOL = 'baseline_diverse_role_complete_three_view_support_v4'
FAST_NEIGHBORS = 4
MAXIMUM_NEIGHBORS = 8
WINDOW_NS = 800_000_000


def pair_matches(connection, first, second):
    a, b = sorted((first, second))
    row = connection.execute('SELECT rows,data FROM two_view_geometries WHERE pair_id=?',
        (PAIR_BASE * a + b,)).fetchone()
    if row is None or not row[0]:
        return np.empty((0, 2), np.uint32)
    matches = np.frombuffer(row[1], np.uint32).reshape(row[0], 2)
    counts = dict(connection.execute('SELECT image_id,rows FROM keypoints WHERE image_id IN (?,?)', (a, b)))
    if set(counts) != {a, b} or np.any(matches[:, 0] >= counts[a]) or np.any(matches[:, 1] >= counts[b]):
        raise ValueError('Native pair support lies outside its actual keypoint arrays')
    if len(np.unique(matches[:, 0])) != len(matches) or len(np.unique(matches[:, 1])) != len(matches):
        raise ValueError('Native pair support contains duplicate keypoint assignments')
    return matches if first == a else matches[:, ::-1]


def geometric_support(connection, name, subjects, camera, allowed_features, *, witness_pair=None,
        verified_camera=None, verified_allowed_features=None, coarse_centers=None, coarse_policy=None):
    """Use private relative geometry only; do not publish a camera pose or scale."""
    if name not in subjects or len(set(key for key, _ in subjects.values())) != len(subjects):
        raise ValueError('Local geometry requires distinct actual subject image identities')
    if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
        raise ValueError('Local geometry requires finite actual optical calibration')
    output_name = name if verified_camera is None else verified_camera
    if output_name not in subjects or (output_name != name and verified_allowed_features is None):
        raise ValueError('A seed camera needs its actual active image and qualified feature identities')
    verified_allowed = None if verified_allowed_features is None else set(map(int, verified_allowed_features))
    mapper = pycolmap.IncrementalMapperOptions()
    tri = pycolmap.IncrementalTriangulatorOptions()
    minimum = mapper.abs_pose_min_num_inliers
    key, stamp = subjects[name]
    neighbors = []
    for other, (other_key, other_stamp) in subjects.items():
        if other != name:
            matches = pair_matches(connection, key, other_key)
            if len(matches) >= minimum:
                neighbors.append((other, other_key, matches))
    neighbors.sort(key=lambda row: (-len(row[2]), row[0]))
    all_neighbors = neighbors
    neighbors = neighbors[:MAXIMUM_NEIGHBORS]
    if witness_pair is not None:
        if len(witness_pair) != 2 or len(set(witness_pair)) != 2 or name in witness_pair:
            raise ValueError('A recorded geometric witness must contain two distinct other images')
        neighbors = []
        for other in witness_pair:
            if other not in subjects:
                raise ValueError('A recorded geometric witness is not an actual active subject')
            other_key = subjects[other][0]
            matches = pair_matches(connection, key, other_key)
            if len(matches) >= minimum:
                neighbors.append((other, other_key, matches))

    def pixels(image_id):
        stored = connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',
            (image_id,)).fetchone()
        if stored is None or not stored[0] or stored[1] < 2:
            raise ValueError('Local geometry requires the actual native keypoints')
        values = np.frombuffer(stored[2], np.float32).reshape(stored[0], stored[1])[:, :2].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Native keypoint coordinates are nonfinite')
        return values

    target_pixels = pixels(key)
    allowed = set(map(int, allowed_features))
    if any(feature < 0 or feature >= len(target_pixels) for feature in allowed):
        raise ValueError('Qualified feature support lies outside the actual query keypoints')
    if verified_allowed is not None:
        output_pixels = pixels(subjects[output_name][0])
        if any(feature < 0 or feature >= len(output_pixels) for feature in verified_allowed):
            raise ValueError('Qualified seed support lies outside the actual verified camera keypoints')
    qualified, attempts, witness_features, witness_names = set(), [], [], []
    source_features, source_names, camera_role = [], [], 'independent_third_view'
    fast_names = {row[0] for row in neighbors[:FAST_NEIGHBORS]}
    pairs = list(combinations(neighbors[:FAST_NEIGHBORS], 2))
    pairs += [(a, b) for a, b in combinations(neighbors, 2)
        if not {a[0], b[0]} <= fast_names]
    extension_pairs = 0
    if (coarse_centers is None) != (coarse_policy is None):
        raise ValueError('Coarse-baseline search requires both current coarse cameras and original sampling policy')
    if coarse_centers is not None and witness_pair is None:
        by_name = {row[0]: row for row in all_neighbors}
        chosen = diverse_neighbors(name, by_name, coarse_centers, coarse_policy,
            kept=[row[0] for row in neighbors[:FAST_NEIGHBORS]], maximum=MAXIMUM_NEIGHBORS)
        chosen += [row[0] for row in all_neighbors if row[0] not in chosen][:MAXIMUM_NEIGHBORS - len(chosen)]
        tried = {frozenset((a[0], b[0])) for a, b in pairs}
        extension = [(by_name[a], by_name[b]) for a, b in combinations(chosen, 2)
            if frozenset((a, b)) not in tried]
        pairs += extension
        extension_pairs = len(extension)
    for first, second in pairs:
        first_name, first_id, first_matches = first
        second_name, second_id, second_matches = second
        if output_name not in (name, first_name, second_name):
            continue
        first_map = {int(a): int(b) for a, b in first_matches}
        second_map = {int(a): int(b) for a, b in second_matches}
        seed_matches = pair_matches(connection, first_id, second_id)
        seed_set = set(map(tuple, seed_matches.tolist()))
        common = sorted(feature for feature in allowed & first_map.keys() & second_map.keys()
            if (first_map[feature], second_map[feature]) in seed_set)
        input_features = list(common)
        row = {'anchor_names': [first_name, second_name], 'closed_three_view_features': len(common),
            'qualified_features': 0, 'status': 'abstain_insufficient_closed_three_view_correspondences'}
        attempts.append(row)
        if len(common) < minimum:
            continue
        first_pixels, second_pixels = pixels(first_id), pixels(second_id)
        if (np.any(seed_matches[:, 0] >= len(first_pixels)) or np.any(seed_matches[:, 1] >= len(second_pixels))):
            raise ValueError('Native seed matches point outside the actual keypoint arrays')
        options = pycolmap.TwoViewGeometryOptions()
        options.compute_relative_pose = True
        options.ransac.random_seed = 0
        geometry = pycolmap.estimate_calibrated_two_view_geometry(camera, first_pixels, camera,
            second_pixels, seed_matches.copy(), options)
        pose = geometry.cam2_from_cam1
        if pose is None or not np.isfinite(pose.matrix()).all():
            row['status'] = 'abstain_no_finite_calibrated_seed_pose'
            continue
        a_indices = np.asarray([first_map[feature] for feature in common])
        b_indices = np.asarray([second_map[feature] for feature in common])
        if np.any(a_indices >= len(first_pixels)) or np.any(b_indices >= len(second_pixels)):
            raise ValueError('A closed feature track points outside its actual anchor image')
        a_xy, b_xy = first_pixels[a_indices], second_pixels[b_indices]
        rays_a, rays_b = camera.cam_from_img(a_xy), camera.cam_from_img(b_xy)
        finite = np.isfinite(rays_a).all(axis=1) & np.isfinite(rays_b).all(axis=1)
        common = np.asarray(common)[finite]
        a_xy, b_xy, rays_a, rays_b = a_xy[finite], b_xy[finite], rays_a[finite], rays_b[finite]
        if len(common) < minimum:
            row['status'] = 'abstain_nonfinite_seed_bearings'
            continue
        xyz_h = cv2.triangulatePoints(np.eye(3, 4), pose.matrix(), rays_a.T, rays_b.T).T
        nonzero = np.isfinite(xyz_h).all(axis=1) & (np.abs(xyz_h[:, 3]) > np.finfo(np.float64).eps)
        xyz = np.full((len(xyz_h), 3), np.nan)
        xyz[nonzero] = xyz_h[nonzero, :3] / xyz_h[nonzero, 3:4]
        xyz_b = xyz @ pose.rotation.matrix().T + pose.translation
        ray_b_world = xyz - pose.inverse().translation
        denominator = np.linalg.norm(xyz, axis=1) * np.linalg.norm(ray_b_world, axis=1)
        defined = np.isfinite(denominator) & (denominator > 0)
        cosine = np.full(len(xyz), np.nan)
        cosine[defined] = np.sum(xyz[defined] * ray_b_world[defined], axis=1) / denominator[defined]
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz_b[:, 2] > 0)
        valid &= np.isfinite(angles) & (angles >= tri.min_angle)
        valid &= np.linalg.norm(camera.img_from_cam(xyz) - a_xy, axis=1) <= tri.complete_max_reproj_error
        valid &= np.linalg.norm(camera.img_from_cam(xyz_b) - b_xy, axis=1) <= tri.complete_max_reproj_error
        row['valid_seed_points'] = int(np.count_nonzero(valid))
        if np.count_nonzero(valid) < minimum:
            row['status'] = 'abstain_seed_geometry'
            continue
        features = common[valid]
        xy, xyz = target_pixels[features], xyz[valid]
        estimation = pycolmap.AbsolutePoseEstimationOptions()
        estimation.ransac.max_error = mapper.abs_pose_max_error
        estimation.ransac.min_inlier_ratio = mapper.abs_pose_min_inlier_ratio
        estimation.ransac.random_seed = 0
        estimate = pycolmap.estimate_and_refine_absolute_pose(xy, xyz, camera,
            estimation, pycolmap.AbsolutePoseRefinementOptions())
        if estimate is None or not np.isfinite(estimate['cam_from_world'].matrix()).all():
            row['status'] = 'abstain_no_finite_genuine_third_pose'
            continue
        pose_c = estimate['cam_from_world']
        xyz_c = xyz @ pose_c.rotation.matrix().T + pose_c.translation
        projection = camera.img_from_cam(xyz_c)
        valid = np.asarray(estimate['inlier_mask'], bool) & np.isfinite(xyz_c).all(axis=1) & (xyz_c[:, 2] > 0)
        valid &= np.isfinite(projection).all(axis=1)
        valid &= np.linalg.norm(projection - xy, axis=1) <= tri.complete_max_reproj_error
        row['qualified_features'] = int(np.count_nonzero(valid))
        row['native_pnp_verified_inlier_ratio'] = float(np.mean(valid))
        if np.count_nonzero(valid) >= minimum and np.mean(valid) >= mapper.abs_pose_min_inlier_ratio:
            query_features, query_inputs = features[valid], input_features
            actual_witness_names = [first_name, second_name]
            if output_name != name:
                mapping = first_map if output_name == first_name else second_map
                seed_indices = np.asarray([mapping[int(feature)] for feature in features[valid]])
                permitted = np.asarray([int(feature) in verified_allowed for feature in seed_indices])
                seed_indices, seed_xyz = seed_indices[permitted], xyz[valid][permitted]
                if len(seed_indices) < minimum:
                    row['status'] = 'abstain_seed_camera_without_qualified_feature_support'
                    continue
                seed_pixels = pixels(subjects[output_name][0])[seed_indices]
                seed_estimate = pycolmap.estimate_and_refine_absolute_pose(seed_pixels, seed_xyz, camera,
                    estimation, pycolmap.AbsolutePoseRefinementOptions())
                if seed_estimate is None or not np.isfinite(seed_estimate['cam_from_world'].matrix()).all():
                    row['status'] = 'abstain_seed_camera_native_pose_readback'
                    continue
                seed_pose = seed_estimate['cam_from_world']
                seed_camera_xyz = seed_xyz @ seed_pose.rotation.matrix().T + seed_pose.translation
                seed_projection = camera.img_from_cam(seed_camera_xyz)
                seed_valid = np.asarray(seed_estimate['inlier_mask'], bool)
                seed_valid &= np.isfinite(seed_camera_xyz).all(axis=1) & (seed_camera_xyz[:, 2] > 0)
                seed_valid &= np.isfinite(seed_projection).all(axis=1)
                seed_valid &= np.linalg.norm(seed_projection - seed_pixels, axis=1) <= tri.complete_max_reproj_error
                row['seed_camera_qualified_features'] = int(seed_valid.sum())
                row['seed_camera_native_inlier_ratio'] = float(seed_valid.mean())
                if seed_valid.sum() < minimum or seed_valid.mean() < mapper.abs_pose_min_inlier_ratio:
                    row['status'] = 'abstain_seed_camera_geometry_readback'
                    continue
                query_features, query_inputs = seed_indices[seed_valid], seed_indices.tolist()
                actual_witness_names = [name, second_name if output_name == first_name else first_name]
                camera_role = 'first_seed_camera' if output_name == first_name else 'second_seed_camera'
            row['status'] = 'actual_calibrated_three_view_support'
            qualified.update(map(int, query_features))
            witness_features, witness_names = query_inputs, actual_witness_names
            source_features, source_names = input_features, [first_name, second_name]
            break
        row['status'] = 'abstain_third_view_geometry'
    return {'name': output_name, 'qualified_distinct_features': len(qualified), 'minimum_features': minimum,
        'qualified_feature_indices': sorted(qualified),
        'witness_input_feature_indices': witness_features, 'witness_anchor_names': witness_names,
        'source_witness_query_name': name, 'source_witness_anchor_names': source_names,
        'source_witness_input_feature_indices': source_features, 'verified_camera_role': camera_role,
        'minimum_angle_degrees': tri.min_angle, 'maximum_reprojection_px': tri.complete_max_reproj_error,
        'maximum_neighbor_candidates': MAXIMUM_NEIGHBORS + (FAST_NEIGHBORS if extension_pairs else 0),
        'maximum_geometry_pair_attempts': len(pairs), 'coarse_baseline_extension_pairs': extension_pairs,
        'original_strongest_eight_search_preserved_before_baseline_extension': True,
        'temporal_window_ns': WINDOW_NS,
        'fast_neighbor_candidates': FAST_NEIGHBORS,
        'temporal_limit_applied_to_geometric_witness': False,
        'witness_scope': 'actual_native_matches_from_current_temporal_and_coarse_SfM_candidate_graph',
        'status': 'qualified_three_view_geometry' if len(qualified) >= minimum else 'abstain_three_view_geometry',
        'relative_geometry_role': 'selection_quality_only_not_a_full_SfM_pose_or_metric_scale',
        'attempts': attempts}


def proof_inputs_valid(proof, name, subjects, allowed):
    """Keep the canonical triad and both native pose input sets qualified."""
    source = proof['source_witness_query_name']
    members = [source, *proof['source_witness_anchor_names']]
    query_members = [name, *proof['witness_anchor_names']]
    return (proof['name'] == name and len(set(members)) == 3 and len(set(query_members)) == 3
        and set(members) == set(query_members) and set(members) <= subjects.keys()
        and set(proof['source_witness_input_feature_indices']) <= set(allowed[source])
        and set(proof['witness_input_feature_indices']) <= set(allowed[name]))


def replay_geometry_proof(connection, name, proof, subjects, camera, allowed):
    """Recompute a canonical three-view proof in its actual camera role."""
    if not proof_inputs_valid(proof, name, subjects, allowed):
        raise ValueError('A geometric proof lost its actual canonical views or unambiguous native features')
    source = proof['source_witness_query_name']
    return geometric_support(connection, source, subjects, camera,
        proof['source_witness_input_feature_indices'], witness_pair=proof['source_witness_anchor_names'],
        verified_camera=name, verified_allowed_features=proof['witness_input_feature_indices'])


def witness_identity(connection, name, witnesses, subjects, camera):
    """Bind a computed proof to its actual pixel, match and calibration inputs."""
    names = [name, *witnesses]
    mapper = pycolmap.IncrementalMapperOptions()
    tri = pycolmap.IncrementalTriangulatorOptions()
    metadata = {'subjects': [[value, *subjects[value]] for value in names],
        'camera_model': camera.model.name, 'camera_dimensions': [camera.width, camera.height],
        'camera_params': camera.params.tolist(), 'protocol': PROTOCOL,
        'native_pose_gates': [mapper.abs_pose_min_num_inliers, mapper.abs_pose_min_inlier_ratio, mapper.abs_pose_max_error],
        'native_triangulation_gates': [tri.min_angle, tri.complete_max_reproj_error],
        'native_two_view_options': pycolmap.TwoViewGeometryOptions().todict()}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode())
    for value in names:
        row = connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (subjects[value][0],)).fetchone()
        if row is None:
            raise ValueError('A computed geometry proof lost its actual pixel input')
        digest.update(json.dumps(list(row[:2])).encode())
        digest.update(row[2])
    for first, second in combinations(names, 2):
        digest.update(pair_matches(connection, subjects[first][0], subjects[second][0]).tobytes())
    return digest.hexdigest()


def evaluate_geometry(connection, selected, decisions, camera, *, cache=None, trace=None, coarse_policy=None):
    ids = {name: int(key) for key, name in connection.execute('SELECT image_id,name FROM images')}
    active = {row['image_name']: (ids[row['image_name']], row['camera_header_ns']) for row in selected
        if decisions[row['image_name']]['status'] == 'qualified_visual_trackability'}
    allowed_by_name = {name: decisions[name]['supported_feature_indices'] for name in active}
    centers = checked_centers(selected, coarse_policy) if coarse_policy is not None else None
    results, hits, role_rechecks = {}, 0, 0
    for name in sorted(active):
        allowed = decisions[name]['supported_feature_indices']
        cached = cache.get(name) if cache is not None else None
        if cached:
            proof = cached['result']
            witnesses = proof['witness_anchor_names']
            if (proof_inputs_valid(proof, name, active, allowed_by_name)
                    and cached['identity'] == witness_identity(connection, name, witnesses, active, camera)):
                results[name] = proof
                hits += 1
        if name not in results:
            results[name] = geometric_support(connection, name, active, camera, allowed,
                coarse_centers=centers, coarse_policy=coarse_policy)
    triads = {}
    for proof in results.values():
        if proof['status'] == 'qualified_three_view_geometry':
            identity = (proof['source_witness_query_name'], tuple(proof['source_witness_anchor_names']),
                tuple(proof['source_witness_input_feature_indices']))
            triads.setdefault(identity, proof)
    for source_proof in triads.values():
        source = source_proof['source_witness_query_name']
        members = [source, *source_proof['source_witness_anchor_names']]
        for name in members:
            if results[name]['status'] == 'qualified_three_view_geometry':
                continue
            role_rechecks += 1
            verified = geometric_support(connection, source, active, camera,
                source_proof['source_witness_input_feature_indices'],
                witness_pair=source_proof['source_witness_anchor_names'], verified_camera=name,
                verified_allowed_features=allowed_by_name[name])
            if verified['status'] == 'qualified_three_view_geometry':
                results[name] = verified
    if cache is not None:
        for name, proof in results.items():
            if proof['status'] == 'qualified_three_view_geometry' and proof_inputs_valid(proof, name, active, allowed_by_name):
                cache[name] = {'result': proof,
                    'identity': witness_identity(connection, name, proof['witness_anchor_names'], active, camera)}
    for name, result in results.items():
        decisions[name]['local_three_view_geometry'] = result
        if result['status'] != 'qualified_three_view_geometry':
            decisions[name]['status'] = 'abstain_local_three_view_geometry'
    if trace is not None:
        trace.update(same_attempt_positive_proof_cache_hits=hits, actual_native_geometry_evaluations=len(active) - hits,
            actual_native_camera_role_rechecks=role_rechecks,
            coarse_baseline_alternative_search_used=coarse_policy is not None,
            qualified_seed_role_cameras=sum(row['status'] == 'qualified_three_view_geometry'
                and row['verified_camera_role'] != 'independent_third_view' for row in results.values()),
            actual_pixel_match_calibration_and_active_feature_identities_checked=True,
            negative_decisions_cached=False, cross_attempt_cache_loaded=False)
    return results
