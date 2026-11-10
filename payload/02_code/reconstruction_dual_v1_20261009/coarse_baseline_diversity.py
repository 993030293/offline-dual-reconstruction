"""Propose bounded coarse-baseline alternatives without accepting a visual pose."""

from contextlib import contextmanager
from itertools import combinations

import numpy as np

PROTOCOL = 'coarse_baseline_cell_candidate_extension_v1'
ANCHORS = 4


def checked_centers(rows, policy):
    cell, minimum = float(policy['position_cell_m']), float(policy['translation_min_m'])
    if not np.isfinite([cell, minimum]).all() or cell <= 0 or minimum <= 0:
        raise ValueError('Coarse-baseline proposals require positive finite original sampling distances')
    centers = {}
    for row in rows:
        name, value = row['image_name'], np.asarray(row['C_W_m'], dtype=np.float64)
        if name in centers or value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Coarse-baseline proposals require unique images and finite metric camera positions')
        if np.max(np.abs(value), initial=0.) / cell > 2.**40:
            raise ValueError('Coarse-baseline position cells exceed their reliable numerical range')
        centers[name] = value
    return centers


def diverse_neighbors(name, candidates, centers, policy, *, kept=(), maximum=8):
    """Retain the strongest prefix and add nearest distinct metric position cells."""
    cell, minimum = float(policy['position_cell_m']), float(policy['translation_min_m'])
    if name not in centers or len(set(kept)) != len(kept) or len(kept) > maximum:
        raise ValueError('A coarse-baseline neighborhood has invalid actual image identities')
    chosen = list(kept)
    candidates = sorted(set(candidates) - {name} - set(chosen))
    if not set([name, *chosen, *candidates]) <= centers.keys():
        raise ValueError('A coarse-baseline neighbor has no actual current coarse camera')
    key = lambda value: tuple(np.floor(centers[value] / cell).astype(np.int64))
    seen = {key(name), *[key(value) for value in chosen]}
    if len(chosen) == maximum or not candidates:
        return chosen
    distances = np.linalg.norm(np.asarray([centers[value] for value in candidates]) - centers[name], axis=1)
    if not np.isfinite(distances).all():
        raise ValueError('A coarse-baseline proposal has nonfinite relative metric distance')
    for index in np.argsort(distances, kind='stable'):
        other = candidates[index]
        if distances[index] >= minimum and key(other) not in seen:
            chosen.append(other)
            seen.add(key(other))
            if len(chosen) == maximum:
                break
    return chosen


def baseline_pairs(rows, policy):
    centers = checked_centers(rows, policy)
    indices = {row['image_name']: number for number, row in enumerate(rows)}
    pairs = set()
    for name in centers:
        anchors = diverse_neighbors(name, centers, centers, policy, maximum=ANCHORS)
        for first, second in combinations([name, *anchors], 2):
            pairs.add(tuple(sorted((indices[first], indices[second]))))
    return pairs, {'protocol': PROTOCOL, 'actual_frames': len(rows), 'actual_unique_baseline_pairs': len(pairs),
        'maximum_distinct_anchor_cells': ANCHORS, 'maximum_pairs_proposed_per_camera': 10,
        'position_cell_m': policy['position_cell_m'], 'minimum_translation_m': policy['translation_min_m'],
        'coarse_role': 'proposals_only_actual_matching_and_calibrated_geometry_required',
        'class_dimensions_or_final_SfM_used': False}


def witness_pairs(rows):
    """Carry checked same-attempt triangle edges into the fresh SfM pair file."""
    indices = {row['image_name']: number for number, row in enumerate(rows)}
    all_ready = all(row.get('initial_trackability', {}).get('status') == 'qualified_visual_trackability'
        for row in rows)
    pairs = set()
    for row in rows:
        proof = row.get('initial_trackability', {}).get('local_three_view_geometry')
        if not proof or proof['status'] != 'qualified_three_view_geometry':
            continue
        members = [proof['source_witness_query_name'], *proof['source_witness_anchor_names']]
        if proof['name'] != row['image_name'] or len(set(members)) != 3 or row['image_name'] not in members:
            raise ValueError('A same-attempt geometric witness changed its actual triangle identities')
        if not set(members) <= indices.keys():
            if all_ready:
                raise ValueError('A frozen valid geometric witness lost a nominated camera')
            continue
        for first, second in combinations(members, 2):
            pairs.add(tuple(sorted((indices[first], indices[second]))))
    return pairs


def connected_pairs(rows):
    """Carry the same-attempt native spanning witness into the fresh SfM graph."""
    if not any('initial_connectivity_parent_image_name' in row for row in rows):
        return set()
    if not all('initial_connectivity_parent_image_name' in row for row in rows):
        raise ValueError('A final connection witness has incomplete camera roles')
    indices = {row['image_name']: key for key, row in enumerate(rows)}
    roots = [row['image_name'] for row in rows if row['initial_connectivity_parent_image_name'] is None]
    if len(indices) != len(rows) or len(roots) != 1:
        raise ValueError('A final connection witness requires distinct cameras and one root')
    edges, adjacency = set(), {name: [] for name in indices}
    for row in rows:
        name, parent = row['image_name'], row['initial_connectivity_parent_image_name']
        if parent is None:
            continue
        if parent == name or parent not in indices:
            raise ValueError('A final connection witness lost a nominated camera')
        edges.add(tuple(sorted((indices[name], indices[parent]))))
        adjacency[name].append(parent)
        adjacency[parent].append(name)
    visited, queue = {roots[0]}, [roots[0]]
    for name in queue:
        for other in adjacency[name]:
            if other not in visited:
                visited.add(other)
                queue.append(other)
    if len(edges) != len(rows) - 1 or visited != indices.keys():
        raise ValueError('A final connection witness contains a cycle or disconnected cameras')
    return edges


@contextmanager
def candidate_pair_extension(native, enabled, policy):
    if not enabled:
        yield
        return
    original = native.candidate_pairs

    def proposed(rows, *args, **kwargs):
        preserved, original_trace = original(rows, *args, **kwargs)
        if any(not 0 <= first < second < len(rows) for first, second in preserved):
            raise ValueError('The original candidate graph changed its actual image index contract')
        additional, trace = baseline_pairs(rows, policy)
        verified = witness_pairs(rows)
        connected = connected_pairs(rows)
        merged = set(preserved) | additional | verified | connected
        return merged, {'protocol': PROTOCOL, 'original_candidate_graph': original_trace,
            'coarse_baseline_extension': trace, 'additional_unique_pairs': len(merged - set(preserved)),
            'same_attempt_verified_witness_triangle_pairs': len(verified),
            'same_attempt_verified_connected_spanning_pairs': len(connected),
            'actual_combined_pairs': len(merged), 'original_indexed_graph_preserved': True,
            'private_local_poses_or_3D_points_imported_into_SfM': False}

    try:
        native.candidate_pairs = proposed
        yield
    finally:
        native.candidate_pairs = original
