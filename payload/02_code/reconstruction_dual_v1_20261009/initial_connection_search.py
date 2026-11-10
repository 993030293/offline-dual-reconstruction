"""Verify native scene connections with ranked bounded batches and real tree edges."""

from collections import deque
from contextlib import closing
import sqlite3

import numpy as np

from coarse_baseline_diversity import checked_centers
from initial_connectivity import PAIR_BASE, component_recovery_pairs, matching_components

PROTOCOL = 'bounded_ranked_native_scene_connections_v1'
BATCH_PAIRS = 256
PAIRS_PER_CAMERA = 8


def ranked_pairs(connection, components, rows, policy):
    """Balance component pairs; coarse cameras rank proposals but never validate them."""
    names = {name for group in components for name in group}
    selected = {row['image_name']: row for row in rows}
    if len(selected) != len(rows) or set(selected) != names:
        raise ValueError('Connection proposals require the actual complete current camera cohort')
    centers = checked_centers(rows, policy)
    angle_step, time_step = float(policy['direction_cell_degrees']), float(policy['bridge_gap_seconds'])
    if not np.isfinite([angle_step, time_step]).all() or min(angle_step, time_step) <= 0:
        raise ValueError('Connection ranking requires positive finite original sampling intervals')
    ordered = sorted(names)
    index = {name: key for key, name in enumerate(ordered)}
    rotations = np.asarray([selected[name]['R_WC'] for name in ordered], dtype=np.float64)
    if (rotations.shape != (len(names), 3, 3) or not np.isfinite(rotations).all()
            or np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))) > 1e-8
            or np.max(np.abs(np.linalg.det(rotations) - 1.)) > 1e-8):
        raise ValueError('Connection ranking requires actual finite proper coarse camera rotations')
    positions = np.asarray([centers[name] for name in ordered])
    stamps = [int(selected[name]['camera_header_ns']) for name in ordered]
    pairs = component_recovery_pairs(connection, components)
    if not pairs:
        return []
    first = np.asarray([index[a] for a, b in pairs])
    second = np.asarray([index[b] for a, b in pairs])
    distance = np.linalg.norm(positions[first] - positions[second], axis=1) / policy['position_cell_m']
    trace = np.einsum('nij,nij->n', rotations[first], rotations[second])
    angle = np.degrees(np.arccos(np.clip((trace - 1.) / 2., -1., 1.))) / angle_step
    elapsed = np.asarray([abs(stamps[a] - stamps[b]) * 1e-9 for a, b in zip(first, second)])
    score = distance + angle + np.minimum(elapsed / time_step, PAIRS_PER_CAMERA)
    if not np.isfinite(score).all():
        raise ValueError('Connection proposal priorities are not finite')
    membership = {name: key for key, group in enumerate(components) for name in group}
    buckets = {}
    for key in np.argsort(score, kind='stable'):
        pair = pairs[key]
        group = tuple(sorted((membership[pair[0]], membership[pair[1]])))
        buckets.setdefault(group, deque()).append(pair)
    order, result = list(buckets), []
    while order:
        remaining = []
        for group in order:
            result.append(buckets[group].popleft())
            if buckets[group]:
                remaining.append(group)
        order = remaining
    return result


def recover_connections(database, rows, policy, minimum_pair_inliers, match_batch):
    """Stop only on real connectivity, exhausted proposals, or a declared search budget."""
    names = {row['image_name'] for row in rows}

    def opened():
        return closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True))

    with opened() as connection:
        groups = matching_components(connection, names, minimum_pair_inliers)
        before = [len(group) for group in groups]
        queue = ranked_pairs(connection, groups, rows, policy) if len(groups) > 1 else []
    maximum, offset, requested, batches = PAIRS_PER_CAMERA * len(names), 0, 0, []
    while len(groups) > 1 and offset < len(queue) and requested < maximum:
        membership = {name: key for key, group in enumerate(groups) for name in group}
        pairs = []
        while offset < len(queue) and len(pairs) < min(BATCH_PAIRS, maximum - requested):
            pair = queue[offset]
            offset += 1
            if membership[pair[0]] != membership[pair[1]]:
                pairs.append(pair)
        if not pairs:
            break
        label = 'connectivity_matching_' + str(len(batches) + 1).zfill(3)
        match_batch(pairs, label)
        requested += len(pairs)
        with opened() as connection:
            ids = {name: int(key) for key, name in connection.execute('SELECT image_id,name FROM images')}
            attempted = {int(pair) for pair, in connection.execute('SELECT pair_id FROM matches')}
            expected = {PAIR_BASE * min(ids[a], ids[b]) + max(ids[a], ids[b]) for a, b in pairs}
            if not expected <= attempted:
                raise ValueError('Native connection matching left an actually requested batch unattempted')
            groups = matching_components(connection, names, minimum_pair_inliers)
        batches.append({'label': label, 'actual_requested_pairs': len(pairs),
            'actual_component_sizes_after_matching': [len(group) for group in groups]})
    return groups, {'protocol': PROTOCOL, 'before_component_sizes': before,
        'after_component_sizes': [len(group) for group in groups], 'actual_requested_pairs': requested,
        'maximum_requested_pairs': maximum, 'maximum_batch_pairs': BATCH_PAIRS, 'batches': batches,
        'original_unattempted_cross_component_proposals': len(queue),
        'all_cross_component_pairs_attempted': requested == len(queue),
        'every_requested_batch_actually_attempted': True, 'real_connectivity_checked_after_each_batch': True,
        'coarse_cameras_used_only_for_proposal_ranking': True,
        'unresolved_groups_are_abstentions_not_proven_unmatchable': len(groups) > 1}


def spanning_parents(connection, names, minimum_pair_inliers):
    """Retain a maximum-inlier tree of actual nominated native graph edges."""
    names = set(names)
    images = {int(key): name for key, name in connection.execute('SELECT image_id,name FROM images')
        if name in names}
    if not names or set(images.values()) != names:
        raise ValueError('A connection witness needs all actual current image identities')
    parents = {key: key for key in images}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    edges = []
    for pair, count in connection.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>=?',
            (minimum_pair_inliers,)):
        first, second = divmod(int(pair), PAIR_BASE)
        if first in images and second in images:
            edges.append((int(count), int(pair), first, second))
    tree = {name: [] for name in names}
    for count, pair, first, second in sorted(edges, key=lambda row: (-row[0], row[1])):
        a, b = find(first), find(second)
        if a != b:
            parents[a] = b
            tree[images[first]].append(images[second])
            tree[images[second]].append(images[first])
    result, queue = {min(names): None}, deque([min(names)])
    while queue:
        first = queue.popleft()
        for other in sorted(tree[first]):
            if other not in result:
                result[other] = first
                queue.append(other)
    if set(result) != names:
        raise ValueError('The final qualified cameras do not have an actual native connected tree')
    return result
