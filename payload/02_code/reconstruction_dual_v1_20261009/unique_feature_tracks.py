"""Exclude ambiguous feature components before counting multiview support."""

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PAIR_BASE = 2147483647
POINT_BASE = 2**32


def reference_unambiguous_support(connection, image_ids, keypoint_counts, minimum_views):
    """Keep the original scalar oracle for exact regression comparisons."""
    ordered = sorted(image_ids)
    bits = {key: 1 << index for index, key in enumerate(ordered)}
    parent, degree = {}, defaultdict(int)

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    for pair, count, payload in connection.execute('SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0'):
        first, second = divmod(pair, PAIR_BASE)
        if first not in bits or second not in bits:
            continue
        if first == second:
            raise ValueError('A single image cannot provide two distinct feature views')
        matches = np.frombuffer(payload, np.uint32).reshape(count, 2)
        if np.any(matches[:, 0] >= keypoint_counts[first]) or np.any(matches[:, 1] >= keypoint_counts[second]):
            raise ValueError('A native verified match points outside its actual feature array')
        for a, b in matches:
            u, v = POINT_BASE * first + int(a), POINT_BASE * second + int(b)
            parent.setdefault(u, u)
            parent.setdefault(v, v)
            left, right = find(u), find(v)
            if left != right:
                parent[max(left, right)] = min(left, right)
            degree[u] |= bits[second]
            degree[v] |= bits[first]
    nodes = np.fromiter(parent, np.uint64, count=len(parent))
    roots = np.asarray([find(int(node)) for node in nodes], dtype=np.uint64)
    membership = np.empty(len(nodes), dtype=[('root', np.uint64), ('image', np.uint32)])
    membership['root'] = roots
    membership['image'] = nodes // POINT_BASE
    distinct, occurrences = np.unique(membership, return_counts=True)
    ambiguous = set(map(int, distinct['root'][occurrences > 1]))
    component_ids, view_counts = np.unique(distinct['root'], return_counts=True)
    qualified = set(map(int, component_ids[view_counts >= minimum_views])) - ambiguous
    support = {key: {} for key in ordered}
    for node, root in zip(nodes, roots):
        key = int(node)
        if int(root) in qualified:
            image_id, point_index = divmod(key, POINT_BASE)
            support[image_id][point_index] = degree[key]
    return support, {'feature_components': len(component_ids),
        'ambiguous_components_excluded': len(ambiguous), 'unambiguous_multiview_components': len(qualified),
        'support_definition': 'one_keypoint_per_image_in_each_connected_feature_component',
        'ambiguity_policy': 'conservative_component_abstention_not_geometric_or_pose_acceptance'}


def unambiguous_support(connection, image_ids, keypoint_counts, minimum_views):
    """Use the same graph and support rules with a compiled sparse graph kernel."""
    ordered = sorted(image_ids)
    bits = {key: 1 << index for index, key in enumerate(ordered)}
    blocks = []
    for pair, count, payload in connection.execute(
            'SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0'):
        first, second = divmod(pair, PAIR_BASE)
        if first not in bits or second not in bits:
            continue
        if first == second:
            raise ValueError('A single image cannot provide two distinct feature views')
        matches = np.frombuffer(payload, np.uint32).reshape(count, 2)
        if (np.any(matches[:, 0] >= keypoint_counts[first])
                or np.any(matches[:, 1] >= keypoint_counts[second])):
            raise ValueError('A native verified match points outside its actual feature array')
        nodes = matches.astype(np.uint64)
        nodes[:, 0] += POINT_BASE * first
        nodes[:, 1] += POINT_BASE * second
        blocks.append(nodes)
    support = {key: {} for key in ordered}
    if not blocks:
        return support, {'feature_components': 0, 'ambiguous_components_excluded': 0,
            'unambiguous_multiview_components': 0,
            'support_definition': 'one_keypoint_per_image_in_each_connected_feature_component',
            'ambiguity_policy': 'conservative_component_abstention_not_geometric_or_pose_acceptance'}
    endpoints = np.concatenate(blocks)
    del blocks
    nodes, inverse = np.unique(endpoints, return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(endpoints), dtype=bool),
        (inverse[:, 0], inverse[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    del graph
    membership = np.empty(len(nodes), dtype=[('root', np.int32), ('image', np.uint32)])
    membership['root'] = labels
    membership['image'] = nodes // POINT_BASE
    distinct, occurrences = np.unique(membership, return_counts=True)
    view_counts = np.bincount(distinct['root'], minlength=component_count)
    ambiguous = np.bincount(distinct['root'][occurrences > 1], minlength=component_count) > 0
    qualified = (view_counts >= minimum_views) & ~ambiguous
    qualified_nodes = qualified[labels]
    first = qualified_nodes[inverse[:, 0]]
    second = qualified_nodes[inverse[:, 1]]
    # Compact indices preserve uint64 packing even when native image IDs are large.
    neighbors = np.concatenate((
        inverse[first, 0].astype(np.uint64) * POINT_BASE + endpoints[first, 1] // POINT_BASE,
        inverse[second, 1].astype(np.uint64) * POINT_BASE + endpoints[second, 0] // POINT_BASE))
    neighbors = np.unique(neighbors)
    degree = {}
    for value in neighbors:
        node_index, other_image = divmod(int(value), POINT_BASE)
        degree[node_index] = degree.get(node_index, 0) | bits[other_image]
    for node_index, mask in degree.items():
        image_id, point_index = divmod(int(nodes[node_index]), POINT_BASE)
        support[image_id][point_index] = mask
    return support, {'feature_components': int(component_count),
        'ambiguous_components_excluded': int(ambiguous.sum()),
        'unambiguous_multiview_components': int(qualified.sum()),
        'support_definition': 'one_keypoint_per_image_in_each_connected_feature_component',
        'ambiguity_policy': 'conservative_component_abstention_not_geometric_or_pose_acceptance'}
