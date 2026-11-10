"""Frozen M3 local terrain graph and grid-MRF ground classification."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow

import ground_target_extraction as base


CELL_M = 0.02


def make_floor(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normal: np.ndarray,
    classification: dict,
) -> tuple[np.ndarray, np.ndarray]:
    height = np.asarray(classification["centroid_height_m"], dtype=np.float64)
    alignment = np.asarray(classification["normal_alignment"], dtype=np.float64)
    axis_u, axis_v = base.plane_basis(normal)
    centroid = vertices[triangles].mean(axis=1)
    horizontal = np.column_stack((centroid @ axis_u, centroid @ axis_v))
    keys = np.floor((horizontal - horizontal.min(axis=0)) / CELL_M).astype(np.int32)
    shape = tuple((keys.max(axis=0) + 1).tolist())
    floor = np.full(shape, np.inf, dtype=np.float64)
    sources = (alignment >= 0.65) & (height >= -0.05) & (height <= 0.15)
    np.minimum.at(floor, (keys[sources, 0], keys[sources, 1]), height[sources])
    return floor, keys


def connected_ground_cells(floor: np.ndarray) -> tuple[np.ndarray, dict]:
    eligible = np.isfinite(floor) & (floor >= -0.05) & (floor <= 0.12)
    rows, columns = floor.shape
    exterior = np.zeros(floor.shape, dtype=bool)
    edge_rows = max(1, int(np.ceil(rows * 0.10)))
    edge_columns = max(1, int(np.ceil(columns * 0.10)))
    exterior[:edge_rows, :] = True
    exterior[-edge_rows:, :] = True
    exterior[:, :edge_columns] = True
    exterior[:, -edge_columns:] = True
    seed = eligible & exterior & (np.abs(floor) <= 0.02)
    if not np.any(seed):
        raise RuntimeError("M2 has no reliable exterior ground cells")

    index = np.arange(floor.size, dtype=np.int32).reshape(floor.shape)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for row_shift, column_shift in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        if row_shift >= rows or abs(column_shift) >= columns:
            continue
        left_row = slice(0, rows - row_shift)
        right_row = slice(row_shift, rows)
        if column_shift >= 0:
            left_column = slice(0, columns - column_shift)
            right_column = slice(column_shift, columns)
        else:
            left_column = slice(-column_shift, columns)
            right_column = slice(0, columns + column_shift)
        left = floor[left_row, left_column]
        right = floor[right_row, right_column]
        step_m = CELL_M * float(np.hypot(row_shift, column_shift))
        slope_limit = max(0.02, 0.50 * step_m)
        pair_eligible = eligible[left_row, left_column] & eligible[right_row, right_column]
        difference = np.zeros_like(left)
        np.subtract(left, right, out=difference, where=pair_eligible)
        allowed = pair_eligible & (np.abs(difference) <= slope_limit)
        a = index[left_row, left_column][allowed]
        b = index[right_row, right_column][allowed]
        row_parts.extend((a, b))
        column_parts.extend((b, a))
    if not row_parts:
        raise RuntimeError("M2 local terrain graph has no edges")
    graph = csr_matrix(
        (
            np.ones(sum(len(part) for part in row_parts), dtype=np.uint8),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(floor.size, floor.size),
    )
    component_count, labels = connected_components(graph, directed=False)
    seeded = np.zeros(component_count, dtype=bool)
    seeded[labels[np.flatnonzero(seed.ravel())]] = True
    ground = eligible & seeded[labels].reshape(floor.shape)
    return ground, {
        "eligible_cells": int(eligible.sum()),
        "seed_cells": int(seed.sum()),
        "connected_ground_cells": int(ground.sum()),
        "component_count": int(component_count),
        "seeded_component_count": int(seeded.sum()),
        "graph_nonzero_edges": int(graph.nnz),
    }


def mrf_ground_cells(floor: np.ndarray) -> tuple[np.ndarray, dict]:
    rows, columns = floor.shape
    finite = np.isfinite(floor)
    eligible = finite & (floor >= -0.05) & (floor <= 0.12)
    exterior = np.zeros(floor.shape, dtype=bool)
    edge_rows = max(1, int(np.ceil(rows * 0.10)))
    edge_columns = max(1, int(np.ceil(columns * 0.10)))
    exterior[:edge_rows] = True
    exterior[-edge_rows:] = True
    exterior[:, :edge_columns] = True
    exterior[:, -edge_columns:] = True
    ground_seed = eligible & exterior & (np.abs(floor) <= 0.02)
    if not np.any(ground_seed):
        raise RuntimeError("M3 has no reliable exterior ground seeds")

    node_count = floor.size
    source = node_count
    sink = node_count + 1
    node = np.arange(node_count, dtype=np.int32).reshape(floor.shape)
    ground_cost = np.full(floor.shape, 1000, dtype=np.int32)
    target_cost = np.ones(floor.shape, dtype=np.int32)
    height_ratio = np.zeros(floor.shape, dtype=np.float64)
    np.divide(np.abs(floor), 0.08, out=height_ratio, where=finite)
    height_ratio = np.clip(height_ratio, 0, 1)
    ground_cost[eligible] = 1 + np.rint(998 * height_ratio[eligible] ** 2).astype(np.int32)
    target_cost[eligible] = 1000 - ground_cost[eligible]
    target_cost[ground_seed] = 10_000
    ground_cost[ground_seed] = 1
    target_seed = ~eligible | (floor >= 0.08)
    ground_cost[target_seed] = 10_000
    target_cost[target_seed] = 1

    from_nodes = [np.full(node_count, source, dtype=np.int32), node.ravel()]
    to_nodes = [node.ravel(), np.full(node_count, sink, dtype=np.int32)]
    capacities = [target_cost.ravel(), ground_cost.ravel()]
    for left, right in ((node[:, :-1], node[:, 1:]), (node[:-1, :], node[1:, :])):
        left_ids = left.ravel()
        right_ids = right.ravel()
        left_floor = floor.ravel()[left_ids]
        right_floor = floor.ravel()[right_ids]
        valid = eligible.ravel()[left_ids] & eligible.ravel()[right_ids]
        delta = np.zeros(len(left_ids), dtype=np.float64)
        np.subtract(left_floor, right_floor, out=delta, where=valid)
        valid &= np.abs(delta) <= 0.04
        edge_cost = np.rint(100 * np.exp(-np.abs(delta[valid]) / 0.02)).astype(np.int32)
        a, b = left_ids[valid], right_ids[valid]
        from_nodes.extend((a, b))
        to_nodes.extend((b, a))
        capacities.extend((edge_cost, edge_cost))
    graph = csr_matrix(
        (np.concatenate(capacities), (np.concatenate(from_nodes), np.concatenate(to_nodes))),
        shape=(node_count + 2, node_count + 2),
        dtype=np.int32,
    )
    flow = maximum_flow(graph, source, sink)
    residual = graph - flow.flow
    residual.data = (residual.data > 0).astype(np.int32)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    source_side = np.zeros(node_count + 2, dtype=bool)
    source_side[reachable] = True
    ground = source_side[:node_count].reshape(floor.shape) & eligible
    uncertain = eligible & ~ground_seed & ~target_seed & (np.abs(ground_cost - target_cost) <= 100)
    ground[uncertain] = False
    return ground, {
        "grid_shape": [rows, columns],
        "seed_ground_cells": int(ground_seed.sum()),
        "seed_target_cells": int(target_seed.sum()),
        "mrf_ground_cells": int(ground.sum()),
        "ambiguous_cells_left_unknown": int(uncertain.sum()),
        "graph_nonzero_edges": int(graph.nnz),
        "maximum_flow_value": int(flow.flow_value),
        "capacity_scale": 1000,
        "pairwise_maximum_cost": 100,
    }


def compute_m3_mask(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normal: np.ndarray,
    offset_m: float,
    classification: dict,
) -> tuple[np.ndarray, dict]:
    base_remove = np.asarray(classification["remove"], dtype=bool)
    contact = np.asarray(classification["preserve_contact"], dtype=bool)
    height = np.asarray(classification["centroid_height_m"], dtype=np.float64)
    alignment = np.asarray(classification["normal_alignment"], dtype=np.float64)
    maximum_height = np.max(vertices[triangles] @ normal + offset_m, axis=1)
    floor, keys = make_floor(vertices, triangles, normal, classification)
    connected, connected_stats = connected_ground_cells(floor)
    local_floor = floor[keys[:, 0], keys[:, 1]]
    near_floor = (height >= local_floor - 0.02) & (height <= local_floor + 0.03)
    planar_or_low = (alignment >= 0.55) | (maximum_height <= local_floor + 0.03)
    m2_extra = connected[keys[:, 0], keys[:, 1]] & near_floor & planar_or_low & (height <= 0.15) & ~contact
    m2_remove = base_remove | m2_extra
    mrf, mrf_stats = mrf_ground_cells(floor)
    m3_extra = mrf[keys[:, 0], keys[:, 1]] & near_floor & planar_or_low & (height <= 0.15) & ~contact
    remove = m2_remove | m3_extra
    return remove, {
        "cell_size_m": CELL_M,
        "m2": connected_stats,
        "m2_additional_removed_triangles": int((m2_remove & ~base_remove).sum()),
        "m3_additional_beyond_m2_triangles": int((remove & ~m2_remove).sum()),
        "total_ground_removed_triangles": int(remove.sum()),
        **mrf_stats,
    }
