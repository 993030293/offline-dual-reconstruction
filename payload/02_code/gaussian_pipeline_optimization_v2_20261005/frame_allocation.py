"""Build nested frame budgets using only raw quality and recorded coarse poses."""

from collections import defaultdict

import numpy as np


def eligible_rows(rows, policy, sharpness_min):
    """Recompute immutable quality gates rather than trusting a selection label."""
    result = []
    names, identifiers = set(), set()
    for row in rows:
        name, identifier = row["image_name"], row["camera_frame"]
        if name in names or identifier in identifiers:
            raise ValueError("Duplicate raw frame identity")
        names.add(name)
        identifiers.add(identifier)
        if row.get("excluded_role") or not row.get("coarse_pose_valid"):
            continue
        q = row["quality"]
        if not all(np.isfinite(value) for value in q.values()):
            raise ValueError("Nonfinite quality input")
        good = (q["sharpness"] >= sharpness_min
                and q["clipped_fraction"] <= policy["clipped_fraction_max"]
                and policy["brightness_min"] <= q["mean_brightness"] <= policy["brightness_max"]
                and q["corner_count_half_resolution"] >= policy["corners_min"]
                and q["occupied_grid_cells_4x4"] >= policy["occupied_cells_min"])
        expected = row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
        if good != expected:
            raise ValueError("Recomputed quality qualification disagrees with source")
        if good:
            if row.get("extrapolated") or row.get("source_role") != "recorded_raw_odometry_coarse_estimate":
                raise ValueError("Unsupported pose provenance")
            if row["bracket_span_ns"] > policy["maximum_interpolation_span_ns"]:
                raise ValueError("Coarse pose interpolation exceeds the frozen bound")
            position = np.asarray(row["C_W_m"], dtype=float)
            quat = np.asarray(row["quaternion_WC_xyzw"], dtype=float)
            if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat).all():
                raise ValueError("Invalid coarse pose")
            if abs(np.linalg.norm(quat) - 1) > 1e-6:
                raise ValueError("Coarse rotation quaternion is not unit length")
            result.append(row)
    return sorted(result, key=lambda row: (row["camera_header_ns"], row["camera_frame"]))


def arrays(rows):
    positions = np.asarray([row["C_W_m"] for row in rows], dtype=np.float64)
    quaternions = np.asarray([row["quaternion_WC_xyzw"] for row in rows], dtype=np.float64)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    quality = np.asarray([row["quality"]["quality_score"] for row in rows])
    return positions, quaternions, quality


def pose_distance(positions, quaternions, position, quaternion):
    distance = np.linalg.norm(positions - position, axis=1)
    dot = np.abs(np.einsum("ij,j->i", quaternions, quaternion))
    angle = np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))
    return np.maximum(distance / .10, angle / 10.), distance, angle


def temporal_backbone(rows, origin_ns, bin_ns=400_000_000, gap_ns=800_000_000):
    """Keep quality representatives and both endpoints of unsupported gaps."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["camera_header_ns"] - origin_ns) // bin_ns].append(index)
    chosen = {0, len(rows) - 1}
    for indices in groups.values():
        chosen.add(min(indices, key=lambda i: (-rows[i]["quality"]["quality_score"], rows[i]["camera_frame"])))
    for i in range(1, len(rows)):
        if rows[i]["camera_header_ns"] - rows[i-1]["camera_header_ns"] > gap_ns:
            chosen.update((i-1, i))
    # Empty bins can leave a gap slightly larger than two bin widths.
    while True:
        inserted = False
        ordered = sorted(chosen)
        for left, right in zip(ordered, ordered[1:]):
            if rows[right]["camera_header_ns"] - rows[left]["camera_header_ns"] <= gap_ns:
                continue
            candidates = [i for i in range(left+1, right) if i not in chosen]
            if candidates:
                midpoint = (rows[right]["camera_header_ns"] + rows[left]["camera_header_ns"]) / 2
                chosen.add(min(candidates, key=lambda i: (abs(rows[i]["camera_header_ns"]-midpoint),
                                                         -rows[i]["quality"]["quality_score"], rows[i]["camera_frame"])))
                inserted = True
        if not inserted:
            break
    return sorted(chosen)


def rank_rows(rows, maximum, mandatory):
    """Quality-weighted farthest coarse-pose order with deterministic ties."""
    if maximum < len(mandatory) or maximum > len(rows) or len(set(mandatory)) != len(mandatory):
        raise ValueError("Budget cannot contain the required backbone")
    positions, quaternions, quality = arrays(rows)
    quality_order = np.argsort(quality, kind="stable")
    quality_rank = np.empty(len(rows), dtype=float)
    quality_rank[quality_order] = np.arange(len(rows))
    weight = .5 + .5 * quality_rank / max(1, len(rows)-1)
    nearest = np.full(len(rows), np.inf)
    selected = np.zeros(len(rows), dtype=bool)
    order = []
    for index in mandatory:
        if index < 0 or index >= len(rows):
            raise ValueError("Mandatory frame lies outside the eligible pool")
        order.append(int(index))
        selected[index] = True
        distance, _, _ = pose_distance(positions, quaternions, positions[index], quaternions[index])
        nearest = np.minimum(nearest, distance)
    while len(order) < maximum:
        scores = np.where(selected, -np.inf, nearest * weight)
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise ValueError("A temporal seed is required before allocation")
        selected[index] = True
        order.append(index)
        distance, _, _ = pose_distance(positions, quaternions, positions[index], quaternions[index])
        nearest = np.minimum(nearest, distance)
    return order


def coverage_metrics(eligible, selected):
    """Report pose coverage proxies; no visibility or target quality is inferred."""
    positions, quaternions, _ = arrays(eligible)
    nearest = np.full(len(eligible), np.inf)
    nearest_position = np.zeros(len(eligible))
    nearest_angle = np.zeros(len(eligible))
    for row in selected:
        distance, position, angle = pose_distance(positions, quaternions, np.asarray(row["C_W_m"]),
                                                  np.asarray(row["quaternion_WC_xyzw"]))
        update = distance < nearest
        nearest[update], nearest_position[update], nearest_angle[update] = distance[update], position[update], angle[update]
    chronological = sorted(selected, key=lambda row: row["camera_header_ns"])
    gaps = []
    for a, b in zip(chronological, chronological[1:]):
        delta = (b["camera_header_ns"] - a["camera_header_ns"]) / 1e9
        if delta > .8:
            between = sum(a["camera_header_ns"] < row["camera_header_ns"] < b["camera_header_ns"] for row in eligible)
            gaps.append({"left_frame": a["camera_frame"], "right_frame": b["camera_frame"],
                         "seconds": delta, "eligible_frames_between": between})
    scores = [row["quality"]["quality_score"] for row in selected]
    quantiles = [.5, .9, .95, .99, 1.]
    return {"quantile_levels": quantiles,
            "nearest_joint_pose_position_m": np.quantile(nearest_position, quantiles).tolist(),
            "nearest_joint_pose_rotation_degrees": np.quantile(nearest_angle, quantiles).tolist(),
            "normalized_joint_pose_error": np.quantile(nearest, quantiles).tolist(),
            "quality_score_p10_median_p90": np.quantile(scores, [.1, .5, .9]).tolist(),
            "gaps_over_0_8_seconds": gaps, "target_surface_visibility": "not_measured"}


def graph_summary(names, edges):
    adjacency = {name: set() for name in names}
    count = 0
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
            count += 1
    remaining, sizes = set(adjacency), []
    while remaining:
        reached = {min(remaining)}
        queue = list(reached)
        for a in queue:
            for b in adjacency[a] - reached:
                reached.add(b)
                queue.append(b)
        remaining -= reached
        sizes.append(len(reached))
    return {"candidate_edge_count": count, "candidate_component_sizes": sorted(sizes, reverse=True),
            "minimum_candidate_degree": min(map(len, adjacency.values())), "feature_verified": False}
