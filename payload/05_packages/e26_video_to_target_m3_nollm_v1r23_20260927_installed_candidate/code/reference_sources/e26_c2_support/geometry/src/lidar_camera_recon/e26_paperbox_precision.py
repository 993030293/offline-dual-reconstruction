"""Pure geometry helpers for the E26 paper-box precision candidate.

The helpers deliberately avoid image, mask, and reference access.  They only
interpolate an already-produced diagnostic trajectory and aggregate points
whose source-point lineage has been frozen upstream.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return value / norm


def slerp_xyzw(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc quaternion interpolation for ``alpha`` in [0, 1]."""

    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    q0 = normalize_quaternion_xyzw(first)
    q1 = normalize_quaternion_xyzw(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    result = (
        np.sin((1.0 - alpha) * theta) / sin_theta * q0
        + np.sin(alpha * theta) / sin_theta * q1
    )
    return normalize_quaternion_xyzw(result)


def interpolate_pose(
    timestamps_ns: np.ndarray,
    poses_xyz_xyzw: np.ndarray,
    target_ns: int,
    max_bracket_span_ns: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Interpolate a pose at an exact sensor timestamp.

    Extrapolation is forbidden. Translation is linear and orientation uses
    shortest-arc SLERP.  The returned diagnostics bind both bracketing samples.
    """

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    poses = np.asarray(poses_xyz_xyzw, dtype=np.float64)
    if timestamps.ndim != 1 or poses.shape != (len(timestamps), 7):
        raise ValueError("trajectory must have timestamps[N] and poses[N,7]")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    if max_bracket_span_ns <= 0:
        raise ValueError("max bracket span must be positive")
    right = int(np.searchsorted(timestamps, int(target_ns), side="left"))
    if right < len(timestamps) and int(timestamps[right]) == int(target_ns):
        pose = poses[right].copy()
        pose[3:] = normalize_quaternion_xyzw(pose[3:])
        return pose, {
            "left_delta_ns": 0,
            "right_delta_ns": 0,
            "bracket_span_ns": 0,
            "alpha": 0.0,
        }
    if right == 0 or right == len(timestamps):
        raise ValueError("target timestamp is outside the trajectory; extrapolation forbidden")
    left = right - 1
    left_ns = int(timestamps[left])
    right_ns = int(timestamps[right])
    span_ns = right_ns - left_ns
    if span_ns > max_bracket_span_ns:
        raise ValueError(f"trajectory bracket is {span_ns / 1e6:.3f} ms")
    alpha = (int(target_ns) - left_ns) / span_ns
    translation = (1.0 - alpha) * poses[left, :3] + alpha * poses[right, :3]
    orientation = slerp_xyzw(poses[left, 3:], poses[right, 3:], float(alpha))
    return np.concatenate([translation, orientation]), {
        "left_delta_ns": int(target_ns) - left_ns,
        "right_delta_ns": right_ns - int(target_ns),
        "bracket_span_ns": span_ns,
        "alpha": float(alpha),
    }


def rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def aggregate_voxels(
    points_m: np.ndarray, frame_ordinal: np.ndarray, voxel_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centroid aggregation without dilation or gap filling."""

    points = np.asarray(points_m, dtype=np.float64)
    frames = np.asarray(frame_ordinal, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or frames.shape != (len(points),):
        raise ValueError("points must be [N,3] and frame ordinal [N]")
    if len(points) == 0 or not np.all(np.isfinite(points)) or voxel_m <= 0:
        raise ValueError("finite non-empty points and a positive voxel size are required")
    keys = np.floor(points / float(voxel_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    centroid = sums / count[:, None]
    squared = np.zeros(len(count), dtype=np.float64)
    np.add.at(squared, inverse, np.sum((points - centroid[inverse]) ** 2, axis=1))
    rms_m = np.sqrt(squared / count)
    pairs = np.column_stack([inverse.astype(np.int64), frames])
    support = np.bincount(
        np.unique(pairs, axis=0)[:, 0], minlength=len(count)
    ).astype(np.int16)
    return (
        centroid.astype(np.float32),
        count.astype(np.int32),
        support,
        rms_m.astype(np.float32),
    )


def apply_rigid_transform(
    points_m: np.ndarray, rotation: np.ndarray, translation_m: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    value_rotation = np.asarray(rotation, dtype=np.float64)
    value_translation = np.asarray(translation_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if value_rotation.shape != (3, 3) or value_translation.shape != (3,):
        raise ValueError("rotation must be [3,3] and translation [3]")
    return points @ value_rotation.T + value_translation


def kabsch_rigid_transform(
    source_m: np.ndarray, target_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the proper rigid transform mapping paired source to target."""

    source = np.asarray(source_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("paired source and target must both be [N,3]")
    if len(source) < 3 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("at least three finite paired points are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1, :] *= -1
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation must be [3,3]")
    cosine = float(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def neighborhood_frame_support(
    points_m: np.ndarray,
    frame_ordinal: np.ndarray,
    voxel_m: float,
    radius_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count distinct supporting frames in neighbouring occupied voxels.

    The function never creates an unoccupied voxel.  Neighbourhood support is
    used only to retain a measured centroid at fine resolution, so a gap cannot
    be filled by this operation.
    """

    points = np.asarray(points_m, dtype=np.float64)
    frames = np.asarray(frame_ordinal, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or frames.shape != (len(points),):
        raise ValueError("points must be [N,3] and frame ordinal [N]")
    if len(points) == 0 or voxel_m <= 0 or radius_voxels < 0:
        raise ValueError("non-empty points, positive voxel, and non-negative radius required")
    if np.any(frames < 0):
        raise ValueError("frame ordinals must be non-negative")
    keys = np.floor(points / float(voxel_m)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    observation_count = np.bincount(inverse).astype(np.int32)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    centroid = (sums / observation_count[:, None]).astype(np.float32)

    pairs = np.unique(
        np.column_stack([inverse.astype(np.int64), frames]), axis=0
    )
    exact_support = np.bincount(
        pairs[:, 0], minlength=len(unique_keys)
    ).astype(np.int16)
    block_count = int(frames.max()) // 64 + 1
    frame_bits = np.zeros((len(unique_keys), block_count), dtype=np.uint64)
    pair_blocks = pairs[:, 1] // 64
    pair_masks = np.left_shift(
        np.uint64(1), (pairs[:, 1] % 64).astype(np.uint64)
    )
    np.bitwise_or.at(frame_bits, (pairs[:, 0], pair_blocks), pair_masks)

    structured_dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    key_structured = np.ascontiguousarray(unique_keys).view(structured_dtype).reshape(-1)
    order = np.argsort(key_structured, order=("x", "y", "z"))
    sorted_keys = key_structured[order]
    neighbourhood_bits = np.zeros_like(frame_bits)
    offsets = [
        (x, y, z)
        for x in range(-radius_voxels, radius_voxels + 1)
        for y in range(-radius_voxels, radius_voxels + 1)
        for z in range(-radius_voxels, radius_voxels + 1)
        if x * x + y * y + z * z <= radius_voxels * radius_voxels
    ]
    for offset in offsets:
        target = np.ascontiguousarray(
            unique_keys + np.asarray(offset, dtype=np.int64)
        )
        target_structured = target.view(structured_dtype).reshape(-1)
        position = np.searchsorted(sorted_keys, target_structured)
        bounded = position < len(sorted_keys)
        matching = np.zeros(len(position), dtype=bool)
        matching[bounded] = sorted_keys[position[bounded]] == target_structured[bounded]
        if np.any(matching):
            neighbourhood_bits[matching] |= frame_bits[order[position[matching]]]
    byte_popcount = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1)
    neighbourhood_support = byte_popcount[
        neighbourhood_bits.view(np.uint8)
    ].sum(axis=1).astype(np.int16)
    return centroid, observation_count, exact_support, neighbourhood_support


def fit_horizontal_object_frame(
    points_map_m: np.ndarray,
    *,
    minimum_point_count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic, display-only horizontal PCA frame.

    The frame changes coordinates only. It does not move individual
    observations or create geometry, so measured gaps remain measured gaps.
    """

    points = np.asarray(points_map_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_map_m must have shape (N, 3)")
    if len(points) < minimum_point_count:
        raise ValueError("not enough points to fit the object frame")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_map_m contains non-finite values")
    origin = np.median(points, axis=0)
    centered_xy = points[:, :2] - origin[:2]
    covariance = centered_xy.T @ centered_xy / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    first_xy = vectors[:, int(np.argmax(values))]
    if first_xy[int(np.argmax(np.abs(first_xy)))] < 0:
        first_xy = -first_xy
    second_xy = np.array([-first_xy[1], first_xy[0]], dtype=np.float64)
    rotation = np.array(
        [
            [first_xy[0], first_xy[1], 0.0],
            [second_xy[0], second_xy[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError("object frame is not right handed")
    return origin, rotation


def transform_points_to_object_frame(
    points_map_m: np.ndarray,
    origin_map_m: np.ndarray,
    rotation_object_from_map: np.ndarray,
) -> np.ndarray:
    """Transform map points into a fixed object display frame."""

    points = np.asarray(points_map_m, dtype=np.float64)
    origin = np.asarray(origin_map_m, dtype=np.float64)
    rotation = np.asarray(rotation_object_from_map, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_map_m must have shape (N, 3)")
    if origin.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("invalid object-frame transform shape")
    return (rotation @ (points - origin).T).T


def build_six_box_reference_model(
    center_uv_m: np.ndarray,
    floor_z_m: float,
    *,
    gap_m: float = 0.02,
    gap_uv_m: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Build the dimension-reference model requested for the six-box stack.

    Every physical carton has edge lengths 0.4 x 0.5 x 0.6 m.  The four
    bottom cartons use the 0.4 m edge as height.  Image inspection of the
    source orbit frames shows that both upper cartons also use the 0.4 m edge
    as height.  Placement is a visual reference, not a fitted or completed
    reconstruction.
    """

    center = np.asarray(center_uv_m, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_uv_m must be a finite (2,) vector")
    if not np.isfinite(floor_z_m):
        raise ValueError("floor_z_m must be finite")
    if gap_uv_m is None:
        gaps = np.array([gap_m, gap_m], dtype=np.float64)
    else:
        gaps = np.asarray(gap_uv_m, dtype=np.float64)
    if gaps.shape != (2,) or not np.all(np.isfinite(gaps)):
        raise ValueError("gap_uv_m must be a finite (2,) vector")
    if np.any(gaps < 0.0) or np.any(gaps > 0.40):
        raise ValueError("each box gap must be between 0 and 0.40 m")
    bottom_size = np.array([0.6, 0.5, 0.4], dtype=np.float64)
    offset_u = (bottom_size[0] + gaps[0]) / 2.0
    offset_v = (bottom_size[1] + gaps[1]) / 2.0
    boxes: list[dict[str, object]] = []
    ordinal = 0
    for row, v_sign in enumerate((-1.0, 1.0)):
        for column, u_sign in enumerate((-1.0, 1.0)):
            boxes.append(
                {
                    "id": f"base_{row}_{column}",
                    "ordinal": ordinal,
                    "role": "bottom_carton_0p4m_height",
                    "center_m": [
                        float(center[0] + u_sign * offset_u),
                        float(center[1] + v_sign * offset_v),
                        float(floor_z_m + 0.2),
                    ],
                    "size_uvz_m": bottom_size.tolist(),
                }
            )
            ordinal += 1
    boxes.append(
        {
            "id": "top_middle_horizontal",
            "ordinal": 4,
            "role": "image_inferred_0p4m_height",
            "center_m": [float(center[0]), float(center[1]), float(floor_z_m + 0.6)],
            "size_uvz_m": [0.6, 0.5, 0.4],
        }
    )
    boxes.append(
        {
            "id": "top_upper_horizontal",
            "ordinal": 5,
            "role": "image_inferred_0p4m_height_upper",
            "center_m": [float(center[0]), float(center[1]), float(floor_z_m + 1.0)],
            "size_uvz_m": [0.6, 0.5, 0.4],
        }
    )
    return boxes


def exposed_axis_aligned_box_faces(
    boxes: list[dict[str, object]],
    *,
    include_bottom: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand axis-aligned cartons into finite rectangular face primitives.

    The return arrays contain face centres, outward normals, half extents and
    source-box ordinals.  Bottom faces are excluded by default because the
    cartons rest on the floor or on another carton and are not directly
    observed.  No hidden surface points are generated by this helper.
    """

    centres: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    half_sizes: list[np.ndarray] = []
    ordinals: list[int] = []
    for ordinal, box in enumerate(boxes):
        center = np.asarray(box["center_m"], dtype=np.float64)
        size = np.asarray(box["size_uvz_m"], dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
            raise ValueError("each box needs finite center_m and positive size_uvz_m")
        half = size / 2.0
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if axis == 2 and sign < 0 and not include_bottom:
                    continue
                face_center = center.copy()
                face_center[axis] += sign * half[axis]
                normal = np.zeros(3, dtype=np.float64)
                normal[axis] = sign
                centres.append(face_center)
                normals.append(normal)
                half_sizes.append(half.copy())
                ordinals.append(int(box.get("ordinal", ordinal)))
    return (
        np.stack(centres),
        np.stack(normals),
        np.stack(half_sizes),
        np.asarray(ordinals, dtype=np.int16),
    )


def associate_axis_aligned_box_faces(
    points_m: np.ndarray,
    face_centres_m: np.ndarray,
    face_normals: np.ndarray,
    face_half_sizes_m: np.ndarray,
    *,
    tangential_margin_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Associate measurements to the nearest finite carton face.

    A point is eligible for a face only when its two tangential coordinates
    lie inside that finite rectangle plus an optional association margin.
    Projection changes only the face-normal coordinate; tangential coordinates
    are never clamped, so the routine cannot fill inter-carton gaps.
    """

    points = np.asarray(points_m, dtype=np.float64)
    centres = np.asarray(face_centres_m, dtype=np.float64)
    normals = np.asarray(face_normals, dtype=np.float64)
    half_sizes = np.asarray(face_half_sizes_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (N, 3)")
    if centres.ndim != 2 or centres.shape[1] != 3 or normals.shape != centres.shape or half_sizes.shape != centres.shape:
        raise ValueError("face arrays must all have shape (F, 3)")
    if tangential_margin_m < 0:
        raise ValueError("tangential_margin_m must be non-negative")
    best_distance = np.full(len(points), np.inf, dtype=np.float64)
    best_face = np.full(len(points), -1, dtype=np.int32)
    best_signed = np.full(len(points), np.nan, dtype=np.float64)
    best_projection = np.full_like(points, np.nan, dtype=np.float64)
    for face_index, (center, normal, half) in enumerate(zip(centres, normals, half_sizes)):
        axis = int(np.argmax(np.abs(normal)))
        tangential = [value for value in range(3) if value != axis]
        inside = np.ones(len(points), dtype=bool)
        for tangent in tangential:
            inside &= np.abs(points[:, tangent] - center[tangent]) <= half[tangent] + tangential_margin_m
        signed = (points - center) @ normal
        distance = np.abs(signed)
        update = inside & (distance < best_distance)
        if not np.any(update):
            continue
        projection = points[update] - signed[update, None] * normal[None, :]
        best_distance[update] = distance[update]
        best_signed[update] = signed[update]
        best_face[update] = face_index
        best_projection[update] = projection
    return best_face, best_distance, best_signed, best_projection


def small_angle_rotation(rotation_vector_rad: np.ndarray) -> np.ndarray:
    """Convert a compact rotation vector to a proper 3-D rotation matrix."""

    vector = np.asarray(rotation_vector_rad, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must be finite with shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-14:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
