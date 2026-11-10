"""Batch raw-odometry camera queries without changing temporal support policy."""

import numpy as np
from scipy.spatial.transform import Rotation


def coarse_camera_at_many(curve, timestamps, maximum_span_ns=120_000_000):
    query = np.asarray(timestamps)
    if query.ndim != 1 or (len(query) and query.dtype.kind not in "iu"):
        raise ValueError("Query timestamps must be a one-dimensional integer sequence")
    if (not isinstance(maximum_span_ns, (int, np.integer)) or maximum_span_ns < 0
            or (len(query) and query.dtype.kind == "u" and int(query.max()) > np.iinfo(np.int64).max)):
        raise ValueError("Timestamp range or interpolation-span policy is invalid")
    if len(query) == 0:
        return []
    query = query.astype(np.int64, copy=False)
    supported = (query >= curve.stamps[0]) & (query <= curve.stamps[-1])
    right = np.searchsorted(curve.stamps, query).clip(0, len(curve.stamps)-1)
    left = np.where(curve.stamps[right] == query, right, right-1).clip(0)
    span = curve.stamps[right]-curve.stamps[left]
    valid = supported & (span <= maximum_span_ns)
    good = np.flatnonzero(valid)
    values = {}
    if len(good):
        lo, hi, width = left[good], right[good], span[good]
        alpha = np.zeros(len(good), dtype=np.float64)
        nonzero = width != 0
        alpha[nonzero] = (query[good][nonzero]-curve.stamps[lo[nonzero]])/width[nonzero]
        position = (1-alpha[:, None])*curve.poses[lo, :3]+alpha[:, None]*curve.poses[hi, :3]
        rotation = curve.slerp((query[good]-curve.origin)/1e9)
        centers = position+rotation.apply(curve.transform[:3, 3])
        camera_rotation = rotation*Rotation.from_matrix(curve.transform[:3, :3])
        matrices, quaternions = camera_rotation.as_matrix(), camera_rotation.as_quat()
        for offset, index in enumerate(good):
            values[int(index)] = {"C_W_m": centers[offset].tolist(), "R_WC": matrices[offset].tolist(),
                "quaternion_WC_xyzw": quaternions[offset].tolist(), "left_pose_index": int(lo[offset]),
                "right_pose_index": int(hi[offset]), "bracket_span_ns": int(width[offset]),
                "interpolation_alpha": float(alpha[offset]), "source_role": "recorded_raw_odometry_coarse_estimate",
                "pose_producer": "unconfirmed", "time_offset_ns": 0,
                "offline_future_bracketing": True, "extrapolated": False, "coarse_pose_valid": True}
    return [values[index] if valid[index] else {"coarse_pose_valid": False,
            "pose_rejection": "outside_coarse_time_support" if not supported[index] else "coarse_interpolation_gap_too_large"}
            for index in range(len(query))]
