"""Retain quality-passing images lacking odometry without fabricating pose priors."""

import math


def select_pose_gap_candidates(rows, policy, sharpness_min):
    candidates, decisions = [], []
    for row in rows:
        if row["coarse_pose_valid"]:
            continue
        quality = row["quality"]
        if not all(math.isfinite(value) for value in quality.values()):
            raise ValueError("Non-finite candidate quality metadata")
        passed = (not row["excluded_role"] and quality["sharpness"] >= sharpness_min
            and quality["clipped_fraction"] <= policy["clipped_fraction_max"]
            and policy["brightness_min"] <= quality["mean_brightness"] <= policy["brightness_max"]
            and quality["corner_count_half_resolution"] >= policy["corners_min"]
            and quality["occupied_grid_cells_4x4"] >= policy["occupied_cells_min"])
        decisions.append({"image_name": row["image_name"], "original_decision": row["decision"],
            "pose_rejection": row.get("pose_rejection"), "quality_passed": passed,
            "role": "visual_pose_recovery_pending" if passed else "rejected_image_quality_or_excluded",
            "position_prior_active": False, "extrapolated_position": False})
        if passed:
            if any(key in row for key in ("C_W_m", "R_WC", "T_WC")):
                raise ValueError("A pose-gap candidate must not contain an invented coarse pose")
            candidates.append(dict(row))
    return candidates, decisions
