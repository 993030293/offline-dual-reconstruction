"""Preserve selection policy while batching repeated geometric coverage checks."""

from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation


def choose_batched(rows, policy, sharpness_min, baseline, additional_quality_gate=None):
    eligible = []
    for row in rows:
        if row.get("excluded_role"):
            row["decision"] = "excluded_development_evaluation"
        elif not row.get("coarse_pose_valid"):
            row["decision"] = "rejected_coarse_time_support"
        else:
            q = row["quality"]
            good = (q["sharpness"] >= sharpness_min and q["clipped_fraction"] <= policy["clipped_fraction_max"]
                and policy["brightness_min"] <= q["mean_brightness"] <= policy["brightness_max"]
                and q["corner_count_half_resolution"] >= policy["corners_min"]
                and q["occupied_grid_cells_4x4"] >= policy["occupied_cells_min"])
            extra_good = additional_quality_gate is None or additional_quality_gate(row)
            row["decision"] = ("eligible_not_selected" if good and extra_good else
                "rejected_initial_visual_trackability" if good and not extra_good else "rejected_image_quality")
            good = good and extra_good
            if good:
                eligible.append(row)
    if not eligible:
        raise ValueError("No frame passes the declared quality and coarse-pose gates")
    selected, selected_ids = [], set()
    indices = {row["camera_frame"]: i for i, row in enumerate(eligible)}
    positions = np.asarray([row["C_W_m"] for row in eligible])
    rotations = Rotation.from_quat([row["quaternion_WC_xyzw"] for row in eligible])
    chosen_indices = []
    origin = min(row["camera_header_ns"] for row in rows)
    bins = defaultdict(list)
    for row in eligible:
        bins[int((row["camera_header_ns"]-origin)//round(policy["window_seconds"]*1e9))].append(row)

    def add(row, reason):
        if len(selected) >= policy["maximum_exported_frames"]:
            return False
        if row["camera_frame"] not in selected_ids:
            row["decision"] = "selected_initial_sfm_pending"
            row["selection_reason"] = reason
            selected.append(row)
            selected_ids.add(row["camera_frame"])
            chosen_indices.append(indices[row["camera_frame"]])
        return True

    for key in sorted(bins):
        for row in sorted(bins[key], key=lambda item: (-item["quality"]["quality_score"], item["camera_frame"])):
            distance, angle = (float("inf"), float("inf")) if not selected else baseline.differences(row, selected[-1])
            elapsed = float("inf") if not selected else (row["camera_header_ns"]-selected[-1]["camera_header_ns"])/1e9
            if distance >= policy["translation_min_m"] or angle >= policy["rotation_min_degrees"] or elapsed >= policy["bridge_gap_seconds"]:
                add(row, "temporal_quality_with_coarse_motion")
                break
    covered = {baseline.coverage_key(row, policy) for row in selected}
    for row in sorted(eligible, key=lambda item: (-item["quality"]["quality_score"], item["camera_frame"])):
        key = baseline.coverage_key(row, policy)
        if key in covered or row["camera_frame"] in selected_ids:
            continue
        index = indices[row["camera_frame"]]
        chosen = np.asarray(chosen_indices, dtype=int)
        distances = np.linalg.norm(positions[index]-positions[chosen], axis=1)
        # SciPy performs the same normalized relative quaternion operation in one batch.
        angles = np.rad2deg((rotations[index].inv()*rotations[chosen]).magnitude())
        novel = bool(np.all((distances >= policy["translation_min_m"]) |
                            (angles >= policy["rotation_min_degrees"])))
        if novel and add(row, "new_coarse_position_and_direction_cell"):
            covered.add(key)
    while len(selected) < policy["maximum_exported_frames"]:
        chronological = sorted(selected, key=lambda item: item["camera_header_ns"])
        inserted = False
        for a, b in zip(chronological, chronological[1:]):
            gap = b["camera_header_ns"]-a["camera_header_ns"]
            if gap <= round(policy["bridge_gap_seconds"]*1e9):
                continue
            middle = (a["camera_header_ns"]+b["camera_header_ns"])/2
            candidates = [row for row in eligible if row["camera_frame"] not in selected_ids
                          and abs(row["camera_header_ns"]-middle) < .30*gap]
            if candidates:
                best = max(candidates, key=lambda item: (item["quality"]["quality_score"],
                           -abs(item["camera_header_ns"]-middle)))
                add(best, "quality_supported_temporal_bridge")
                inserted = True
        if not inserted:
            break
    selected.sort(key=lambda item: item["camera_header_ns"])
    return selected, eligible
