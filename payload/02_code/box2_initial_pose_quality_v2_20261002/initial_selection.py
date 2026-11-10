"""Select initial SfM images using only raw quality and recorded coarse poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
TOOLS = PACKAGE / "code/reference_sources/e26_c2_support/frontend/tools"
RAW = Path("E:/lidar_reconstruction_video_data/raw_recordings/e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820B_Box2/recon_collection_20260820_170743_689578_0.mcap")
CAMERA_TOPIC = "/sensing/camera/image_raw"
POLICY = {"window_seconds": .20, "translation_min_m": .035, "rotation_min_degrees": 1.2,
          "bridge_gap_seconds": .8, "position_cell_m": .10, "direction_cell_degrees": 10.,
          "sharpness_quantile": .05, "clipped_fraction_max": .15,
          "brightness_min": 20., "brightness_max": 235., "corners_min": 80,
          "occupied_cells_min": 4, "maximum_interpolation_span_ns": 120_000_000,
          "maximum_exported_frames": 1200, "jpeg_quality": 97, "maximum_wall_seconds": 1800}


def write_json(path, value, *, update=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if update:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def data_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CoarseCameraCurve:
    """Interpolate a recorded body trajectory without extrapolation or SfM."""

    def __init__(self, stamps, poses, transform):
        self.stamps = np.asarray(stamps, dtype=np.int64)
        self.poses = np.asarray(poses, dtype=float)
        if len(self.stamps) < 2 or np.any(np.diff(self.stamps) <= 0):
            raise ValueError("Coarse timestamps must be strictly increasing")
        if self.poses.shape != (len(self.stamps), 7) or not np.isfinite(self.poses).all():
            raise ValueError("Invalid coarse pose array")
        if not np.allclose(np.linalg.norm(self.poses[:, 3:], axis=1), 1, atol=1e-6):
            raise ValueError("Invalid body quaternion norm")
        self.origin = int(self.stamps[0])
        self.slerp = Slerp((self.stamps - self.origin) / 1e9, Rotation.from_quat(self.poses[:, 3:]))
        self.transform = np.asarray(transform, dtype=float)
        if self.transform.shape != (4, 4) or not np.isfinite(self.transform).all():
            raise ValueError("Invalid camera extrinsic")

    def at(self, stamp, maximum_span_ns=120_000_000):
        stamp = int(stamp)
        if stamp < self.stamps[0] or stamp > self.stamps[-1]:
            raise ValueError("outside_coarse_time_support")
        right = int(np.searchsorted(self.stamps, stamp))
        left = right if self.stamps[right] == stamp else right - 1
        span = int(self.stamps[right] - self.stamps[left])
        if span > maximum_span_ns:
            raise ValueError("coarse_interpolation_gap_too_large")
        alpha = 0. if span == 0 else (stamp - int(self.stamps[left])) / span
        position = (1 - alpha) * self.poses[left, :3] + alpha * self.poses[right, :3]
        rotation = self.slerp((stamp - self.origin) / 1e9)
        center = position + rotation.apply(self.transform[:3, 3])
        camera_rotation = rotation * Rotation.from_matrix(self.transform[:3, :3])
        return {"C_W_m": center.tolist(), "R_WC": camera_rotation.as_matrix().tolist(),
                "quaternion_WC_xyzw": camera_rotation.as_quat().tolist(),
                "left_pose_index": left, "right_pose_index": right,
                "bracket_span_ns": span, "interpolation_alpha": float(alpha),
                "source_role": "recorded_raw_odometry_coarse_estimate",
                "pose_producer": "unconfirmed", "time_offset_ns": 0,
                "offline_future_bracketing": True, "extrapolated": False}


def static_transform(transforms, source="camera_optical", target="base_link"):
    graph = defaultdict(list)
    known = {}
    for item in transforms:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(item["rotation_xyzw"]).as_matrix()
        matrix[:3, 3] = item["translation"]
        edge = (item["parent"], item["child"])
        if edge in known:
            if not np.allclose(known[edge], matrix, atol=1e-10):
                raise ValueError("The static extrinsic changes inside the recording")
            continue
        known[edge] = matrix
        graph[item["child"]].append((item["parent"], matrix))
        graph[item["parent"]].append((item["child"], np.linalg.inv(matrix)))
    queue = [(source, np.eye(4), [source])]
    visited = {source}
    for frame, accumulated, path in queue:
        if frame == target:
            return accumulated, path
        for neighbor, edge in graph[frame]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, edge @ accumulated, path + [neighbor]))
    raise ValueError("Raw tf_static does not connect the optical camera to the body")


def quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    corners = cv2.goodFeaturesToTrack(reduced, maxCorners=800, qualityLevel=.01, minDistance=5, blockSize=5)
    count, occupied = 0, 0
    if corners is not None:
        count = len(corners)
        cells = np.minimum(3, np.floor(corners[:, 0, :] / [reduced.shape[1] / 4, reduced.shape[0] / 4]).astype(int))
        occupied = len(set(map(tuple, cells)))
    score = math.log1p(sharpness) + .25 * math.log1p(count) + .4 * occupied / 16 - 4 * clipped
    return {"sharpness": sharpness, "clipped_fraction": clipped, "mean_brightness": float(gray.mean()),
            "corner_count_half_resolution": count, "occupied_grid_cells_4x4": occupied,
            "quality_score": score}


def differences(a, b):
    distance = float(np.linalg.norm(np.asarray(a["C_W_m"]) - b["C_W_m"]))
    qa = Rotation.from_quat(a["quaternion_WC_xyzw"])
    qb = Rotation.from_quat(b["quaternion_WC_xyzw"])
    angle = float(np.rad2deg((qa.inv() * qb).magnitude()))
    return distance, angle


def coverage_key(row, policy):
    position = np.floor(np.asarray(row["C_W_m"]) / policy["position_cell_m"]).astype(int)
    direction = np.asarray(row["R_WC"])[:, 2]
    yaw = np.degrees(np.arctan2(direction[1], direction[0])) % 360
    pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))
    step = policy["direction_cell_degrees"]
    return tuple(position) + (int(yaw // step), int((pitch + 90) // step))


def choose(rows, policy, sharpness_min):
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
            row["decision"] = "eligible_not_selected" if good else "rejected_image_quality"
            if good:
                eligible.append(row)
    if not eligible:
        raise ValueError("No frame passes the declared quality and coarse-pose gates")
    selected, ids = [], set()
    origin = min(row["camera_header_ns"] for row in rows)
    bins = defaultdict(list)
    for row in eligible:
        bins[int((row["camera_header_ns"] - origin) // round(policy["window_seconds"] * 1e9))].append(row)

    def add(row, reason):
        if len(selected) >= policy["maximum_exported_frames"]:
            return False
        if row["camera_frame"] not in ids:
            row["decision"] = "selected_initial_sfm_pending"
            row["selection_reason"] = reason
            selected.append(row)
            ids.add(row["camera_frame"])
        return True

    for key in sorted(bins):
        for row in sorted(bins[key], key=lambda item: (-item["quality"]["quality_score"], item["camera_frame"])):
            distance, angle = (float("inf"), float("inf")) if not selected else differences(row, selected[-1])
            elapsed = float("inf") if not selected else (row["camera_header_ns"] - selected[-1]["camera_header_ns"]) / 1e9
            if distance >= policy["translation_min_m"] or angle >= policy["rotation_min_degrees"] or elapsed >= policy["bridge_gap_seconds"]:
                add(row, "temporal_quality_with_coarse_motion")
                break
    covered = {coverage_key(row, policy) for row in selected}
    for row in sorted(eligible, key=lambda item: (-item["quality"]["quality_score"], item["camera_frame"])):
        key = coverage_key(row, policy)
        if key in covered or row["camera_frame"] in ids:
            continue
        pairs = [differences(row, existing) for existing in selected]
        novel = all(d >= policy["translation_min_m"] or a >= policy["rotation_min_degrees"] for d, a in pairs)
        if novel and add(row, "new_coarse_position_and_direction_cell"):
            covered.add(key)
    # Fill only existing quality-supported gaps; failed gates are never relaxed.
    while len(selected) < policy["maximum_exported_frames"]:
        chronological = sorted(selected, key=lambda item: item["camera_header_ns"])
        inserted = False
        for a, b in zip(chronological, chronological[1:]):
            gap = b["camera_header_ns"] - a["camera_header_ns"]
            if gap <= round(policy["bridge_gap_seconds"] * 1e9):
                continue
            middle = (a["camera_header_ns"] + b["camera_header_ns"]) / 2
            candidates = [row for row in eligible if row["camera_frame"] not in ids
                          and abs(row["camera_header_ns"] - middle) < .30 * gap]
            if candidates:
                best = max(candidates, key=lambda item: (item["quality"]["quality_score"], -abs(item["camera_header_ns"] - middle)))
                add(best, "quality_supported_temporal_bridge")
                inserted = True
        if not inserted:
            break
    selected.sort(key=lambda item: item["camera_header_ns"])
    return selected, eligible


def run(output, exclusion_source):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    cv2.setNumThreads(4)
    sys.path.insert(0, str(TOOLS))
    from mcap_audit import McapIndex, Cdr, decode_camera_info, decode_tf_static
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    exclusions = {}
    # Read only evaluation IDs and their role, never old training poses or quality.
    split = json.loads(exclusion_source.read_text(encoding="utf-8"))
    for item in split["extra_frames"]:
        exclusions[int(item["camera_frame"])] = item["role"]
    del split
    write_json(output / "EXCLUDED_EVALUATION_IDS.json", {"source_role": "evaluation_exclusion_only",
        "source_path": str(exclusion_source), "frames": exclusions,
        "old_train_ids_used": False, "strict_blind_holdout": False,
        "boundary": "Previously inspected development recording and holdout; no new blind-test claim"})
    write_json(output / "SELECTION_PROTOCOL.json", {"status": "frozen_before_full_scan", "policy": POLICY,
        "source_roles": ["raw_measurement", "recorded_coarse_odometry", "raw_sensor_calibration", "evaluation_exclusion_only"],
        "forbidden_inputs": ["old_sfm_poses", "old_sfm_point_cloud", "old_sfm_ROI", "GT_trajectory", "GT_geometry", "reference_colmap"],
        "target_frame_count": None, "quality_threshold_source": "nonexcluded_coarse_valid_raw_image_sharpness_P05",
        "coarse_pose_accuracy_confirmed": False, "offline_future_bracketing": True})
    index = McapIndex(RAW)
    stamps, poses, covariances, frames = [], [], [], set()
    for message in index.iter_messages("/localization/pose/odom"):
        cdr = Cdr(message["payload"])
        sec, ns, parent = cdr.header()
        child = cdr.string()
        stamps.append(sec * 1_000_000_000 + ns)
        poses.append([cdr.f64() for _ in range(7)])
        covariances.append([cdr.f64() for _ in range(36)])
        frames.add((parent, child))
    if frames != {("odom", "base_link")}:
        raise ValueError("The raw body/world frame contract is not supported")
    transforms = []
    for message in index.iter_messages("/tf_static"):
        transforms.extend(decode_tf_static(message["payload"]))
    extrinsic, chain = static_transform(transforms)
    camera_info = decode_camera_info(next(index.iter_messages("/sensing/camera/camera_info"))["payload"])
    curve = CoarseCameraCurve(stamps, poses, extrinsic)
    write_json(output / "RAW_SENSOR_CONTRACT.json", {"raw_file": str(RAW), "raw_bytes": RAW.stat().st_size,
        "raw_registered_sha256": "a48f1c62300d507fd9b5c1ab7af0218ae3d984cf8cae23b312350c76cebd2acf",
        "raw_full_hash_recomputed": False, "camera_info_source": "raw_mcap_camera_info",
        "camera_info": camera_info, "T_body_camera": extrinsic.tolist(), "extrinsic_chain": chain,
        "odometry_count": len(stamps), "odometry_frames": sorted(frames),
        "odometry_producer": "unconfirmed", "odometry_covariance_all_zero": bool(np.all(np.asarray(covariances) == 0)),
        "odometry_role": "recorded_estimate_not_GT", "color_contract": COLOR_CONTRACT_VERSION,
        "max_raw_pose_gap_ns": int(np.diff(stamps).max()), "sfm_or_gt_used_for_selection": False})
    np.savez(output / "raw_body_odometry.npz", timestamp_ns=np.asarray(stamps, dtype=np.int64), pose_xyz_xyzw=np.asarray(poses))
    count = index.message_count(CAMERA_TOPIC)
    rows = []
    for frame, message in index.iter_selected_messages(CAMERA_TOPIC, range(count)):
        if time.monotonic() - start > POLICY["maximum_wall_seconds"]:
            raise TimeoutError("Initial selection exceeded the declared scan budget")
        stamp, optical_frame, raw = decode_bayer(message["payload"])
        image = demosaic_rggb8_to_bgr(raw)
        row = {"camera_frame": frame, "camera_header_ns": int(stamp), "optical_frame": optical_frame,
               "image_name": f"frame_{frame:05d}_{int(stamp)}.jpg", "width": image.shape[1], "height": image.shape[0],
               "quality": quality(image), "excluded_role": exclusions.get(frame), "coarse_pose_valid": False}
        if optical_frame != "camera_optical" or [image.shape[1], image.shape[0]] != [camera_info["width"], camera_info["height"]]:
            raise ValueError("The image frame or resolution differs from raw calibration")
        try:
            row.update(curve.at(stamp, POLICY["maximum_interpolation_span_ns"]), coarse_pose_valid=True)
        except ValueError as error:
            row["pose_rejection"] = str(error)
        rows.append(row)
        if (frame + 1) % 250 == 0 or frame + 1 == count:
            write_json(output / "STATE.json", {"status": "scanning", "pid": os.getpid(), "scanned": frame + 1,
                "total": count, "elapsed_seconds": time.monotonic() - start, "timestamp_unix": time.time()}, update=True)
            print(f"Scanned {frame + 1}/{count} raw frames", flush=True)
    reference = [row["quality"]["sharpness"] for row in rows if row["coarse_pose_valid"] and not row["excluded_role"]]
    threshold = float(np.quantile(reference, POLICY["sharpness_quantile"]))
    selected, eligible = choose(rows, POLICY, threshold)
    write_json(output / "STATE.json", {"status": "exporting", "pid": os.getpid(), "scanned": count,
        "selected": len(selected), "elapsed_seconds": time.monotonic() - start, "timestamp_unix": time.time()}, update=True)
    images = output / "images"
    images.mkdir()
    by_frame = {row["camera_frame"]: row for row in selected}
    for frame, message in index.iter_selected_messages(CAMERA_TOPIC, by_frame):
        stamp, _, raw = decode_bayer(message["payload"])
        row = by_frame[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw frame identity changed between scan and export")
        image = demosaic_rggb8_to_bgr(raw)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, POLICY["jpeg_quality"]])
        if not ok:
            raise RuntimeError("Frame encoding failed")
        path = images / row["image_name"]
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())
        row["image_sha256"] = data_digest(path)
    with (output / "frame_decisions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    pairs = set()
    positions = np.asarray([row["C_W_m"] for row in selected])
    for i, row in enumerate(selected):
        for j in range(i + 1, min(len(selected), i + 9)):
            pairs.add((i, j))
        nearest = np.argsort(np.linalg.norm(positions - positions[i], axis=1))[:17]
        for j in nearest:
            if abs(int(j) - i) > 8:
                pairs.add(tuple(sorted((i, int(j)))))
    with (output / "initial_matching_pairs.txt").open("x", encoding="utf-8") as stream:
        for i, j in sorted(pairs):
            stream.write(selected[i]["image_name"] + " " + selected[j]["image_name"] + "\n")
    times = np.asarray([row["camera_header_ns"] for row in selected], dtype=np.int64)
    gaps = np.diff(times) / 1e9
    unresolved = [{"left": selected[i]["image_name"], "right": selected[i + 1]["image_name"], "gap_seconds": float(gap)}
                  for i, gap in enumerate(gaps) if gap > POLICY["bridge_gap_seconds"]]
    receipt = {"status": "initial_frames_exported_sfm_not_run", "raw_frame_count": count,
        "quality_and_coarse_pose_eligible_count": len(eligible), "selected_count": len(selected),
        "selection_reason_counts": dict(Counter(row["selection_reason"] for row in selected)),
        "decision_counts": dict(Counter(row["decision"] for row in rows)), "sharpness_min": threshold,
        "selected_max_gap_seconds": float(gaps.max()) if len(gaps) else None,
        "selected_gap_P50_P95_seconds": np.quantile(gaps, [.5, .95]).tolist() if len(gaps) else [],
        "unresolved_quality_supported_time_gaps": unresolved, "resource_cap_reached": len(selected) == POLICY["maximum_exported_frames"],
        "selected_time_range_ns": [int(times[0]), int(times[-1])], "matching_candidate_pair_count": len(pairs),
        "matching_pairs_verified_by_features": False, "excluded_overlap": len(set(by_frame) & set(exclusions)),
        "exported_resolution": [camera_info["width"], camera_info["height"]],
        "old_sfm_inputs_used": False, "GT_used": False, "SfM_run": False, "Gaussian_retrained": False,
        "elapsed_seconds": time.monotonic() - start, "claim_boundary": "quality_and_motion_development_initialization_not_precision_validation"}
    write_json(output / "INITIAL_FRAME_MANIFEST.json", {"receipt": receipt, "selected_frames": selected,
        "coarse_pose_convention": "T_world_camera; R_WC optical axes; C_W in odom metric coordinates",
        "camera_intrinsics": camera_info, "policy": POLICY})
    write_json(output / "SELECTION_RECEIPT.json", receipt)
    write_json(output / "STATE.json", {"status": "completed_initial_selection", "pid": os.getpid(),
        "selected": len(selected), "timestamp_unix": time.time()}, update=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-source", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.exclude_source.resolve(strict=True))
