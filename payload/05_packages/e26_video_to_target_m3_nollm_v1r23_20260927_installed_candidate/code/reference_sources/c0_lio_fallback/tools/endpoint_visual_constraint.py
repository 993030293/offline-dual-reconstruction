#!/usr/bin/env python
"""Create a rotation-only long-baseline visual constraint from repeated views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_motion import decode_bayer
from lio_baseline import build_orientation_slerp, load_imu
from mcap_audit import McapIndex, decode_camera_info, find_static_transform


TOPIC = "/sensing/camera/image_raw"


def lidar_from_camera_rotation(index: McapIndex) -> Rotation:
    transform = find_static_transform(index, "base_link", "camera_optical")
    return Rotation.from_quat(transform["rotation_xyzw"])


def extract_nearest(
    index: McapIndex, target_s: float, rate_hz: float = 50.0
) -> tuple[int, np.ndarray]:
    number = min(
        max(int(round(target_s * rate_hz)), 0),
        index.message_count(TOPIC) - 1,
    )
    _, message = next(index.iter_selected_messages(TOPIC, [number]))
    timestamp_ns, _, raw = decode_bayer(message["payload"])
    gray = cv2.cvtColor(raw, cv2.COLOR_BayerRG2GRAY)
    return timestamp_ns, gray


def analyze(
    index: McapIndex,
    first_s: float,
    second_s: float,
    ratio_threshold: float,
) -> tuple[dict, np.ndarray]:
    first_ns, first = extract_nearest(index, first_s)
    second_ns, second = extract_nearest(index, second_s)
    sift = cv2.SIFT_create(nfeatures=5000)
    first_keypoints, first_descriptors = sift.detectAndCompute(first, None)
    second_keypoints, second_descriptors = sift.detectAndCompute(second, None)
    pairs = cv2.BFMatcher().knnMatch(
        first_descriptors, second_descriptors, k=2
    )
    matches = [
        best
        for best, next_best in pairs
        if best.distance < ratio_threshold * next_best.distance
    ]
    first_points = np.float32(
        [first_keypoints[match.queryIdx].pt for match in matches]
    )
    second_points = np.float32(
        [second_keypoints[match.trainIdx].pt for match in matches]
    )
    homography, homography_mask = cv2.findHomography(
        first_points, second_points, cv2.RANSAC, 3.0
    )

    info_message = next(index.iter_messages("/sensing/camera/camera_info"))
    camera_info = decode_camera_info(info_message["payload"])
    matrix = np.asarray(camera_info["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info["distortion"], dtype=np.float64)
    first_normalized = cv2.undistortPoints(
        first_points.reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    second_normalized = cv2.undistortPoints(
        second_points.reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    essential, essential_mask = cv2.findEssentialMat(
        first_normalized,
        second_normalized,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.002,
    )
    essential = essential[:3, :3]
    essential_ransac_inliers = int(np.count_nonzero(essential_mask))
    cheirality_inliers, rotation_matrix, translation, pose_mask = cv2.recoverPose(
        essential,
        first_normalized,
        second_normalized,
        np.eye(3),
        mask=essential_mask.copy(),
    )
    visual_rotation = Rotation.from_matrix(rotation_matrix)

    imu_ns, gyro, _ = load_imu(index)
    imu_slerp, gyro_bias = build_orientation_slerp(imu_ns, gyro, 11.5)
    first_lidar = imu_slerp([first_ns / 1e9])[0]
    second_lidar = imu_slerp([second_ns / 1e9])[0]
    lidar_from_camera = lidar_from_camera_rotation(index)
    predicted_camera2_from_camera1 = (
        lidar_from_camera.inv()
        * second_lidar.inv()
        * first_lidar
        * lidar_from_camera
    )
    rotation_disagreement = (
        visual_rotation.inv() * predicted_camera2_from_camera1
    )

    inlier_matches = [
        match
        for match, keep in zip(matches, homography_mask.ravel().astype(bool))
        if keep
    ]
    visualization = cv2.drawMatches(
        first,
        first_keypoints,
        second,
        second_keypoints,
        inlier_matches[:200],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    result = {
        "source": str(index.path),
        "constraint_type": "rotation_only",
        "first_target_s": first_s,
        "second_target_s": second_s,
        "first_header_ns": first_ns,
        "second_header_ns": second_ns,
        "sift_keypoints": [len(first_keypoints), len(second_keypoints)],
        "ratio_matches": len(matches),
        "homography_inliers": int(np.count_nonzero(homography_mask)),
        "homography_inlier_fraction": float(np.mean(homography_mask)),
        "essential_ransac_inliers": essential_ransac_inliers,
        "recover_pose_cheirality_inliers": int(cheirality_inliers),
        "visual_camera2_from_camera1_quaternion_xyzw": [
            float(value) for value in visual_rotation.as_quat()
        ],
        "visual_rotation_angle_deg": float(
            np.degrees(visual_rotation.magnitude())
        ),
        "visual_translation_direction_camera2": [
            float(value) for value in translation[:, 0]
        ],
        "imu_predicted_rotation_angle_deg": float(
            np.degrees(predicted_camera2_from_camera1.magnitude())
        ),
        "visual_vs_imu_rotation_disagreement_deg": float(
            np.degrees(rotation_disagreement.magnitude())
        ),
        "gyro_bias_rad_s": [float(value) for value in gyro_bias],
        "homography": [[float(value) for value in row] for row in homography],
        "interpretation": (
            "Use rotation only. Low cheirality support makes translation direction "
            "unsuitable as a metric or loop-closure translation constraint."
        ),
    }
    return result, visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--first-s", type=float, default=2.0)
    parser.add_argument("--second-s", type=float, default=159.5)
    parser.add_argument("--ratio-threshold", type=float, default=0.72)
    args = parser.parse_args()
    source_dir = args.mcap.resolve().parent
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source bag directory")
    result, visualization = analyze(
        McapIndex(args.mcap),
        args.first_s,
        args.second_s,
        args.ratio_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "endpoint_visual_constraint.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# 端点重复视角约束",
        "",
        f"- SIFT ratio 匹配：{result['ratio_matches']}",
        f"- Homography 内点：{result['homography_inliers']} "
        f"({result['homography_inlier_fraction']:.1%})",
        f"- Essential RANSAC 内点：{result['essential_ransac_inliers']}",
        f"- recoverPose 正深度内点：{result['recover_pose_cheirality_inliers']}",
        f"- 视觉相对旋转：{result['visual_rotation_angle_deg']:.3f}°",
        f"- IMU 预测相对旋转：{result['imu_predicted_rotation_angle_deg']:.3f}°",
        "- 视觉与 IMU 旋转差："
        f"{result['visual_vs_imu_rotation_disagreement_deg']:.3f}°",
        "",
        "只使用旋转约束。由于正深度内点很少，不使用平移方向，更不施加首尾重合。",
    ]
    (output_dir / "endpoint_visual_constraint.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    encoded, buffer = cv2.imencode(".jpg", visualization)
    if not encoded:
        raise RuntimeError("Failed to encode match visualization")
    buffer.tofile(output_dir / "endpoint_visual_matches.jpg")
    print(output_dir / "endpoint_visual_constraint.json")


if __name__ == "__main__":
    main()
