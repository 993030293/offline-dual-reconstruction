#!/usr/bin/env python
"""Sliding-local-map LIO baseline with rotational and constant-velocity deskew."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lio_baseline import (
    build_orientation_slerp,
    decode_points,
    deskew_to_scan_end,
    load_imu,
    summary,
    translation_icp,
    voxel_select,
)
from mcap_audit import McapIndex, find_static_transform


TOPIC = "/sensing/lidar/points_raw"


def lidar_from_camera_rotation(index: McapIndex) -> Rotation:
    transform = find_static_transform(index, "base_link", "camera_optical")
    return Rotation.from_quat(transform["rotation_xyzw"])


def build_rotation_correction(
    index: McapIndex,
    slerp,
    constraint: dict | None,
):
    if constraint is None:
        return lambda orientation, timestamp_ns: orientation, None
    first_ns = int(constraint["first_header_ns"])
    second_ns = int(constraint["second_header_ns"])
    desired = Rotation.from_quat(
        constraint["visual_camera2_from_camera1_quaternion_xyzw"]
    )
    lidar_from_camera = lidar_from_camera_rotation(index)
    first_camera = slerp([first_ns / 1e9])[0] * lidar_from_camera
    second_camera = slerp([second_ns / 1e9])[0] * lidar_from_camera
    end_correction = first_camera * desired.inv() * second_camera.inv()
    correction_vector = end_correction.as_rotvec()

    def correct(orientation: Rotation, timestamp_ns: int) -> Rotation:
        alpha = float(
            np.clip((timestamp_ns - first_ns) / (second_ns - first_ns), 0.0, 1.0)
        )
        return Rotation.from_rotvec(correction_vector * alpha) * orientation

    metadata = {
        "constraint_type": constraint["constraint_type"],
        "first_header_ns": first_ns,
        "second_header_ns": second_ns,
        "applied_end_correction_deg": float(
            np.degrees(end_correction.magnitude())
        ),
        "visual_rotation_angle_deg": constraint["visual_rotation_angle_deg"],
        "imu_predicted_rotation_angle_deg": constraint[
            "imu_predicted_rotation_angle_deg"
        ],
        "translation_constraint_used": False,
    }
    return correct, metadata


def point_to_plane_translation_icp(
    source: np.ndarray,
    target: np.ndarray,
    rotation: Rotation,
    initial_translation: np.ndarray,
    maximum_correspondence: float,
    iterations: int,
    plane_constraint: dict | None = None,
) -> tuple[np.ndarray, dict]:
    rotated = rotation.apply(source)
    translation = initial_translation.copy()
    tree = cKDTree(target)
    previous_plane_rmse = np.inf
    metrics = {
        "correspondences": 0,
        "rmse_m": np.inf,
        "plane_rmse_m": np.inf,
        "iterations": 0,
    }
    for iteration in range(iterations):
        moved = rotated + translation
        distances, neighbors = tree.query(moved, k=1, workers=-1)
        eligible = np.flatnonzero(distances < maximum_correspondence)
        if eligible.size < 50:
            break
        trim_limit = np.percentile(distances[eligible], 75)
        selected = eligible[distances[eligible] <= trim_limit]
        matched = target[neighbors[selected]]
        _, normal_neighbors = tree.query(matched, k=12, workers=-1)
        neighborhoods = target[normal_neighbors]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / 11.0
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        residual = np.einsum("ij,ij->i", normals, moved[selected] - matched)
        solve_normals, solve_residual = normals, residual
        if plane_constraint is not None:
            normal = np.asarray(plane_constraint['normal'], dtype=np.float64)
            weight = float(np.sqrt(plane_constraint['independent_inlier_cells']))
            if (normal.shape != (3,) or not np.isfinite(normal).all()
                    or not np.isclose(np.linalg.norm(normal), 1., atol=1e-10, rtol=0)
                    or not np.isfinite([weight, plane_constraint['target_m']]).all() or weight <= 0):
                raise ValueError('The actual measured plane ICP constraint is invalid')
            floor_residual = float(np.dot(normal, translation) - plane_constraint['target_m'])
            solve_normals = np.vstack((normals, weight * normal))
            solve_residual = np.append(residual, weight * floor_residual)
        delta, _, _, _ = np.linalg.lstsq(solve_normals, -solve_residual, rcond=None)
        if np.linalg.norm(delta) > 0.2:
            delta *= 0.2 / np.linalg.norm(delta)
        translation += delta
        point_rmse = float(np.sqrt(np.mean(distances[selected] ** 2)))
        plane_rmse = float(np.sqrt(np.mean(residual**2)))
        metrics = {
            "correspondences": int(selected.size),
            "rmse_m": point_rmse,
            "plane_rmse_m": plane_rmse,
            "iterations": iteration + 1,
        }
        if (
            abs(previous_plane_rmse - plane_rmse) < 1e-5
            and np.linalg.norm(delta) < 1e-4
        ):
            break
        previous_plane_rmse = plane_rmse
    return translation, metrics


def run(
    index: McapIndex,
    voxel_size: float,
    maximum_points: int,
    maximum_map_points: int,
    local_map_frames: int,
    maximum_correspondence: float,
    iterations: int,
    static_hold_s: float,
    maximum_frames: int | None,
    rotation_constraint: dict | None,
    point_to_plane: bool,
    motion_constraint=None,
) -> tuple[dict, dict[str, np.ndarray]]:
    if motion_constraint is not None and (not point_to_plane or rotation_constraint is not None):
        raise ValueError('Measured plane motion requires raw point-to-plane ICP without visual adoption')
    imu_ns, gyro, _ = load_imu(index)
    slerp, gyro_bias = build_orientation_slerp(imu_ns, gyro, static_hold_s)
    correct_orientation, correction_metadata = build_rotation_correction(
        index, slerp, rotation_constraint
    )
    imu_min_ns = int(imu_ns[0])
    imu_max_ns = int(imu_ns[-1])

    timestamps = []
    poses = []
    accepted = [True]
    rmse = [0.0]
    correspondences = [0]
    step_translation = [0.0]
    point_counts = []
    map_counts = [0]
    plane_rmse = [0.0]
    local_scans: deque[np.ndarray] = deque(maxlen=local_map_frames)
    world_translation = np.zeros(3)
    previous_world_translation = np.zeros(3)
    previous_velocity_world = np.zeros(3)
    previous_end_ns = None
    first_scan_ns = None

    for frame_number, message in enumerate(index.iter_messages(TOPIC)):
        if maximum_frames is not None and frame_number >= maximum_frames:
            break
        header_ns, points, offsets_ns = decode_points(message["payload"])
        dense_points, dense_offsets_ns = points, offsets_ns
        points, offsets_ns = voxel_select(
            points, offsets_ns, voxel_size, maximum_points
        )
        point_ns = np.clip(header_ns + offsets_ns, imu_min_ns, imu_max_ns)
        clock_offsets = dense_offsets_ns if motion_constraint is not None else offsets_ns
        end_ns = int(min(header_ns + int(clock_offsets.max()), imu_max_ns))
        points, end_orientation = deskew_to_scan_end(
            points, point_ns, end_ns, slerp
        )
        end_orientation = correct_orientation(end_orientation, end_ns)
        plane_constraint = None
        if motion_constraint is not None:
            dense_point_ns = np.clip(header_ns + dense_offsets_ns, imu_min_ns, imu_max_ns)
            gyro_rotated = slerp(dense_point_ns.astype(np.float64) / 1e9).apply(dense_points)
            dense_dt_s = (dense_point_ns - end_ns).astype(np.float64) / 1e9
            end_orientation, plane_constraint = motion_constraint.propose(gyro_rotated, dense_dt_s,
                previous_velocity_world, end_orientation, world_translation, end_ns)
        point_dt_s = (point_ns - end_ns).astype(np.float64) / 1e9
        translation_deskew_world = (
            point_dt_s[:, None] * previous_velocity_world[None, :]
        )
        points += end_orientation.inv().apply(translation_deskew_world)
        timestamps.append(end_ns)
        point_counts.append(points.shape[0])
        if first_scan_ns is None:
            first_scan_ns = end_ns

        if not local_scans:
            transformed = end_orientation.apply(points) + world_translation
            local_scans.append(transformed)
            pose = np.eye(4)
            pose[:3, :3] = end_orientation.as_matrix()
            pose[:3, 3] = world_translation
            poses.append(pose)
            if motion_constraint is not None:
                accepted[0] = motion_constraint.accepted_translation(plane_constraint, world_translation)
            previous_end_ns = end_ns
            map_counts[0] = transformed.shape[0]
            continue

        local_map = np.vstack(local_scans)
        if local_map.shape[0] > maximum_map_points:
            keep = np.linspace(
                0, local_map.shape[0] - 1, maximum_map_points, dtype=np.int64
            )
            local_map = local_map[keep]
        dt_s = max((end_ns - previous_end_ns) / 1e9, 1e-3)
        initial_translation = world_translation + previous_velocity_world * dt_s
        registration = (
            point_to_plane_translation_icp if point_to_plane else translation_icp
        )
        constraint_arguments = {'plane_constraint': plane_constraint} if motion_constraint is not None else {}
        estimated_translation, metrics = registration(
            points,
            local_map,
            end_orientation,
            initial_translation,
            maximum_correspondence,
            iterations,
            **constraint_arguments,
        )
        step = estimated_translation - world_translation
        step_norm = float(np.linalg.norm(step))
        relative_s = (end_ns - first_scan_ns) / 1e9
        is_static_hold = relative_s <= static_hold_s
        valid = (
            metrics["correspondences"] >= 100
            and metrics["rmse_m"] <= maximum_correspondence * 0.75
            and step_norm <= 0.6
        )
        if is_static_hold:
            estimated_translation = np.zeros(3)
            step = estimated_translation - world_translation
            step_norm = float(np.linalg.norm(step))
            valid = True
        if motion_constraint is not None:
            plane_valid = motion_constraint.accepted_translation(plane_constraint, estimated_translation)
            motion_constraint.rows[-1]['actual_native_step_quality_passed'] = bool(valid)
            valid = bool(valid and plane_valid)
            motion_constraint.rows[-1]['motion_accepted'] = valid
        if not valid and not is_static_hold:
            estimated_translation = world_translation
            step = np.zeros(3)
            step_norm = 0.0

        previous_world_translation = world_translation.copy()
        world_translation = estimated_translation
        previous_velocity_world = (
            world_translation - previous_world_translation
        ) / dt_s
        transformed = end_orientation.apply(points) + world_translation
        if motion_constraint is None or valid or motion_constraint.reference_normal is None:
            local_scans.append(transformed)
        pose = np.eye(4)
        pose[:3, :3] = end_orientation.as_matrix()
        pose[:3, 3] = world_translation
        poses.append(pose)
        accepted.append(valid)
        rmse.append(metrics["rmse_m"])
        plane_rmse.append(metrics.get("plane_rmse_m", np.nan))
        correspondences.append(metrics["correspondences"])
        step_translation.append(step_norm)
        map_counts.append(local_map.shape[0])
        previous_end_ns = end_ns

    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    pose_array = np.asarray(poses)
    xyz = pose_array[:, :3, 3]
    relative_s_array = (
        timestamp_array - timestamp_array[0]
    ).astype(np.float64) / 1e9
    accepted_array = np.asarray(accepted, dtype=bool)
    rmse_array = np.asarray(rmse, dtype=np.float64)
    final_mask = relative_s_array >= max(float(relative_s_array[-1]) - 2.0, 0.0)
    final_xyz = xyz[final_mask]
    start_to_end_rotation_deg = float(
        np.degrees(
            (
                Rotation.from_matrix(pose_array[0, :3, :3]).inv()
                * Rotation.from_matrix(pose_array[-1, :3, :3])
            ).magnitude()
        )
    )
    report = {
        "source": str(index.path),
        "method": (
            "gyro rotational deskew, constant-velocity translational deskew, "
            "IMU orientation, translation-only ICP to a sliding local map"
        ),
        "frame_count": int(timestamp_array.size),
        "duration_s": float(relative_s_array[-1]),
        "gyro_bias_rad_s": [float(value) for value in gyro_bias],
        "rotation_constraint": correction_metadata,
        "registration_residual": (
            "point_to_plane" if point_to_plane else "point_to_point"
        ),
        "parameters": {
            "voxel_size_m": voxel_size,
            "maximum_points": maximum_points,
            "maximum_map_points": maximum_map_points,
            "local_map_frames": local_map_frames,
            "maximum_correspondence_m": maximum_correspondence,
            "iterations": iterations,
            "confirmed_static_hold_s": static_hold_s,
        },
        "accepted_steps": int(np.count_nonzero(accepted_array[1:])),
        "rejected_steps": int(np.count_nonzero(~accepted_array[1:])),
        "acceptance_fraction": float(np.mean(accepted_array[1:])),
        "downsampled_points": summary(np.asarray(point_counts)),
        "local_map_points": summary(np.asarray(map_counts)),
        "correspondences": summary(np.asarray(correspondences[1:])),
        "rmse_m": summary(rmse_array[1:]),
        "plane_rmse_m": (
            summary(np.asarray(plane_rmse[1:]))
            if point_to_plane
            else None
        ),
        "step_translation_m": summary(np.asarray(step_translation[1:])),
        "final_2s_max_displacement_m": float(
            np.max(np.linalg.norm(final_xyz - final_xyz[0], axis=1))
        ),
        "start_to_end_translation_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "start_to_end_rotation_deg": start_to_end_rotation_deg,
        "trajectory_length_m": float(
            np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        ),
        "limitations": [
            "Translation deskew assumes constant velocity from the previous step",
            "No accelerometer propagation",
            "Gyro orientation has no global correction",
            "No loop closure",
            "The confirmed initial static interval is explicitly held fixed",
        ],
    }
    if motion_constraint is not None:
        report['method'] = (
            'gyro prediction with measured low-plane orientation correction, '
            'constant-velocity translational deskew, point-to-plane translation '
            'ICP with the same measured plane in a sliding local map'
        )
        report['registration_residual'] = 'point_to_plane_with_measured_reference_plane'
        report['parameters']['confirmed_static_hold_s'] = None
        report['parameters']['assumed_initial_static_hold_s'] = static_hold_s
        report['limitations'] = [
            'Translation deskew assumes constant velocity from the previous step',
            'No accelerometer propagation or independent trajectory ground truth',
            'One measured low plane is assumed to remain fixed across scans',
            'The raw gyro correction is constant within each scan',
            'Insufficient measured floor support explicitly rejects scan evidence',
            'No loop closure',
            'The initial static interval is an unconfirmed algorithm assumption',
        ]
    series = {
        "timestamps_ns": timestamp_array,
        "relative_s": relative_s_array,
        "poses": pose_array,
        "accepted": accepted_array,
        "rmse": rmse_array,
        "plane_rmse": np.asarray(plane_rmse),
        "correspondences": np.asarray(correspondences),
        "point_counts": np.asarray(point_counts),
        "map_counts": np.asarray(map_counts),
        "step_translation": np.asarray(step_translation),
    }
    return report, series


def write_outputs(output_dir: Path, report: dict, series: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lio_local_map.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "lio_local_map.tum").open("w", encoding="utf-8") as stream:
        for timestamp_ns, pose in zip(series["timestamps_ns"], series["poses"]):
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            translation = pose[:3, 3]
            stream.write(
                f"{timestamp_ns / 1e9:.9f} "
                f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} "
                f"{quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )
    markdown = [
        "# 滑动局部地图 LIO 基线",
        "",
        f"- 帧数：{report['frame_count']}",
        f"- 配准残差：{report['registration_residual']}",
        f"- 接受/拒绝步：{report['accepted_steps']}/{report['rejected_steps']}",
        f"- 接受率：{report['acceptance_fraction']:.3%}",
        f"- ICP RMSE 中位数：{report['rmse_m']['median']:.4f} m",
        f"- 末 2 秒最大位移：{report['final_2s_max_displacement_m']:.4f} m",
        f"- 累计路径长度：{report['trajectory_length_m']:.3f} m",
        f"- 首尾距离：{report['start_to_end_translation_m']:.3f} m",
        f"- 首尾姿态差：{report['start_to_end_rotation_deg']:.3f}°",
    ]
    if report["rotation_constraint"] is not None:
        markdown.append(
            "- 长基线视觉旋转校正："
            f"{report['rotation_constraint']['applied_end_correction_deg']:.3f}°"
        )
    markdown.extend(
        [
            "",
            "该版本增加扫描内恒速平移补偿和滑动局部地图；"
            "视觉约束只校正旋转，不约束端点平移或强制首尾重合。",
        ]
    )
    (output_dir / "lio_local_map.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    xyz = series["poses"][:, :3, 3]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(xyz[:, 0], xyz[:, 1], linewidth=0.8)
    axes[0, 0].scatter(xyz[0, 0], xyz[0, 1], label="start", s=25)
    axes[0, 0].scatter(xyz[-1, 0], xyz[-1, 1], label="end", s=25)
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(series["relative_s"], xyz)
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("position (m)")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(series["relative_s"], series["rmse"])
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("local-map ICP RMSE (m)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(series["relative_s"], series["step_translation"])
    rejected = ~series["accepted"]
    axes[1, 1].scatter(
        series["relative_s"][rejected],
        np.zeros(np.count_nonzero(rejected)),
        color="red",
        s=8,
        label="rejected",
    )
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("step translation (m)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "lio_local_map.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--maximum-points", type=int, default=2500)
    parser.add_argument("--maximum-map-points", type=int, default=10000)
    parser.add_argument("--local-map-frames", type=int, default=8)
    parser.add_argument("--maximum-correspondence", type=float, default=0.45)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--static-hold-s", type=float, default=11.5)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--rotation-constraint", type=Path)
    parser.add_argument("--point-to-plane", action="store_true")
    args = parser.parse_args()
    source_dir = args.mcap.resolve().parent
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source bag directory")
    rotation_constraint = None
    if args.rotation_constraint is not None:
        rotation_constraint = json.loads(
            args.rotation_constraint.read_text(encoding="utf-8")
        )
    report, series = run(
        McapIndex(args.mcap),
        args.voxel_size,
        args.maximum_points,
        args.maximum_map_points,
        args.local_map_frames,
        args.maximum_correspondence,
        args.iterations,
        args.static_hold_s,
        args.maximum_frames,
        rotation_constraint,
        args.point_to_plane,
    )
    write_outputs(output_dir, report, series)
    print(output_dir / "lio_local_map.json")


if __name__ == "__main__":
    main()
