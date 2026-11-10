#!/usr/bin/env python
"""IMU-rotation deskew plus translation-only ICP baseline for Livox AVIA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from mcap_audit import Cdr, McapIndex, decode_imu


LIDAR_TOPIC = "/sensing/lidar/points_raw"
IMU_TOPIC = "/sensing/lidar/imu"


def decode_points(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    cdr = Cdr(payload)
    sec, nanosec, _ = cdr.header()
    height = cdr.u32()
    width = cdr.u32()
    field_count = cdr.u32()
    offsets = {}
    for _ in range(field_count):
        name = cdr.string()
        offsets[name] = cdr.u32()
        cdr.u8()
        cdr.u32()
    is_bigendian = cdr.u8()
    point_step = cdr.u32()
    cdr.u32()
    data_length = cdr.u32()
    if is_bigendian:
        raise ValueError("Big-endian PointCloud2 is not supported")
    count = height * width
    if data_length < count * point_step:
        raise ValueError("PointCloud2 payload is truncated")
    raw = memoryview(payload)[cdr.offset : cdr.offset + data_length]
    xyz = np.ndarray(
        (count, 3),
        dtype="<f4",
        buffer=raw,
        offset=offsets["x"],
        strides=(point_step, 4),
    ).astype(np.float64)
    time_offset_ns = np.ndarray(
        (count,),
        dtype="<u4",
        buffer=raw,
        offset=offsets["time_offset_ns"],
        strides=(point_step,),
    ).astype(np.int64)
    squared = np.einsum("ij,ij->i", xyz, xyz)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (squared >= 0.15**2)
        & (squared <= 30.0**2)
    )
    return (
        sec * 1_000_000_000 + nanosec,
        xyz[valid],
        time_offset_ns[valid],
    )


def voxel_select(
    points: np.ndarray,
    offsets_ns: np.ndarray,
    voxel_size: float,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int32)
    _, selected = np.unique(keys, axis=0, return_index=True)
    selected.sort()
    if selected.size > maximum:
        keep = np.linspace(0, selected.size - 1, maximum, dtype=np.int64)
        selected = selected[keep]
    return points[selected], offsets_ns[selected]


def load_imu(index: McapIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    gyro = []
    accel = []
    for message in index.iter_messages(IMU_TOPIC, prefix_bytes=512):
        decoded = decode_imu(message["payload"])
        timestamps.append(int(decoded["header_ns"]))
        gyro.append(decoded["angular_velocity"])
        accel.append(decoded["linear_acceleration"])
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(gyro, dtype=np.float64),
        np.asarray(accel, dtype=np.float64),
    )


def build_orientation_slerp(
    imu_ns: np.ndarray,
    gyro: np.ndarray,
    bias_end_s: float,
) -> tuple[Slerp, np.ndarray]:
    relative_s = (imu_ns - imu_ns[0]).astype(np.float64) / 1e9
    bias_mask = (relative_s >= 0.5) & (relative_s <= bias_end_s)
    bias = gyro[bias_mask].mean(axis=0)
    corrected = gyro - bias
    quaternions = np.empty((imu_ns.size, 4), dtype=np.float64)
    orientation = Rotation.identity()
    quaternions[0] = orientation.as_quat()
    for index in range(1, imu_ns.size):
        dt = (imu_ns[index] - imu_ns[index - 1]) / 1e9
        omega = 0.5 * (corrected[index - 1] + corrected[index])
        orientation = orientation * Rotation.from_rotvec(omega * dt)
        quaternions[index] = orientation.as_quat()
    time_s = imu_ns.astype(np.float64) / 1e9
    return Slerp(time_s, Rotation.from_quat(quaternions)), bias


def deskew_to_scan_end(
    points: np.ndarray,
    point_ns: np.ndarray,
    end_ns: int,
    slerp: Slerp,
) -> tuple[np.ndarray, Rotation]:
    point_s = point_ns.astype(np.float64) / 1e9
    end_s = end_ns / 1e9
    point_orientation = slerp(point_s)
    end_orientation = slerp([end_s])[0]
    end_from_point = end_orientation.inv() * point_orientation
    return end_from_point.apply(points), end_orientation


def translation_icp(
    source: np.ndarray,
    target: np.ndarray,
    rotation: Rotation,
    initial_translation: np.ndarray,
    maximum_correspondence: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    rotated = rotation.apply(source)
    translation = initial_translation.copy()
    tree = cKDTree(target)
    previous_rmse = math.inf
    metrics = {"correspondences": 0, "rmse_m": math.inf, "iterations": 0}
    for iteration in range(iterations):
        moved = rotated + translation
        distances, neighbors = tree.query(moved, k=1, workers=-1)
        eligible = np.flatnonzero(distances < maximum_correspondence)
        if eligible.size < 50:
            break
        trim_limit = np.percentile(distances[eligible], 75)
        selected = eligible[distances[eligible] <= trim_limit]
        residual = target[neighbors[selected]] - moved[selected]
        delta = np.median(residual, axis=0)
        translation += delta
        rmse = float(np.sqrt(np.mean(distances[selected] ** 2)))
        metrics = {
            "correspondences": int(selected.size),
            "rmse_m": rmse,
            "iterations": iteration + 1,
        }
        if abs(previous_rmse - rmse) < 1e-5 and np.linalg.norm(delta) < 1e-4:
            break
        previous_rmse = rmse
    return translation, metrics


def summary(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def run(
    index: McapIndex,
    voxel_size: float,
    maximum_points: int,
    maximum_correspondence: float,
    iterations: int,
    static_hold_s: float,
    maximum_frames: int | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    imu_ns, gyro, _ = load_imu(index)
    slerp, gyro_bias = build_orientation_slerp(imu_ns, gyro, static_hold_s)
    imu_min_ns = int(imu_ns[0])
    imu_max_ns = int(imu_ns[-1])

    timestamps = []
    poses = []
    accepted = [True]
    held_static = [True]
    rmse = [0.0]
    correspondences = [0]
    step_translation = [0.0]
    point_counts = []
    previous_points = None
    previous_end_orientation = None
    previous_translation = np.zeros(3)
    world_translation = np.zeros(3)
    first_scan_ns = None

    for frame_number, message in enumerate(index.iter_messages(LIDAR_TOPIC)):
        if maximum_frames is not None and frame_number >= maximum_frames:
            break
        header_ns, points, offsets_ns = decode_points(message["payload"])
        points, offsets_ns = voxel_select(
            points, offsets_ns, voxel_size, maximum_points
        )
        point_ns = np.clip(header_ns + offsets_ns, imu_min_ns, imu_max_ns)
        end_ns = int(min(header_ns + int(offsets_ns.max()), imu_max_ns))
        points, end_orientation = deskew_to_scan_end(
            points, point_ns, end_ns, slerp
        )
        timestamps.append(end_ns)
        point_counts.append(points.shape[0])
        if first_scan_ns is None:
            first_scan_ns = end_ns
        if previous_points is None:
            pose = np.eye(4)
            pose[:3, :3] = end_orientation.as_matrix()
            pose[:3, 3] = world_translation
            poses.append(pose)
            previous_points = points
            previous_end_orientation = end_orientation
            continue

        previous_from_current_rotation = (
            previous_end_orientation.inv() * end_orientation
        )
        relative_s = (end_ns - first_scan_ns) / 1e9
        translation, metrics = translation_icp(
            points,
            previous_points,
            previous_from_current_rotation,
            previous_translation,
            maximum_correspondence,
            iterations,
        )
        translation_norm = float(np.linalg.norm(translation))
        is_static_hold = relative_s <= static_hold_s
        valid = (
            metrics["correspondences"] >= 100
            and metrics["rmse_m"] <= maximum_correspondence * 0.75
            and translation_norm <= 0.75
        )
        if is_static_hold:
            translation = np.zeros(3)
            translation_norm = 0.0
            valid = True
        elif not valid:
            translation = np.zeros(3)
            translation_norm = 0.0

        world_translation = (
            world_translation + previous_end_orientation.apply(translation)
        )
        pose = np.eye(4)
        pose[:3, :3] = end_orientation.as_matrix()
        pose[:3, 3] = world_translation
        poses.append(pose)
        accepted.append(valid)
        held_static.append(is_static_hold)
        rmse.append(metrics["rmse_m"])
        correspondences.append(metrics["correspondences"])
        step_translation.append(translation_norm)
        previous_translation = translation
        previous_points = points
        previous_end_orientation = end_orientation

    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    pose_array = np.asarray(poses)
    xyz = pose_array[:, :3, 3]
    relative_s_array = (
        timestamp_array - timestamp_array[0]
    ).astype(np.float64) / 1e9
    accepted_array = np.asarray(accepted, dtype=bool)
    rmse_array = np.asarray(rmse, dtype=np.float64)
    static_mask = relative_s_array <= static_hold_s
    static_xyz = xyz[static_mask]
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
            "gyro-bias-corrected rotational deskew, IMU relative rotation, "
            "translation-only scan-to-scan ICP"
        ),
        "frame_count": int(timestamp_array.size),
        "duration_s": float(relative_s_array[-1]),
        "gyro_bias_rad_s": [float(value) for value in gyro_bias],
        "parameters": {
            "voxel_size_m": voxel_size,
            "maximum_points": maximum_points,
            "maximum_correspondence_m": maximum_correspondence,
            "iterations": iterations,
            "confirmed_static_hold_s": static_hold_s,
        },
        "accepted_steps": int(np.count_nonzero(accepted_array[1:])),
        "rejected_steps": int(np.count_nonzero(~accepted_array[1:])),
        "acceptance_fraction": float(np.mean(accepted_array[1:])),
        "downsampled_points": summary(np.asarray(point_counts)),
        "correspondences": summary(np.asarray(correspondences[1:])),
        "rmse_m": summary(rmse_array[1:]),
        "step_translation_m": summary(np.asarray(step_translation[1:])),
        "initial_static_max_displacement_m": float(
            np.max(np.linalg.norm(static_xyz - static_xyz[0], axis=1))
        ),
        "final_2s_max_displacement_m": float(
            np.max(np.linalg.norm(final_xyz - final_xyz[0], axis=1))
        ),
        "start_to_end_translation_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "start_to_end_rotation_deg": start_to_end_rotation_deg,
        "trajectory_length_m": float(
            np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        ),
        "limitations": [
            "Translation during each 100 ms scan is not deskewed",
            "No accelerometer propagation",
            "No local map",
            "No loop closure",
            "The confirmed initial static interval is explicitly held fixed",
        ],
    }
    series = {
        "timestamps_ns": timestamp_array,
        "relative_s": relative_s_array,
        "poses": pose_array,
        "accepted": accepted_array,
        "held_static": np.asarray(held_static, dtype=bool),
        "rmse": rmse_array,
        "correspondences": np.asarray(correspondences),
        "point_counts": np.asarray(point_counts),
        "step_translation": np.asarray(step_translation),
    }
    return report, series


def write_outputs(output_dir: Path, report: dict, series: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lio_baseline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "lio_baseline.tum").open("w", encoding="utf-8") as stream:
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
        "# IMU 辅助 LiDAR 基线",
        "",
        f"- 帧数：{report['frame_count']}",
        f"- 接受/拒绝步：{report['accepted_steps']}/{report['rejected_steps']}",
        f"- 接受率：{report['acceptance_fraction']:.3%}",
        f"- ICP RMSE 中位数：{report['rmse_m']['median']:.4f} m",
        f"- 初始静止段最大位移：{report['initial_static_max_displacement_m']:.4f} m",
        f"- 末 2 秒最大位移：{report['final_2s_max_displacement_m']:.4f} m",
        f"- 累计路径长度：{report['trajectory_length_m']:.3f} m",
        f"- 首尾距离：{report['start_to_end_translation_m']:.3f} m",
        f"- 首尾姿态差：{report['start_to_end_rotation_deg']:.3f}°",
        "",
        "该基线已做旋转去畸变并使用 IMU 相对旋转，但仍无扫描内平移去畸变、"
        "加速度传播、局部地图和闭环，不作为最终精度结论。",
    ]
    (output_dir / "lio_baseline.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    xyz = series["poses"][:, :3, 3]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(xyz[:, 0], xyz[:, 1], linewidth=0.8)
    axes[0, 0].scatter(xyz[0, 0], xyz[0, 1], label="start", s=25)
    axes[0, 0].scatter(xyz[-1, 0], xyz[-1, 1], label="end", s=25)
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(series["relative_s"], xyz)
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("position (m)")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(series["relative_s"], series["rmse"])
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("translation ICP RMSE (m)")
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
    fig.savefig(output_dir / "lio_baseline.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--maximum-points", type=int, default=3500)
    parser.add_argument("--maximum-correspondence", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--static-hold-s", type=float, default=11.5)
    parser.add_argument("--maximum-frames", type=int)
    args = parser.parse_args()
    source_dir = args.mcap.resolve().parent
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source bag directory")
    report, series = run(
        McapIndex(args.mcap),
        args.voxel_size,
        args.maximum_points,
        args.maximum_correspondence,
        args.iterations,
        args.static_hold_s,
        args.maximum_frames,
    )
    write_outputs(output_dir, report, series)
    print(output_dir / "lio_baseline.json")


if __name__ == "__main__":
    main()
