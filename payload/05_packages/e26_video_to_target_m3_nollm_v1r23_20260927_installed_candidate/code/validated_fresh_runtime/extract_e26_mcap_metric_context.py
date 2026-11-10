#!/usr/bin/env python
"""Extract metric odometry and the camera extrinsic from one E26 MCAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ODOMETRY_TOPIC = "/localization/pose/odom"
TF_STATIC_TOPIC = "/tf_static"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(stamp: Any) -> int:
    """Convert a ROS builtin time value to integer nanoseconds."""

    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_xyz_xyzw(pose: Any) -> list[float]:
    """Serialize one ROS pose-like value as xyz plus quaternion xyzw."""

    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def transform_xyz_xyzw(transform: Any) -> list[float]:
    """Serialize one ROS transform as xyz plus quaternion xyzw."""

    return [
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
        float(transform.rotation.x),
        float(transform.rotation.y),
        float(transform.rotation.z),
        float(transform.rotation.w),
    ]


def normalize_frame(value: str) -> str:
    """Normalize ROS frame names without changing their semantic identity."""

    return str(value).strip().lstrip("/")


def find_transform(
    transforms: list[Any], *, parent: str, child: str
) -> dict[str, Any]:
    """Find exactly one requested static T_parent_child transform."""

    parent = normalize_frame(parent)
    child = normalize_frame(child)
    matches = [
        item
        for item in transforms
        if normalize_frame(item.header.frame_id) == parent
        and normalize_frame(item.child_frame_id) == child
    ]
    if len(matches) != 1:
        available = [
            {
                "parent": normalize_frame(item.header.frame_id),
                "child": normalize_frame(item.child_frame_id),
            }
            for item in transforms
        ]
        raise ValueError(
            f"expected exactly one static transform {parent} <- {child}; "
            f"found {len(matches)}; available={available}"
        )
    item = matches[0]
    return {
        "source_frame": child,
        "target_frame": parent,
        "pose_xyz_xyzw": transform_xyz_xyzw(item.transform),
        "transform_convention": "T_target_source",
    }


def load_external_extrinsic(path: Path) -> dict[str, Any]:
    """Load an explicitly bound frozen camera extrinsic configuration."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    item = payload.get("T_base_link_camera_optical", payload)
    if not isinstance(item, dict):
        raise ValueError("external camera extrinsic is not a JSON object")
    source = normalize_frame(str(item.get("source_frame", "")))
    target = normalize_frame(str(item.get("target_frame", "")))
    pose = item.get("pose_xyz_xyzw")
    if source != "camera_optical" or target != "base_link":
        raise ValueError("external camera extrinsic has an unexpected direction")
    if not isinstance(pose, list) or len(pose) != 7:
        raise ValueError("external camera extrinsic pose must have seven values")
    return {
        "source_frame": source,
        "target_frame": target,
        "pose_xyz_xyzw": [float(value) for value in pose],
        "transform_convention": "T_target_source",
        "external_source": str(resolved),
        "external_source_sha256": sha256_file(resolved),
    }


def write_json_new(path: Path, value: Any) -> None:
    """Write JSON without overwriting an existing artifact."""

    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_tum_trajectory(path: Path) -> tuple[list[int], list[list[float]]]:
    """Validate a TUM trajectory without changing its serialized values."""

    timestamps = []
    poses = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"TUM line {line_number} must contain eight fields")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"TUM line {line_number} contains a nonfinite value")
        quaternion_norm = sum(value * value for value in values[4:8]) ** 0.5
        if abs(quaternion_norm - 1.0) > 1e-4:
            raise ValueError(f"TUM line {line_number} has a non-unit quaternion")
        timestamps.append(int(round(values[0] * 1_000_000_000)))
        poses.append(values[1:])
    if len(timestamps) < 2:
        raise ValueError("fallback metric trajectory has fewer than two samples")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("fallback metric trajectory timestamps are not strictly increasing")
    return timestamps, poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--expected-mcap-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_optical")
    parser.add_argument("--extrinsic-json", type=Path)
    parser.add_argument("--trajectory-file", type=Path)
    parser.add_argument("--skip-full-mcap-hash", action="store_true")
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader

    mcap = args.mcap.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected_hash = args.expected_mcap_sha256.lower()
    actual_hash = None if args.skip_full_mcap_hash else sha256_file(mcap)
    if actual_hash is not None and actual_hash != expected_hash:
        raise ValueError("raw MCAP SHA256 mismatch")

    rows: list[tuple[int, list[float], str, str]] = []
    static_transforms: list[Any] = []
    with AnyReader([mcap]) as reader:
        odom_connections = [
            item for item in reader.connections if item.topic == ODOMETRY_TOPIC
        ]
        static_connections = [
            item for item in reader.connections if item.topic == TF_STATIC_TOPIC
        ]
        expected_odometry_count = 0 if args.trajectory_file is not None else 1
        if len(odom_connections) != expected_odometry_count or len(static_connections) != 1:
            raise ValueError(
                "unexpected odometry or tf_static connection count: "
                f"odometry={len(odom_connections)}, expected_odometry={expected_odometry_count}, "
                f"tf_static={len(static_connections)}"
            )
        for connection, _log_time, raw in reader.messages(
            connections=odom_connections + static_connections
        ):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == ODOMETRY_TOPIC:
                rows.append(
                    (
                        timestamp_ns(message.header.stamp),
                        pose_xyz_xyzw(message.pose.pose),
                        normalize_frame(message.header.frame_id),
                        normalize_frame(message.child_frame_id),
                    )
                )
            else:
                static_transforms.extend(message.transforms)

    trajectory_path = output / "metric_odometry.tum"
    if args.trajectory_file is None:
        if len(rows) < 2:
            raise ValueError("odometry trajectory has fewer than two samples")
        rows.sort(key=lambda item: item[0])
        timestamps = [item[0] for item in rows]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("odometry header timestamps are not strictly increasing")
        parent_frames = sorted({item[2] for item in rows})
        child_frames = sorted({item[3] for item in rows})
        if len(parent_frames) != 1 or len(child_frames) != 1:
            raise ValueError("odometry frame IDs are not constant")
        with trajectory_path.open("x", encoding="utf-8", newline="\n") as handle:
            for stamp_ns, pose, _parent, _child in rows:
                values = " ".join(f"{value:.12g}" for value in pose)
                handle.write(f"{stamp_ns / 1_000_000_000.0:.9f} {values}\n")
        odometry_source = {
            "topic": ODOMETRY_TOPIC,
            "source_role": "recorded_current_mcap_odometry",
        }
    else:
        fallback = args.trajectory_file.resolve(strict=True)
        timestamps, poses = read_tum_trajectory(fallback)
        parent_frames = ["world_lio"]
        child_frames = ["base_link"]
        with trajectory_path.open("xb") as handle:
            handle.write(fallback.read_bytes())
        odometry_source = {
            "topic": None,
            "source_role": "fresh_current_mcap_raw_lidar_imu_visual_rotation_lio",
            "fallback_source_path": str(fallback),
            "fallback_source_sha256": sha256_file(fallback),
            "fallback_pose_count": len(poses),
        }

    try:
        extrinsic = find_transform(
            static_transforms, parent=args.parent_frame, child=args.camera_frame
        )
        extrinsic["source_role"] = "current_mcap_tf_static"
    except ValueError:
        if args.extrinsic_json is None:
            raise
        extrinsic = load_external_extrinsic(args.extrinsic_json)
        extrinsic["source_role"] = "explicit_frozen_calibration_fallback"
    extrinsic_path = output / "T_base_link_camera_optical.json"
    write_json_new(extrinsic_path, extrinsic)
    receipt = {
        "schema_version": "e26_mcap_metric_context_v1",
        "status": "complete",
        "mcap": {
            "path": str(mcap),
            "bytes": mcap.stat().st_size,
            "registered_sha256": expected_hash,
            "recomputed_sha256": actual_hash,
        },
        "odometry": {
            **odometry_source,
            "sample_count": len(rows),
            "start_timestamp_ns": timestamps[0],
            "end_timestamp_ns": timestamps[-1],
            "raw_target_frame": parent_frames[0],
            "raw_source_frame": child_frames[0],
            "tum_path": str(trajectory_path),
            "tum_sha256": sha256_file(trajectory_path),
        },
        "camera_extrinsic": {
            **extrinsic,
            "path": str(extrinsic_path),
            "sha256": sha256_file(extrinsic_path),
        },
        "scientific_boundary": (
            "The metric trajectory is either recorded odometry or a deterministic estimate "
            "from the current MCAP LiDAR, IMU, and rotation-only visual endpoint constraint. "
            "It is a pose prior, not proof of absolute trajectory or reconstruction accuracy."
        ),
    }
    write_json_new(output / "metric_context_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
