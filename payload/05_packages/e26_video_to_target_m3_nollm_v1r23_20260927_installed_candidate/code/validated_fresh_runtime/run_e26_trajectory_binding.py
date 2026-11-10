#!/usr/bin/env python
"""Generate E26 image-to-LIO continuous-pose bindings from a frozen config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_camera_recon.e26.contracts import RigidTransformRecord
from lidar_camera_recon.e26.receipts import write_json_new
from lidar_camera_recon.e26.trajectory_io import (
    load_odometry_csv,
    load_tum_trajectory,
    sha256_file,
)
from lidar_camera_recon.e26_trajectory_binding import (
    bind_image_records,
    canonicalize_trajectory_frames,
    load_image_manifest,
)


def load_config(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e26_trajectory_binding_config_v1":
        raise ValueError("unsupported trajectory binding config schema")
    payload["_path"] = str(resolved)
    payload["_sha256"] = sha256_file(resolved)
    return payload


def trajectory_from_config(config: dict[str, object]):
    item = dict(config["trajectory"])
    path = Path(str(item["path"]))
    kind = str(item["format"])
    if kind == "ros_odometry_csv":
        trajectory = load_odometry_csv(
            path,
            time_domain=str(item["time_domain"]),
            clock_source=str(item["clock_source"]),
            timestamp_field=str(item.get("timestamp_field", "field.header.stamp")),
            expected_target_frame=str(item["raw_target_frame"]),
            expected_source_frame=str(item["raw_source_frame"]),
            duplicate_policy=str(item.get("duplicate_policy", "last")),
        )
    elif kind == "tum":
        trajectory = load_tum_trajectory(
            path,
            source_frame=str(item["raw_source_frame"]),
            target_frame=str(item["raw_target_frame"]),
            time_domain=str(item["time_domain"]),
            clock_source=str(item["clock_source"]),
        )
    else:
        raise ValueError(f"unsupported trajectory format: {kind}")
    return canonicalize_trajectory_frames(trajectory)


def extrinsic_from_config(config: dict[str, object]) -> RigidTransformRecord:
    item = dict(config["T_base_link_camera_optical"])
    return RigidTransformRecord.from_pose(
        np.asarray(item["pose_xyz_xyzw"], dtype=np.float64),
        source_frame=str(item["source_frame"]),
        target_frame=str(item["target_frame"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Trajectory binding output already exists: {output}")
    records, manifest_hash = load_image_manifest(Path(str(config["image_manifest"])))
    report = bind_image_records(
        records,
        trajectory_from_config(config),
        extrinsic_from_config(config),
        maximum_bracket_span_ns=int(config["maximum_bracket_span_ns"]),
        trajectory_role=str(dict(config["trajectory"])["role"]),
        image_manifest_sha256=manifest_hash,
    )
    report["config_path"] = config["_path"]
    report["config_sha256"] = config["_sha256"]
    write_json_new(output / "interpolated_camera_poses.json", report)
    comparison = {
        "schema_version": "e26_nearest_vs_interpolated_diagnostic_v1",
        "config_sha256": config["_sha256"],
        "trajectory_sha256": report["trajectory"]["source_sha256"],
        "image_manifest_sha256": manifest_hash,
        "diagnostics": report["diagnostics"],
        "images": [
            {
                key: value
                for key, value in image.items()
                if key
                in {
                    "candidate_id",
                    "image_name",
                    "image_timestamp_ns",
                    "left_pose_timestamp_ns",
                    "right_pose_timestamp_ns",
                    "alpha",
                    "bracket_span_ms",
                    "nearest_pose_timestamp_ns",
                    "nearest_pose_error_ms",
                    "nearest_vs_interpolated_camera_center_delta_m",
                    "nearest_vs_interpolated_rotation_delta_deg",
                }
            }
            for image in report["images"]
        ],
    }
    write_json_new(output / "nearest_vs_interpolated.json", comparison)
    print(json.dumps({k: report[k] for k in ("status", "image_count", "bound_count", "abstained_count", "diagnostics")}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
