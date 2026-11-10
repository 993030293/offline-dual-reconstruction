"""Select metric-supported frames and write policy-bound pose configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve(strict=True)
    policy_path = args.policy.resolve(strict=True)
    policy = read_json(policy_path)
    if policy.get("schema_version") != "e26_metric_alignment_policy_v1":
        raise ValueError("Unsupported metric alignment policy schema")
    policy_sha256 = sha256_file(policy_path)
    metric = run / "03_metric_context"
    visual = run / "01_visual_workspace"
    binding = run / "04_trajectory_binding"
    receipt = read_json(metric / "metric_context_receipt.json")
    registered = read_json(visual / "registered_image_manifest.json")
    start = int(receipt["odometry"]["start_timestamp_ns"])
    end = int(receipt["odometry"]["end_timestamp_ns"])
    supported = [row for row in registered if start <= int(row["camera_header_ns"]) <= end]
    excluded = [
        {"image_name": row["image_name"], "camera_header_ns": int(row["camera_header_ns"])}
        for row in registered
        if not start <= int(row["camera_header_ns"]) <= end
    ]
    support_policy = policy["metric_time_support"]
    minimum_count = int(support_policy["minimum_supported_frames"])
    minimum_fraction = float(support_policy["minimum_supported_fraction"])
    supported_fraction = len(supported) / len(registered) if registered else 0.0
    support_passed = len(supported) >= minimum_count and supported_fraction >= minimum_fraction
    support_receipt = {
        "schema_version": "e26_metric_time_support_filter_v2",
        "status": "complete" if support_passed else "failed_closed",
        "input_registered_count": len(registered),
        "metric_bound_count": len(supported),
        "metric_bound_fraction": supported_fraction,
        "visual_only_count": len(excluded),
        "minimum_supported_frames": minimum_count,
        "minimum_supported_fraction": minimum_fraction,
        "rule": "camera_header_ns within inclusive raw odometry timestamp range; no extrapolation",
        "odometry_start_ns": start,
        "odometry_end_ns": end,
        "excluded": excluded,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "language_model_used": False,
    }
    write_json_new(metric / "metric_time_support_receipt.json", support_receipt)
    if not support_passed:
        raise RuntimeError(
            "Metric time support gate failed: "
            f"count={len(supported)}/{minimum_count}, "
            f"fraction={supported_fraction:.6f}/{minimum_fraction:.6f}"
        )
    write_json_new(metric / "registered_metric_bound_manifest.json", supported)
    extrinsic = receipt["camera_extrinsic"]
    odometry = receipt["odometry"]
    odometry_source_role = odometry.get(
        "source_role", "recorded_current_mcap_odometry"
    )
    trajectory_config = {
        "schema_version": "e26_trajectory_binding_config_v1",
        "image_manifest": str(metric / "registered_metric_bound_manifest.json"),
        "trajectory": {
            "path": str(metric / "metric_odometry.tum"),
            "format": "tum",
            "role": odometry_source_role,
            "raw_source_frame": odometry["raw_source_frame"],
            "raw_target_frame": odometry["raw_target_frame"],
            "time_domain": "sensor_header_relative_epoch",
            "clock_source": "mcap_sensor_header",
            "parameter_source": odometry_source_role,
        },
        "T_base_link_camera_optical": {
            "source_frame": extrinsic["source_frame"],
            "target_frame": extrinsic["target_frame"],
            "pose_xyz_xyzw": extrinsic["pose_xyz_xyzw"],
            "parameter_source": "current MCAP tf_static or explicitly hashed calibration fallback",
        },
        "maximum_bracket_span_ns": int(policy["trajectory_binding"]["maximum_bracket_span_ns"]),
        "maximum_bracket_span_parameter_source": policy["trajectory_binding"]["parameter_source"],
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "scientific_boundary": "Offline bracketing uses future samples and is not an online-causal estimate.",
    }
    write_json_new(metric / "trajectory_binding_config.json", trajectory_config)
    sim3_config = {
        "schema_version": "e26_colmap_pose_refinement_config_v1",
        "baseline_sparse_model": str(visual / "sparse" / "0"),
        "trajectory_binding_report": str(binding / "interpolated_camera_poses.json"),
        "sim3_initialization": {
            **policy["sim3_initialization"],
            "parameter_source": "versioned metric alignment policy",
        },
        "lio_prior": {
            **policy["lio_prior"],
            "position_parameter_source": "frozen framewise correction scale",
            "rotation_parameter_source": "frozen framewise correction scale",
        },
        "camera_geometry_admission": policy["camera_geometry_admission"],
        "model_identity": policy["model_identity"],
        "point_filter": policy["point_filter"],
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "scientific_boundary": (
            "This stage estimates visual-to-odometry Sim3. Sim3 is not sufficient by itself "
            "to admit a camera as a geometry reference."
        ),
    }
    write_json_new(metric / "sim3_config.json", sim3_config)
    print(json.dumps({"registered": len(registered), "metric_supported": len(supported), "excluded": excluded}, indent=2))


if __name__ == "__main__":
    main()
