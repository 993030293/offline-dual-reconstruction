#!/usr/bin/env python
"""Estimate robust SfM-to-LIO Sim(3) and emit the pose-prior manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import pycolmap

from lidar_camera_recon.e26.input_freeze import directory_signature
from lidar_camera_recon.e26.receipts import write_json_new
from lidar_camera_recon.e26.trajectory_io import sha256_file
from lidar_camera_recon.e26_pose_prior_builder import (
    build_pose_correspondences,
    load_binding_report,
    serialize_pose_prior_manifest,
)
from lidar_camera_recon.e26_sim3_initialization import (
    robust_similarity_ransac,
    time_binned_residuals,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "e26_colmap_pose_refinement_config_v1":
        raise ValueError("unsupported pose refinement config schema")
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Sim3 output already exists: {output}")
    policy_path = Path(str(config["policy_path"])).resolve(strict=True)
    policy_sha256 = sha256_file(policy_path)
    if policy_sha256 != config.get("policy_sha256"):
        raise RuntimeError("Metric alignment policy identity changed after config generation")

    binding_path = Path(str(config["trajectory_binding_report"])).resolve(strict=True)
    binding_payload = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding_payload.get("schema_version") != "e26_trajectory_binding_v1":
        raise RuntimeError("Unsupported trajectory binding report schema")
    if binding_payload.get("status") != "complete":
        raise RuntimeError("Trajectory binding report is not complete")
    _, bindings = load_binding_report(binding_path)
    sparse_path = Path(str(config["baseline_sparse_model"])).resolve(strict=True)
    sparse_signature, _ = directory_signature(sparse_path)
    reconstruction = pycolmap.Reconstruction(sparse_path)
    correspondences = build_pose_correspondences(reconstruction, bindings)
    sim3_config = config["sim3_initialization"]
    similarity, residuals, inliers = robust_similarity_ransac(
        correspondences["source_centers"],
        correspondences["target_centers"],
        maximum_error_m=float(sim3_config["maximum_error_m"]),
        minimum_inliers=int(sim3_config["minimum_inliers"]),
        iterations=int(sim3_config["ransac_iterations"]),
        seed=int(sim3_config["random_seed"]),
    )
    manifest = serialize_pose_prior_manifest(
        correspondences,
        similarity,
        residuals,
        inliers,
        position_sigma_m=float(config["lio_prior"]["position_sigma_m"]),
        rotation_sigma_deg=float(config["lio_prior"]["rotation_sigma_deg"]),
        binding_report_sha256=sha256_file(binding_path),
        sparse_model_signature=sparse_signature,
    )
    manifest["config_sha256"] = sha256_file(config_path)
    manifest["policy_sha256"] = policy_sha256
    manifest["image_manifest_sha256"] = binding_payload["image_manifest_sha256"]
    sim3 = {
        **similarity.to_json(),
        "config_sha256": manifest["config_sha256"],
        "policy_sha256": policy_sha256,
        "binding_report_sha256": manifest["binding_report_sha256"],
        "image_manifest_sha256": binding_payload["image_manifest_sha256"],
        "baseline_sparse_model": str(sparse_path),
        "sparse_model_signature": sparse_signature,
        "correspondence_count": len(residuals),
        "inlier_count": int(inliers.sum()),
    }
    staging = output.with_name(f"{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    write_json_new(staging / "pose_prior_manifest.json", manifest)
    write_json_new(staging / "sim3.json", sim3)
    sim3_file_sha256 = sha256_file(staging / "sim3.json")
    residual_report = {
        "schema_version": "e26_sim3_residual_report_v2",
        "sim3_file_sha256": sim3_file_sha256,
        "sim3_content_sha256": hashlib.sha256(
            json.dumps(sim3, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "time_bins": time_binned_residuals(
            correspondences["timestamps_ns"], residuals, bin_count=8
        ),
        "position_residual_m": manifest["position_residual_m"],
        "rotation_residual_deg": manifest["rotation_residual_deg"],
        "inlier_count": manifest["sim3_inlier_count"],
        "correspondence_count": manifest["registered_prior_count"],
    }
    write_json_new(staging / "residuals.json", residual_report)
    staging.replace(output)
    print(json.dumps({"sim3": sim3, "residuals": residual_report}, indent=2))


if __name__ == "__main__":
    main()
