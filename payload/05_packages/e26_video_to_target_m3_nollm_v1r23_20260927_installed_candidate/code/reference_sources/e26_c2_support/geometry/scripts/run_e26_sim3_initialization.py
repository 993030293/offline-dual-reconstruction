#!/usr/bin/env python
"""Estimate robust SfM-to-LIO Sim(3) and emit the pose-prior manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    output.mkdir(parents=True, exist_ok=True)

    binding_path = Path(str(config["trajectory_binding_report"]))
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
    write_json_new(output / "pose_prior_manifest.json", manifest)
    sim3 = {
        **similarity.to_json(),
        "config_sha256": manifest["config_sha256"],
        "binding_report_sha256": manifest["binding_report_sha256"],
        "sparse_model_signature": sparse_signature,
        "correspondence_count": len(residuals),
        "inlier_count": int(inliers.sum()),
    }
    write_json_new(output / "sim3.json", sim3)
    residual_report = {
        "schema_version": "e26_sim3_residual_report_v1",
        "sim3_sha256": hashlib.sha256(
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
    write_json_new(output / "residuals.json", residual_report)
    print(json.dumps({"sim3": sim3, "residuals": residual_report}, indent=2))


if __name__ == "__main__":
    main()

