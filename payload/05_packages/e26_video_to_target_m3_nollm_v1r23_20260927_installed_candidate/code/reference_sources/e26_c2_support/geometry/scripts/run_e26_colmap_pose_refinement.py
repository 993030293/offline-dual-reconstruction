#!/usr/bin/env python
"""Run the E26 pose-graph plus pycolmap/Ceres sparse refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from lidar_camera_recon.e26.input_freeze import directory_signature
from lidar_camera_recon.e26.receipts import write_json_new
from lidar_camera_recon.e26.trajectory_io import sha256_file
from lidar_camera_recon.e26_colmap_pose_refinement import (
    PoseGraphOptions,
    apply_world_camera_poses,
    apply_pose_graph_parameters,
    build_colmap_pose_graph_edges,
    build_lio_motion_edges,
    evaluate_edge_type,
    evaluate_withheld_edges,
    extract_world_camera_poses,
    pose_graph_raw_objective,
    refine_pose_graph,
    run_pycolmap_joint_ba,
    run_pycolmap_points_only_ba,
)
from lidar_camera_recon.e26_pose_prior_builder import (
    build_pose_correspondences,
    load_binding_report,
)
from lidar_camera_recon.e26_sim3_initialization import SimilarityTransform


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value.name)
    return value


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def pose_metrics(
    centers_m: np.ndarray,
    rotations_world_camera: np.ndarray,
    prior_centers_m: np.ndarray,
    prior_rotations_world_camera: np.ndarray,
) -> dict[str, Any]:
    position = np.linalg.norm(centers_m - prior_centers_m, axis=1)
    relative = np.transpose(prior_rotations_world_camera, (0, 2, 1)) @ rotations_world_camera
    rotation = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    return {
        "position_residual_m": summary(position),
        "rotation_residual_deg": summary(rotation),
        "first_last_camera_center_distance_m": float(
            np.linalg.norm(centers_m[-1] - centers_m[0])
        ),
        "first_last_rotation_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    rotations_world_camera[0].T @ rotations_world_camera[-1]
                ).magnitude()
            )
        ),
    }


def load_similarity(path: Path) -> SimilarityTransform:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e26_sim3_v1":
        raise ValueError("unsupported Sim(3) schema")
    return SimilarityTransform(
        scale=float(payload["scale"]),
        rotation=Rotation.from_quat(payload["rotation_xyzw"]).as_matrix(),
        translation=np.asarray(payload["translation_m"], dtype=np.float64),
    )


def pose_graph_options(config: dict[str, Any]) -> PoseGraphOptions:
    graph = config["pose_graph"]
    prior = config["lio_prior"]
    gate = config["pose_correction_gate"]
    return PoseGraphOptions(
        lio_position_sigma_m=float(prior["position_sigma_m"]),
        lio_rotation_sigma_deg=float(prior["rotation_sigma_deg"]),
        lio_position_weight=float(graph["lio_position_weight"]),
        lio_rotation_weight=float(graph["lio_rotation_weight"]),
        local_translation_sigma_m=float(graph["local_translation_sigma_m"]),
        local_rotation_sigma_deg=float(graph["local_rotation_sigma_deg"]),
        lio_local_translation_sigma_m=float(
            graph.get("lio_local_translation_sigma_m", 0.05)
        ),
        lio_local_rotation_sigma_deg=float(
            graph.get("lio_local_rotation_sigma_deg", 1.2)
        ),
        anchor_translation_sigma_m=float(
            graph.get("anchor_translation_sigma_m", 0.002)
        ),
        anchor_rotation_sigma_deg=float(
            graph.get("anchor_rotation_sigma_deg", 0.1)
        ),
        loop_translation_sigma_m=float(graph["loop_translation_sigma_m"]),
        loop_rotation_sigma_deg=float(graph["loop_rotation_sigma_deg"]),
        smooth_translation_sigma_m=float(graph["smooth_translation_sigma_m"]),
        smooth_rotation_sigma_deg=float(graph["smooth_rotation_sigma_deg"]),
        maximum_translation_correction_m=float(gate["maximum_translation_m"]),
        maximum_rotation_correction_deg=float(gate["maximum_rotation_deg"]),
        robust_loss=str(graph["robust_loss"]),
        maximum_evaluations=int(graph["maximum_evaluations"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sim3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-pose-graph-checkpoint", type=Path)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "e26_colmap_pose_refinement_config_v1":
        raise ValueError("unsupported pose refinement config schema")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_sparse = Path(str(config["baseline_sparse_model"])).resolve(strict=True)
    source_signature_before, _ = directory_signature(source_sparse)
    binding_path = Path(str(config["trajectory_binding_report"]))
    _, bindings = load_binding_report(binding_path)
    reconstruction = pycolmap.Reconstruction(source_sparse)
    similarity = load_similarity(arguments.sim3)
    reconstruction.transform(
        pycolmap.Sim3d(
            similarity.scale,
            pycolmap.Rotation3d(similarity.rotation),
            similarity.translation,
        )
    )
    correspondences = build_pose_correspondences(reconstruction, bindings)
    rows = correspondences["rows"]
    ordered_ids = [int(item["image_id"]) for item in rows]
    initial_centers = np.stack([item["sfm_camera_center"] for item in rows])
    initial_rotations = np.stack([item["sfm_R_world_camera"] for item in rows])
    prior_centers = np.stack([item["lio_camera_center_m"] for item in rows])
    prior_rotations = np.stack([item["lio_R_world_camera"] for item in rows])
    timestamps = correspondences["timestamps_ns"]
    loop = config["loop_constraints"]
    edges = build_colmap_pose_graph_edges(
        reconstruction,
        ordered_ids,
        timestamps,
        initial_centers,
        initial_rotations,
        minimum_loop_separation_s=float(loop["minimum_time_separation_s"]),
        minimum_loop_track_support=int(loop["minimum_verified_matches"]),
    )
    motion_prior = config.get("lio_motion_prior", {})
    if motion_prior.get("enabled", False):
        strides = tuple(int(value) for value in motion_prior.get("strides", [1, 2]))
        edges.extend(
            build_lio_motion_edges(
                prior_centers,
                prior_rotations,
                strides=strides,
            )
        )
    options = pose_graph_options(config)
    checkpoint_path = output / "pose_graph_candidate.npz"
    checkpoint_metadata = output / "pose_graph_checkpoint.json"
    checkpoint_identity = {
        "schema_version": "e26_pose_graph_checkpoint_v1",
        "config_sha256": sha256_file(config_path),
        "sim3_sha256": sha256_file(arguments.sim3.resolve(strict=True)),
        "trajectory_binding_sha256": sha256_file(binding_path),
        "source_sparse_signature": source_signature_before,
        "initial_pose_graph_checkpoint_sha256": (
            sha256_file(arguments.initial_pose_graph_checkpoint.resolve(strict=True))
            if arguments.initial_pose_graph_checkpoint is not None
            else None
        ),
    }
    if checkpoint_path.exists() and checkpoint_metadata.exists():
        saved_identity = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
        if saved_identity != checkpoint_identity:
            raise ValueError("pose graph checkpoint identity mismatch")
        saved = np.load(checkpoint_path)
        graph = {
            "success": bool(saved["success"]),
            "status": int(saved["status"]),
            "message": str(saved["message"]),
            "function_evaluations": int(saved["function_evaluations"]),
            "initial_cost": float(saved["initial_cost"]),
            "final_cost": float(saved["final_cost"]),
            "centers_m": saved["centers_m"],
            "rotations_world_camera": saved["rotations_world_camera"],
            "translation_corrections_m": saved["translation_corrections_m"],
            "rotation_corrections_deg": saved["rotation_corrections_deg"],
            "parameters": saved["parameters"],
            "cache_hit": True,
        }
    else:
        initial_parameters = None
        if arguments.initial_pose_graph_checkpoint is not None:
            prior_checkpoint = np.load(
                arguments.initial_pose_graph_checkpoint.resolve(strict=True)
            )
            initial_parameters = prior_checkpoint["parameters"]
        if config["pose_graph"].get("use_frozen_proposal", False):
            if initial_parameters is None:
                raise ValueError("frozen pose-graph proposal requires a checkpoint")
            candidate_centers, candidate_rotations = apply_pose_graph_parameters(
                initial_parameters, initial_centers, initial_rotations
            )
            graph = {
                "success": False,
                "status": 0,
                "message": "frozen_nonconverged_proposal_for_truncated_step_gate",
                "function_evaluations": 0,
                "optimality": float("nan"),
                "initial_cost": pose_graph_raw_objective(
                    np.zeros_like(initial_parameters),
                    initial_centers,
                    initial_rotations,
                    prior_centers,
                    prior_rotations,
                    edges,
                    options,
                ),
                "final_cost": pose_graph_raw_objective(
                    initial_parameters,
                    initial_centers,
                    initial_rotations,
                    prior_centers,
                    prior_rotations,
                    edges,
                    options,
                ),
                "final_raw_squared_cost": pose_graph_raw_objective(
                    initial_parameters,
                    initial_centers,
                    initial_rotations,
                    prior_centers,
                    prior_rotations,
                    edges,
                    options,
                ),
                "centers_m": candidate_centers,
                "rotations_world_camera": candidate_rotations,
                "translation_corrections_m": initial_parameters[:, 3:],
                "rotation_corrections_deg": np.rad2deg(
                    np.linalg.norm(initial_parameters[:, :3], axis=1)
                ),
                "parameters": initial_parameters,
                "frozen_proposal_sha256": sha256_file(
                    arguments.initial_pose_graph_checkpoint.resolve(strict=True)
                ),
            }
        else:
            graph = refine_pose_graph(
                initial_centers,
                initial_rotations,
                prior_centers,
                prior_rotations,
                edges,
                options,
                initial_parameters=initial_parameters,
            )
        np.savez_compressed(
            checkpoint_path,
            **{
                key: value
                for key, value in graph.items()
                if key
                in {
                    "success",
                    "status",
                    "message",
                    "function_evaluations",
                    "optimality",
                    "initial_cost",
                    "final_cost",
                    "final_raw_squared_cost",
                    "centers_m",
                    "rotations_world_camera",
                    "translation_corrections_m",
                    "rotation_corrections_deg",
                    "parameters",
                }
            },
        )
        checkpoint_metadata.write_text(
            json.dumps(checkpoint_identity, indent=2), encoding="utf-8"
        )
        graph["cache_hit"] = False
    reprojection_before = float(reconstruction.compute_mean_reprojection_error())
    truncated_gate = config["pose_graph"].get("truncated_step_gate", {})
    if truncated_gate.get("enabled", False) and not graph["success"]:
        fractions = np.linspace(
            1.0,
            float(truncated_gate.get("minimum_fraction", 0.025)),
            int(truncated_gate.get("candidate_count", 40)),
        )
        full_parameters = np.asarray(graph["parameters"], dtype=np.float64)
        lio_motion_baseline = evaluate_edge_type(
            initial_centers, initial_rotations, edges, "lio_local"
        )
        maximum_raw_reprojection = reprojection_before * float(
            truncated_gate["maximum_raw_reprojection_error_ratio"]
        )
        curve: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None
        for fraction in fractions:
            parameters = full_parameters * float(fraction)
            centers, rotations = apply_pose_graph_parameters(
                parameters, initial_centers, initial_rotations
            )
            apply_world_camera_poses(reconstruction, ordered_ids, centers, rotations)
            reconstruction.update_point_3d_errors()
            reprojection = float(reconstruction.compute_mean_reprojection_error())
            motion = evaluate_edge_type(centers, rotations, edges, "lio_local")
            objective = pose_graph_raw_objective(
                parameters,
                initial_centers,
                initial_rotations,
                prior_centers,
                prior_rotations,
                edges,
                options,
            )
            item = {
                "fraction": float(fraction),
                "raw_reprojection_error_px": reprojection,
                "pose_graph_raw_objective": objective,
                "lio_local_motion": motion,
            }
            curve.append(item)
            translation_improved = (
                motion["translation_residual_m"]["p50"]
                < lio_motion_baseline["translation_residual_m"]["p50"]
            )
            rotation_improved = (
                motion["rotation_residual_deg"]["p50"]
                < lio_motion_baseline["rotation_residual_deg"]["p50"]
            )
            if (
                reprojection <= maximum_raw_reprojection
                and objective < graph["initial_cost"]
                and translation_improved
                and rotation_improved
            ):
                accepted = {**item, "parameters": parameters}
                break
        if accepted is not None:
            graph["parameters"] = accepted.pop("parameters")
            graph["centers_m"], graph["rotations_world_camera"] = (
                apply_pose_graph_parameters(
                    graph["parameters"], initial_centers, initial_rotations
                )
            )
            graph["translation_corrections_m"] = graph["parameters"][:, 3:]
            graph["rotation_corrections_deg"] = np.rad2deg(
                np.linalg.norm(graph["parameters"][:, :3], axis=1)
            )
        graph["truncated_step_gate"] = {
            "accepted": accepted is not None,
            "maximum_raw_reprojection_error_px": maximum_raw_reprojection,
            "accepted_candidate": accepted,
            "curve": curve,
            "scientific_boundary": (
                "A nonconverged full proposal is used only as a descent direction. "
                "The accepted fraction must improve local LIO motion and remain "
                "inside a stricter pre-BA visual reprojection gate."
            ),
        }
    apply_world_camera_poses(
        reconstruction,
        ordered_ids,
        graph["centers_m"],
        graph["rotations_world_camera"],
    )
    reprojection_after_pose_graph = float(
        reconstruction.compute_mean_reprojection_error()
    )
    ba_mode = str(
        config["ceres_bundle_adjustment"].get(
            "mode", "position_prior_joint_then_rotation_restore"
        )
    )
    if ba_mode == "fixed_camera_points_only":
        points_only = run_pycolmap_points_only_ba(
            reconstruction,
            ordered_ids,
            maximum_iterations=int(
                config["ceres_bundle_adjustment"]["maximum_iterations"]
            ),
        )
        ba = {
            "backend": "pycolmap_ceres_points_only",
            "final_points_only": jsonable(points_only),
            "alternating_schedule": [
                "joint_visual_loop_and_local_lio_motion_pose_graph",
                "pycolmap_ceres_points_only_with_final_cameras_fixed",
            ],
        }
        reprojection_after_joint = None
    elif ba_mode == "position_prior_joint_then_rotation_restore":
        ba = run_pycolmap_joint_ba(
            reconstruction,
            ordered_ids,
            prior_centers,
            position_sigma_m=float(config["lio_prior"]["position_sigma_m"]),
            maximum_iterations=int(
                config["ceres_bundle_adjustment"]["maximum_iterations"]
            ),
        )
        joint_centers, _ = extract_world_camera_poses(reconstruction, ordered_ids)
        reconstruction.update_point_3d_errors()
        reprojection_after_joint = float(
            reconstruction.compute_mean_reprojection_error()
        )
        apply_world_camera_poses(
            reconstruction,
            ordered_ids,
            joint_centers,
            graph["rotations_world_camera"],
        )
        points_only = run_pycolmap_points_only_ba(
            reconstruction,
            ordered_ids,
            maximum_iterations=int(
                config["ceres_bundle_adjustment"]["maximum_iterations"]
            ),
        )
        ba["final_points_only"] = jsonable(points_only)
        ba["alternating_schedule"] = [
            "rotation_lio_loop_pose_graph",
            "pycolmap_ceres_position_prior_joint_camera_point_ba",
            "restore_pose_graph_rotations_keep_ceres_centers",
            "pycolmap_ceres_points_only_with_final_cameras_fixed",
        ]
    else:
        raise ValueError(f"unsupported Ceres bundle-adjustment mode: {ba_mode}")
    final_centers, final_rotations = extract_world_camera_poses(
        reconstruction, ordered_ids
    )
    reprojection_after = float(reconstruction.compute_mean_reprojection_error())
    initial_metrics = pose_metrics(
        initial_centers, initial_rotations, prior_centers, prior_rotations
    )
    graph_metrics = pose_metrics(
        graph["centers_m"],
        graph["rotations_world_camera"],
        prior_centers,
        prior_rotations,
    )
    final_metrics = pose_metrics(
        final_centers, final_rotations, prior_centers, prior_rotations
    )
    lio_motion_before = evaluate_edge_type(
        initial_centers, initial_rotations, edges, "lio_local"
    )
    lio_motion_after = evaluate_edge_type(
        final_centers, final_rotations, edges, "lio_local"
    )
    correction_translation = np.linalg.norm(final_centers - initial_centers, axis=1)
    correction_rotation = np.rad2deg(
        Rotation.from_matrix(
            np.transpose(initial_rotations, (0, 2, 1)) @ final_rotations
        ).magnitude()
    )
    gate = config["pose_correction_gate"]
    failures: list[str] = []
    truncated_step_accepted = bool(
        graph.get("truncated_step_gate", {}).get("accepted", False)
    )
    if not graph["success"] and not truncated_step_accepted:
        failures.append("pose_graph_solver_failed")
    if motion_prior.get("enabled", False):
        if (
            lio_motion_after["translation_residual_m"]["p50"]
            >= lio_motion_before["translation_residual_m"]["p50"]
        ):
            failures.append("lio_local_translation_p50_not_improved")
        if (
            lio_motion_after["rotation_residual_deg"]["p50"]
            >= lio_motion_before["rotation_residual_deg"]["p50"]
        ):
            failures.append("lio_local_rotation_p50_not_improved")
    elif (
        final_metrics["position_residual_m"]["p50"]
        >= initial_metrics["position_residual_m"]["p50"]
    ):
        failures.append("lio_position_p50_not_improved")
    if summary(correction_translation)["max"] > float(gate["maximum_translation_m"]):
        failures.append("maximum_translation_correction_exceeded")
    if summary(correction_rotation)["max"] > float(gate["maximum_rotation_deg"]):
        failures.append("maximum_rotation_correction_exceeded")
    if reprojection_after > reprojection_before * float(
        config["ceres_bundle_adjustment"]["maximum_reprojection_error_ratio"]
    ):
        failures.append("reprojection_error_degraded")
    source_signature_after, _ = directory_signature(source_sparse)
    if source_signature_after != source_signature_before:
        failures.append("source_sparse_hash_changed")

    candidate_name = "corrected_sparse" if not failures else "rejected_candidate_sparse"
    candidate_path = output / candidate_name
    candidate_path.mkdir(parents=False, exist_ok=False)
    reconstruction.write(candidate_path)
    reopened = pycolmap.Reconstruction(candidate_path)
    if reopened.num_reg_images() != len(ordered_ids):
        failures.append("corrected_sparse_reopen_image_count_mismatch")
    status = "complete_gate_passed" if not failures else "abstained_gate_failed"
    edge_counts = {
        kind: sum(edge.edge_type == kind for edge in edges)
        for kind in ("local", "lio_local", "loop", "withheld_loop")
    }
    report = {
        "schema_version": "e26_colmap_pose_refinement_receipt_v1",
        "status": status,
        "failure_reasons": failures,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "sim3_path": str(arguments.sim3.resolve(strict=True)),
        "sim3_sha256": sha256_file(arguments.sim3.resolve(strict=True)),
        "trajectory_binding_sha256": sha256_file(binding_path),
        "source_sparse_model": str(source_sparse),
        "source_sparse_signature_before": source_signature_before,
        "source_sparse_signature_after": source_signature_after,
        "source_sparse_unchanged": source_signature_before == source_signature_after,
        "registered_image_count": len(ordered_ids),
        "sparse_point_count": int(reconstruction.num_points3D()),
        "edge_counts": edge_counts,
        "pose_graph": {
            key: jsonable(value)
            for key, value in graph.items()
            if key not in {"centers_m", "rotations_world_camera", "parameters"}
        },
        "withheld_edge_audit": evaluate_withheld_edges(
            final_centers, final_rotations, edges
        ),
        "bundle_adjustment": jsonable(ba),
        "reprojection_error_px": {
            "before_pose_graph": reprojection_before,
            "after_pose_graph_before_ba": reprojection_after_pose_graph,
            "after_joint_ba_before_rotation_prior_restore": reprojection_after_joint,
            "after_joint_ba": reprojection_after,
        },
        "lio_pose_residual_before": initial_metrics,
        "lio_pose_residual_after_pose_graph": graph_metrics,
        "lio_pose_residual_after_joint_ba": final_metrics,
        "lio_local_motion_residual_before": lio_motion_before,
        "lio_local_motion_residual_after": lio_motion_after,
        "final_pose_correction": {
            "translation_m": summary(correction_translation),
            "rotation_deg": summary(correction_rotation),
        },
        "corrected_sparse_path": str(candidate_path),
        "corrected_sparse_reopen_ok": reopened.num_reg_images() == len(ordered_ids),
        "scientific_boundary": (
            "This is an offline corrected sparse model. Gate passage permits "
            "a quick dense A/B; it does not prove dense or metric accuracy."
        ),
    }
    write_json_new(output / "pose_refinement_receipt.json", report)
    write_json_new(
        output / "residuals_before_after.json",
        {
            "schema_version": "e26_pose_residuals_before_after_v1",
            "status": status,
            "before": initial_metrics,
            "after_pose_graph": graph_metrics,
            "after_joint_ba": final_metrics,
            "lio_local_motion_before": lio_motion_before,
            "lio_local_motion_after": lio_motion_after,
            "reprojection_error_px": report["reprojection_error_px"],
            "correction": report["final_pose_correction"],
            "withheld_edge_audit": report["withheld_edge_audit"],
        },
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
