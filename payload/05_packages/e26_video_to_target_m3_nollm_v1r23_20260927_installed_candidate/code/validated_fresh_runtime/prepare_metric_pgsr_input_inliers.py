"""Prepare lineage-bound PGSR input with independent camera geometry gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pycolmap

from lidar_camera_recon.e26.input_freeze import directory_signature
from metric_admission import camera_geometry_metrics
from metric_manifest_lineage import verify_metric_manifest_lineage
from point_track_admission import choose_track_length


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def filter_images_text(
    source: Path,
    target: Path,
    excluded_names: set[str],
    retained_point_ids: set[int],
) -> tuple[int, list[int]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    output = []
    retained = 0
    excluded_ids = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            output.append(line)
            index += 1
            continue
        fields = line.split()
        if len(fields) < 10:
            raise RuntimeError("Malformed COLMAP images.txt record")
        if index + 1 >= len(lines):
            raise RuntimeError("Missing COLMAP points2D line")
        image_id = int(fields[0])
        name = fields[9]
        if name in excluded_names:
            excluded_ids.append(image_id)
        else:
            points = lines[index + 1].split()
            if len(points) % 3 != 0:
                raise RuntimeError("Malformed COLMAP points2D record")
            for offset in range(2, len(points), 3):
                point_id = int(points[offset])
                if point_id >= 0 and point_id not in retained_point_ids:
                    points[offset] = "-1"
            output.extend([line, " ".join(points)])
            retained += 1
        index += 2
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return retained, excluded_ids


def filter_points_text(
    source: Path,
    target: Path,
    minimum: np.ndarray,
    maximum: np.ndarray,
    excluded_image_ids: set[int],
    maximum_error_px: float,
    minimum_track_length: int,
) -> tuple[int, int, set[int]]:
    output = []
    retained = 0
    rejected = 0
    retained_ids: set[int] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        xyz = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
        error = float(fields[7])
        track = fields[8:]
        if len(track) % 2 != 0:
            raise RuntimeError("Malformed COLMAP point track")
        kept_track = []
        for offset in range(0, len(track), 2):
            if int(track[offset]) not in excluded_image_ids:
                kept_track.extend(track[offset : offset + 2])
        track_length = len(kept_track) // 2
        keep = (
            bool(np.isfinite(xyz).all())
            and bool(np.all(xyz >= minimum))
            and bool(np.all(xyz <= maximum))
            and error <= maximum_error_px
            and track_length >= minimum_track_length
        )
        if keep:
            point_id = int(fields[0])
            output.append(" ".join(fields[:8] + kept_track))
            retained += 1
            retained_ids.add(point_id)
        else:
            rejected += 1
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return retained, rejected, retained_ids


def verify_model_identity(
    baseline: pycolmap.Reconstruction,
    dense: pycolmap.Reconstruction,
    policy: dict,
) -> dict:
    baseline_images = {image.name: image for image in baseline.images.values()}
    dense_images = {image.name: image for image in dense.images.values()}
    if set(baseline_images) != set(dense_images):
        raise RuntimeError("Dense and Sim3 baseline models have different image identities")
    if set(baseline.points3D) != set(dense.points3D):
        raise RuntimeError("Dense and Sim3 baseline models have different point identities")
    center_deltas = []
    rotation_deltas = []
    for name in sorted(baseline_images):
        first = baseline_images[name]
        second = dense_images[name]
        center_deltas.append(
            float(np.linalg.norm(first.projection_center() - second.projection_center()))
        )
        rotation_deltas.append(
            float(
                np.max(
                    np.abs(
                        first.cam_from_world().rotation.matrix()
                        - second.cam_from_world().rotation.matrix()
                    )
                )
            )
        )
    point_deltas = [
        float(np.linalg.norm(baseline.points3D[key].xyz - dense.points3D[key].xyz))
        for key in baseline.points3D
    ]
    maximum_center_delta = max(center_deltas, default=0.0)
    maximum_rotation_delta = max(rotation_deltas, default=0.0)
    maximum_point_delta = max(point_deltas, default=0.0)
    if maximum_center_delta > float(policy["maximum_camera_center_delta"]):
        raise RuntimeError("Dense camera centers do not match the Sim3 baseline model")
    if maximum_rotation_delta > float(policy["maximum_rotation_matrix_delta"]):
        raise RuntimeError("Dense camera rotations do not match the Sim3 baseline model")
    if maximum_point_delta > float(policy["maximum_camera_center_delta"]):
        raise RuntimeError("Dense sparse points do not match the Sim3 baseline model")
    return {
        "image_count": len(baseline_images),
        "point_count": len(baseline.points3D),
        "maximum_camera_center_delta": maximum_center_delta,
        "maximum_rotation_matrix_delta": maximum_rotation_delta,
        "maximum_point_delta": maximum_point_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-workspace", required=True, type=Path)
    parser.add_argument("--registered-manifest", required=True, type=Path)
    parser.add_argument("--metric-bound-manifest", required=True, type=Path)
    parser.add_argument("--metric-time-support-receipt", required=True, type=Path)
    parser.add_argument("--baseline-sparse-model", required=True, type=Path)
    parser.add_argument("--sim3", required=True, type=Path)
    parser.add_argument("--sim3-residuals", required=True, type=Path)
    parser.add_argument("--pose-priors", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dense = args.dense_workspace.resolve(strict=True)
    output = args.output.resolve()
    stage_root = output.parent
    if stage_root.exists():
        raise FileExistsError(f"PGSR preparation stage already exists: {stage_root}")
    if output.name != "input":
        raise ValueError("PGSR output must be the input directory inside a fresh stage root")
    registered_manifest_path = args.registered_manifest.resolve(strict=True)
    metric_bound_manifest_path = args.metric_bound_manifest.resolve(strict=True)
    metric_time_support_receipt_path = args.metric_time_support_receipt.resolve(strict=True)
    baseline_sparse_path = args.baseline_sparse_model.resolve(strict=True)
    sim3_path = args.sim3.resolve(strict=True)
    residuals_path = args.sim3_residuals.resolve(strict=True)
    pose_priors_path = args.pose_priors.resolve(strict=True)
    policy_path = args.policy.resolve(strict=True)
    policy = read_json(policy_path)
    if policy.get("schema_version") != "e26_metric_alignment_policy_v1":
        raise RuntimeError("Unsupported metric alignment policy schema")
    policy_sha256 = sha256_file(policy_path)
    pose_priors = read_json(pose_priors_path)
    sim3 = read_json(sim3_path)
    residuals_report = read_json(residuals_path)
    if sim3.get("schema_version") != "e26_sim3_v1":
        raise RuntimeError("Unsupported Sim3 schema")
    if pose_priors.get("schema_version") != "e26_colmap_lio_pose_prior_manifest_v1":
        raise RuntimeError("Unsupported pose-prior schema")
    if residuals_report.get("schema_version") != "e26_sim3_residual_report_v2":
        raise RuntimeError("Unsupported Sim3 residual report schema")
    if residuals_report.get("sim3_file_sha256") != sha256_file(sim3_path):
        raise RuntimeError("Sim3 file identity does not match its residual report")
    if sim3.get("policy_sha256") != policy_sha256 or pose_priors.get("policy_sha256") != policy_sha256:
        raise RuntimeError("Sim3 artifacts do not match the selected metric alignment policy")
    registered, metric_bound = verify_metric_manifest_lineage(
        registered_manifest_path,
        metric_bound_manifest_path,
        metric_time_support_receipt_path,
        sim3.get("image_manifest_sha256"),
        policy_sha256,
    )
    for key in ("config_sha256", "binding_report_sha256", "sparse_model_signature"):
        if sim3.get(key) != pose_priors.get(key):
            raise RuntimeError(f"Sim3 and pose-prior lineage differ for {key}")
    similarity_keys = (
        "schema_version",
        "transform_name",
        "source_frame",
        "target_frame",
        "translation_unit",
        "scale",
        "rotation_xyzw",
        "translation_m",
    )
    if {key: sim3[key] for key in similarity_keys} != pose_priors.get("sim3"):
        raise RuntimeError("Pose-prior manifest embeds a different Sim3 transform")
    baseline_signature, _ = directory_signature(baseline_sparse_path)
    if baseline_signature != sim3.get("sparse_model_signature"):
        raise RuntimeError("Baseline sparse model does not match the model used to estimate Sim3")
    if sim3.get("correspondence_count") != pose_priors.get("registered_prior_count"):
        raise RuntimeError("Sim3 correspondence count does not match pose priors")
    if sim3.get("inlier_count") != pose_priors.get("sim3_inlier_count"):
        raise RuntimeError("Sim3 inlier count does not match pose priors")
    registered_names = [row["image_name"] for row in registered]
    if len(registered_names) != len(set(registered_names)):
        raise RuntimeError("Registered image manifest contains duplicate image names")
    prior_rows = pose_priors["images"]
    prior_names = [row["image_name"] for row in prior_rows]
    if len(prior_names) != len(set(prior_names)):
        raise RuntimeError("Pose-prior manifest contains duplicate image names")
    unknown_prior_names = set(prior_names) - {row["image_name"] for row in metric_bound}
    if unknown_prior_names:
        raise RuntimeError(f"Pose priors reference metric-unsupported images: {sorted(unknown_prior_names)}")

    inlier_rows = [row for row in prior_rows if row["sim3_ransac_inlier"]]
    outlier_rows = [row for row in prior_rows if not row["sim3_ransac_inlier"]]
    sim3_inlier_names = {row["image_name"] for row in inlier_rows}
    unsupported_names = set(registered_names) - set(prior_names)
    reconstruction = pycolmap.Reconstruction(dense / "sparse")
    baseline_reconstruction = pycolmap.Reconstruction(baseline_sparse_path)
    model_identity = verify_model_identity(
        baseline_reconstruction, reconstruction, policy["model_identity"]
    )
    geometry_rows = [
        camera_geometry_metrics(reconstruction, image, policy["camera_geometry_admission"])
        for image in reconstruction.images.values()
        if image.name in sim3_inlier_names
    ]
    geometry_by_name = {row["image_name"]: row for row in geometry_rows}
    geometry_pass_names = {row["image_name"] for row in geometry_rows if row["passed"]}
    retained_name_set = sim3_inlier_names & geometry_pass_names
    geometry_rejected_names = sim3_inlier_names - geometry_pass_names
    excluded_names = set(registered_names) - retained_name_set
    admission_policy = policy["camera_geometry_admission"]
    retained_fraction = (
        len(retained_name_set) / len(sim3_inlier_names) if sim3_inlier_names else 0.0
    )
    if (
        len(retained_name_set) < int(admission_policy["minimum_retained_cameras"])
        or retained_fraction
        < float(admission_policy["minimum_retained_fraction_of_sim3_inliers"])
    ):
        raise RuntimeError(
            "Independent camera geometry admission failed: "
            f"retained={len(retained_name_set)}, Sim3 inliers={len(sim3_inlier_names)}, "
            f"fraction={retained_fraction:.6f}"
        )
    transform = pycolmap.Sim3d(
        float(sim3["scale"]),
        pycolmap.Rotation3d(np.asarray(sim3["rotation_xyzw"], dtype=np.float64)),
        np.asarray(sim3["translation_m"], dtype=np.float64),
    )
    reconstruction.transform(transform)
    retained_centers = np.asarray(
        [
            image.projection_center()
            for image in reconstruction.images.values()
            if image.has_pose and image.name not in excluded_names
        ],
        dtype=np.float64,
    )
    if len(retained_centers) != len(retained_name_set):
        raise RuntimeError(
            "Retained reconstruction camera count does not match the fresh Sim3 inlier count: "
            f"{len(retained_centers)} versus {len(retained_name_set)}"
        )
    point_policy = policy["point_filter"]
    point_margin_m = float(point_policy["camera_aabb_margin_m"])
    minimum = retained_centers.min(axis=0) - point_margin_m
    maximum = retained_centers.max(axis=0) + point_margin_m
    if float(np.max(np.linalg.norm(retained_centers, axis=1))) > float(
        point_policy["maximum_camera_radius_m"]
    ):
        raise RuntimeError("Retained camera extent is not physically plausible")

    staging_root = stage_root.with_name(f"{stage_root.name}.staging-{uuid.uuid4().hex}")
    staging_output = staging_root / output.name
    text_model = staging_root / "_metric_text_model"
    (staging_output / "images").mkdir(parents=True)
    (staging_output / "sparse").mkdir()
    text_model.mkdir()
    try:
        reconstruction.write_text(text_model)
        shutil.copy2(text_model / "cameras.txt", staging_output / "sparse" / "cameras.txt")
        image_ids_by_name = {
            image.name: int(image.image_id) for image in reconstruction.images.values()
        }
        excluded_ids_set = {image_ids_by_name[name] for name in excluded_names}
        preferred_track_length = int(point_policy["minimum_point_track_length"])
        fallback_track_length = int(point_policy["fallback_minimum_point_track_length"])
        if fallback_track_length < 2 or fallback_track_length >= preferred_track_length:
            raise ValueError("Point-track fallback must be weaker than the preferred support")
        minimum_points = int(point_policy["minimum_retained_sparse_points"])
        point_filter_attempts = []
        selected_track_length = None
        for track_length in (preferred_track_length, fallback_track_length):
            retained_point_count, rejected_point_count, retained_point_ids = filter_points_text(
                text_model / "points3D.txt",
                staging_output / "sparse" / "points3D.txt",
                minimum,
                maximum,
                excluded_ids_set,
                float(point_policy["maximum_point_reprojection_error_px"]),
                track_length,
            )
            point_filter_attempts.append(
                {"minimum_track_length": track_length, "retained_points": retained_point_count,
                 "rejected_points": rejected_point_count}
            )
            selected_track_length = choose_track_length(point_filter_attempts, minimum_points)
            if selected_track_length is not None:
                break
        retained_image_count, excluded_ids = filter_images_text(
            text_model / "images.txt",
            staging_output / "sparse" / "images.txt",
            excluded_names,
            retained_point_ids,
        )
        if (
            retained_image_count != len(retained_name_set)
            or selected_track_length is None
        ):
            raise RuntimeError(
                "Filtered PGSR model did not satisfy camera or point gates: "
                f"retained_cameras={retained_image_count}/{len(retained_name_set)}, "
                f"minimum_points={minimum_points}, attempts={point_filter_attempts}"
            )

        retained_names = [
            row["image_name"] for row in registered if row["image_name"] in retained_name_set
        ]
        image_rows = []
        for name in retained_names:
            source = dense / "images" / name
            target = staging_output / "images" / name
            shutil.copy2(source, target)
            digest = sha256_file(target)
            if digest != sha256_file(source):
                raise RuntimeError(f"Copied image mismatch: {name}")
            image_rows.append(
                {"name": name, "bytes": target.stat().st_size, "sha256": digest}
            )

        inlier_residuals = [
            float(row["sim3_position_residual_m"])
            for row in inlier_rows
            if row["image_name"] in retained_name_set
        ]
        sparse_rows = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted((staging_output / "sparse").iterdir())
            if path.is_file()
        ]
        receipt = {
        "schema_version": "e26_mask_free_metric_pgsr_inlier_input_v3",
        "status": "admitted_for_mask_free_pgsr",
        "language_model_used": False,
        "sam2_used": False,
        "object_mask_used": False,
        "source_dense_workspace": str(dense),
        "source_baseline_sparse_model": str(baseline_sparse_path),
        "source_baseline_sparse_signature": baseline_signature,
        "source_sim3": str(sim3_path),
        "source_sim3_sha256": sha256_file(sim3_path),
        "source_sim3_residuals": str(residuals_path),
        "source_sim3_residuals_sha256": sha256_file(residuals_path),
        "source_pose_priors": str(pose_priors_path),
        "source_pose_priors_sha256": sha256_file(pose_priors_path),
        "source_registered_manifest": str(registered_manifest_path),
        "source_registered_manifest_sha256": sha256_file(registered_manifest_path),
        "source_metric_bound_manifest": str(metric_bound_manifest_path),
        "source_metric_bound_manifest_sha256": sha256_file(metric_bound_manifest_path),
        "source_metric_time_support_receipt": str(metric_time_support_receipt_path),
        "source_metric_time_support_receipt_sha256": sha256_file(metric_time_support_receipt_path),
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "model_identity": model_identity,
        "registered_camera_count": retained_image_count,
        "metric_ransac_inlier_prior_count": len(inlier_rows),
        "independent_geometry_admitted_camera_count": len(retained_name_set),
        "independent_geometry_admitted_fraction_of_sim3_inliers": retained_fraction,
        "camera_geometry_policy": admission_policy,
        "excluded_geometry_unsupported_cameras": [
            geometry_by_name[name] for name in sorted(geometry_rejected_names)
        ],
        "excluded_metric_unsupported_cameras": sorted(unsupported_names),
        "excluded_sim3_outlier_cameras": [
            {
                "image_name": row["image_name"],
                "image_id": int(row["image_id"]),
                "position_residual_m": float(row["sim3_position_residual_m"]),
            }
            for row in outlier_rows
        ],
        "excluded_colmap_image_ids": excluded_ids,
        "metric_position_residual_m": {
            "p50": float(np.percentile(inlier_residuals, 50)),
            "p95": float(np.percentile(inlier_residuals, 95)),
            "maximum": float(np.max(inlier_residuals)),
        },
        "retained_camera_bounds_m": {"minimum": retained_centers.min(axis=0).tolist(), "maximum": retained_centers.max(axis=0).tolist()},
        "point_filter_bounds_m": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "point_filter_margin_m": point_margin_m,
        "point_filter_policy": point_policy,
        "selected_minimum_point_track_length": selected_track_length,
        "point_filter_attempts": point_filter_attempts,
        "sparse_point_count": retained_point_count,
        "rejected_sparse_point_count": rejected_point_count,
        "images": image_rows,
        "sparse_files": sparse_rows,
        "claim_boundary": (
            "Sim3 supplies global metric alignment and a residual inlier gate, but does not "
            "alone admit cameras. Retained cameras also pass independent sparse-geometry "
            "support and current-run lineage checks. "
            "Sparse points are bounded by the retained camera AABB plus a fixed metric margin. "
            "No semantic label, object mask, or mesh-quality feedback is consumed."
        ),
    }
        write_json(staging_root / "metric_input_receipt.json", receipt)
        shutil.rmtree(text_model)
        staging_root.replace(stage_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    print(json.dumps({"status": receipt["status"], "cameras": retained_image_count, "points": retained_point_count, "excluded": sorted(excluded_names)}, indent=2))


if __name__ == "__main__":
    main()
