"""Run one frozen same-frame BA case and verify actual prior activation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from common import (Heartbeat, ba_config, ba_options, configure_stdout, make_priors,
                    read_json, solver_receipt, write_json)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max()),
            "mean": float(values.mean())}


def observation_metrics(model):
    all_errors, per_image, observation_count, negative = [], [], 0, 0
    for image in sorted(model.images.values(), key=lambda im: im.image_id):
        points = [point for point in image.points2D if point.has_point3D()]
        xyz = np.asarray([model.points3D[p.point3D_id].xyz for p in points])
        xy = np.asarray([p.xy for p in points])
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        if not np.isfinite(projected).all():
            raise ValueError("Projection contains nonfinite values")
        errors = np.linalg.norm(projected - xy, axis=1)
        all_errors.extend(errors.tolist())
        observation_count += len(points)
        negative += int(np.sum(camera_xyz[:, 2] <= 0))
        per_image.append({"image_id": image.image_id, "image_name": image.name,
                          "observations": len(points), **quantiles(errors)})
    return {"reprojection_px": quantiles(all_errors), "observations": observation_count,
            "negative_depth_observations": negative, "per_image": per_image,
            "registered_images": model.num_reg_images(), "points": model.num_points3D(),
            "track_length": quantiles([p.track.length() for p in model.points3D.values()])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=["A1_visual", "B_w_sigma015", "B_s_sigma005"])
    args = parser.parse_args()
    run = args.run.resolve()
    if read_json(run / "01_priors/CONTRACT_TEST_RECEIPT.json")["status"] != "passed":
        raise RuntimeError("The prior contract preflight did not pass")
    protocol = read_json(run / "00_contract/EVAL_PROTOCOL.json")
    directory = run / "02_same_180_ba" / args.case
    directory.mkdir(parents=True, exist_ok=False)
    heartbeat = Heartbeat(directory / "STATE.json")
    start = time.monotonic()
    try:
        model = pycolmap.Reconstruction(run / "00_contract/initial_model")
        initial = pycolmap.Reconstruction(model)
        rows = read_json(run / "01_priors/camera_prior_manifest.json")["images"]
        before = observation_metrics(model)
        intrinsics = {i: camera.params.copy() for i, camera in model.cameras.items()}
        heartbeat.phase = "points_only_warmup"
        warmup = pycolmap.create_default_ceres_bundle_adjuster(
            ba_options(protocol["max_warmup_iterations"]), ba_config(model, fixed_poses=True), model).solve()
        warmup_info = solver_receipt(warmup)
        if not warmup_info["usable"]:
            raise RuntimeError("Points-only warmup is unusable")
        config = ba_config(model)
        expected_visual_residuals = int(config.num_residuals(model))
        expected_visual_blocks = expected_visual_residuals // 2
        options = ba_options(protocol["max_joint_iterations"])
        remaining_seconds = protocol["max_ba_wall_seconds"] - (time.monotonic() - start)
        if remaining_seconds <= 0:
            raise TimeoutError("The total case BA wall-clock budget is exhausted")
        options.ceres.solver_options.max_solver_time_in_seconds = remaining_seconds
        active_priors = 0
        if args.case == "A1_visual":
            config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
            adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, model)
        else:
            sigma = .15 if args.case == "B_w_sigma015" else .05
            priors = make_priors(model, rows, sigma)
            active_priors = len(priors)
            prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
            prior_options.prior_position_fallback_stddev = sigma
            prior_options.alignment_ransac.max_error = max(.20, 4 * sigma)
            prior_options.alignment_ransac.min_inlier_ratio = .20
            prior_options.alignment_ransac.random_seed = 20260809
            prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
            prior_options.ceres.prior_position_loss_scale = protocol["prior_loss_scale"]
            adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(options, prior_options, config, priors, model)
        aligned_centers = {i: image.projection_center().copy() for i, image in model.images.items()}
        heartbeat.phase = "joint_ba"
        solved = adjuster.solve()
        joint = solver_receipt(solved)
        expected_total = expected_visual_residuals + 3 * active_priors
        if joint["num_residuals"] != expected_total or joint["num_residual_blocks"] != expected_visual_blocks + active_priors:
            write_json(directory / "ACTIVATION_FAILURE.json", {"status": "failed_verification",
                "expected_residuals": expected_total, "expected_prior_count": active_priors, "actual": joint})
            raise RuntimeError("Actual residual counts do not prove expected prior activation")
        if not joint["usable"]:
            raise RuntimeError("Joint BA solution is unusable")
        model.update_point_3d_errors()
        after = observation_metrics(model)
        if set(model.points3D) != set(initial.points3D) or after["observations"] != before["observations"]:
            raise RuntimeError("The frozen common observations changed")
        if model.num_reg_images() != 180:
            raise RuntimeError("A same-frame trial lost camera registration")
        if any(not np.array_equal(c.params, intrinsics[i]) for i, c in model.cameras.items()):
            raise RuntimeError("Frozen intrinsics changed")
        corrections, prior_residuals = [], []
        row_by_id = {row["image_id"]: row for row in rows}
        for image_id, image in sorted(model.images.items()):
            old = initial.images[image_id]
            center = image.projection_center()
            rdelta = image.cam_from_world().rotation.matrix() @ old.cam_from_world().rotation.matrix().T
            rotation_delta = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rdelta).as_rotvec())))
            if abs(np.linalg.det(image.cam_from_world().rotation.matrix()) - 1) > 1e-6:
                raise RuntimeError("An optimized camera has an invalid rotation")
            corrections.append({"image_id": image_id, "image_name": image.name,
                "center_m": center.tolist(), "total_center_delta_m": float(np.linalg.norm(center - old.projection_center())),
                "joint_center_delta_after_alignment_m": float(np.linalg.norm(center - aligned_centers[image_id])),
                "rotation_delta_deg": rotation_delta})
            prior_residuals.append(float(np.linalg.norm(center - np.asarray(row_by_id[image_id]["C_W_m"]))))
        target = directory / "model"
        target.mkdir()
        model.write(target)
        visual_guard = after["reprojection_px"]["p95"] - before["reprojection_px"]["p95"]
        write_json(directory / "RECEIPT.json", {
            "status": "completed_contract_verified", "case": args.case,
            "elapsed_seconds": time.monotonic() - start, "warmup": warmup_info, "joint": joint,
            "expected_visual_residuals": expected_visual_residuals, "active_position_priors": active_priors,
            "prior_activation_verified": active_priors > 0, "before": before, "after": after,
            "position_prior_consistency_m": quantiles(prior_residuals), "camera_corrections": corrections,
            "visual_p95_delta_vs_initial_px": visual_guard, "frozen_intrinsics_preserved": True,
            "frozen_observation_identity_preserved": True, "rotation_restore_used": False,
            "absolute_pose_accuracy": "not_evaluated_missing_independent_GT"})
        print(f"Completed {args.case}: reprojection P95={after['reprojection_px']['p95']:.6f} px, priors={active_priors}")
    finally:
        heartbeat.close()


if __name__ == "__main__":
    configure_stdout()
    main()
