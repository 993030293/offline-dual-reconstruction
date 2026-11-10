"""Verify frozen identities and assemble truthful first-pass execution receipts."""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

from augment_views import augmented_prior_row, target_support
from common import VISUAL, configure_stdout, digest_data, make_priors, read_json, write_json
from run_prior_ba import quantiles


def frozen_common_observations(initial, model):
    errors, negative = [], 0
    for image_id, original in initial.images.items():
        image = model.images[image_id]
        observations = [(i, p) for i, p in enumerate(original.points2D) if p.has_point3D()]
        for index, point in observations:
            if image.points2D[index].point3D_id != point.point3D_id or not np.array_equal(image.points2D[index].xy, point.xy):
                raise ValueError("Augmentation changed the frozen original observation identity")
        xyz = np.asarray([model.points3D[p.point3D_id].xyz for _, p in observations])
        xy = np.asarray([p.xy for _, p in observations])
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T + pose.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        if not np.isfinite(projected).all():
            raise ValueError("Frozen augmented observations project to nonfinite pixels")
        errors.extend(np.linalg.norm(projected - xy, axis=1))
        negative += int(np.sum(camera_xyz[:, 2] <= 0))
    return {"reprojection_px": quantiles(errors), "negative_depth_observations": negative,
            "original_observation_identity_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    if (run / "FINAL_RECEIPT.json").exists():
        raise FileExistsError("Final execution receipts already exist")
    baseline = read_json(run / "00_contract/BASELINE_MANIFEST.json")
    preserved = all(digest_data(Path(path)) == value for path, value in baseline["data_sha256"].items())
    preserved &= all(digest_data(VISUAL / "images" / row["image_name"]) == row["image_sha256"] for row in baseline["source_images"])
    if not preserved:
        raise ValueError("A frozen source image, database or original model changed")
    initial = pycolmap.Reconstruction(run / "00_contract/initial_model")
    augmented = pycolmap.Reconstruction(run / "03_coverage_augmented/model")
    source = read_json(run / "03_coverage_augmented/refinement_attempt2/camera_prior_manifest.json")
    corrected = []
    for row in source["images"]:
        current = dict(row)
        if row["image_id"] > 180:
            current = augmented_prior_row(augmented, row, row["image_id"])
            current["prior_active"] = row["prior_active"]
        if current["image_name"] != augmented.images[current["image_id"]].name:
            raise ValueError("Final augmented prior name/ID binding is invalid")
        required = {"frame_uid", "image_name", "image_id", "camera_id", "frame_id", "corr_data_id", "position_covariance_m2",
                    "left_pose_timestamp_ns", "right_pose_timestamp_ns", "bracket_span_ns", "C_W_m", "R_WC", "quality_sharpness"}
        if not required.issubset(current):
            raise ValueError("The completed augmented manifest is missing required metadata")
        corrected.append(current)
    expected = make_priors(augmented, corrected, .15)
    with pycolmap.Database.open(run / "03_coverage_augmented/refinement_attempt2/database.db") as database:
        actual = {int(p.corr_data_id.id): p for p in database.read_all_pose_priors()}
        if len(actual) != len(expected):
            raise ValueError("Final augmented prior readback count changed")
        for prior in expected:
            restored = actual[int(prior.corr_data_id.id)]
            if restored.corr_data_id != prior.corr_data_id or not np.array_equal(restored.position, prior.position) or not np.array_equal(restored.position_covariance, prior.position_covariance):
                raise ValueError("Completed metadata changed the numerical prior used by BA")
    write_json(run / "03_coverage_augmented/camera_prior_manifest_verified.json", {"status": "verified_metadata_complete",
        "images": corrected, "active_prior_count": len(expected), "numeric_priors_changed": False,
        "source_solver_manifest_preserved": True})
    common = frozen_common_observations(initial, augmented)
    write_json(run / "05_evaluation/AUGMENTED_COMMON_OBSERVATION_METRICS.json", common)
    cases, activation = [], []
    roi = read_json(run / "03_coverage_augmented/SELECTION_MANIFEST.json")["ROI"]
    for case in ["A1_visual", "B_w_sigma015", "C_augmented_212"]:
        state = read_json(run / "04_downstream" / case / "MAIN_STATE.json")
        exported = read_json(run / "04_downstream" / case / "EXPORT_CONTRACT.json")
        if state["status"] != "completed" or state["iteration"] != 10000 or exported["status"] != "passed":
            raise ValueError("A registered training or export is incomplete")
        filtered = pycolmap.Reconstruction(run / "04_downstream" / case / "filtered_opencv_model")
        cases.append({"case": case, "training": state,
            "filtered_points": exported["points_after_filter"], "filtered_ROI_support": target_support(filtered, roi),
            "image_count": exported["input_cameras"], "export_elapsed_seconds": exported["elapsed_seconds"]})
    for case in ["A1_visual", "B_w_sigma015", "B_s_sigma005"]:
        receipt = read_json(run / "02_same_180_ba" / case / "RECEIPT.json")
        activation.append({"case": case, "visual_scalar_residuals": receipt["expected_visual_residuals"],
            "actual_full_scalar_residuals": receipt["joint"]["num_residuals"],
            "actual_full_residual_blocks": receipt["joint"]["num_residual_blocks"],
            "active_prior_count": receipt["active_position_priors"], "usable": receipt["joint"]["usable"],
            "joint_termination": receipt["joint"]["termination_type"]})
    write_json(run / "01_priors/PRIOR_ACTIVATION_RECEIPT.json", {"status": "verified",
        "same_frame_cases": activation, "augmented_active_priors": len(expected),
        "counter_source": "full_Ceres_summary_not_reduced_summary", "position_residual_dimensions": 3})
    render = read_json(run / "05_evaluation/RENDER_COMPARISON.json")
    browser = read_json(run / "viewer/BROWSER_VALIDATION.json")
    if browser["status"] != "passed":
        raise ValueError("Viewer interactions did not pass")
    snapshot = run / "code_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in {".py", ".ps1", ".js", ".html"}:
            shutil.copy2(path, snapshot / path.name)
    receipt = {"status": "FIRST_PASS_DEVELOPMENT_EXPERIMENT_COMPLETED_WITH_LIMITATIONS",
        "plan_revision": "r1_20261002T005049_9bdfd943", "original_inputs_preserved": preserved,
        "recorded_trajectory_provider": "recorded_current_mcap_odometry_exact_algorithm_unconfirmed",
        "Point_LIO_replay_performed": False, "fresh_pose_prior_mapper_performed": False,
        "BA_cases": activation, "augmented_images": augmented.num_reg_images(),
        "augmentation_common_observations": common, "downstream_cases": cases, "render_metrics": render["cases"],
        "weak_prior_quality_gain_accepted": False, "augmentation_quality_status": "candidate_development_render_gain",
        "all_H_images": 24, "H_images_registered_and_rendered_each_case": 24,
        "H_used_for_tuning_or_selecting_this_run_cases": False,
        "H_read_for_final_interpretation": True, "all_data_is_previously_reviewed_development_recording": True,
        "unused_H_correspondence_accuracy_status": "unconfirmed_remaining_mismatches_not_true_pose_error",
        "native_GS_projection_status": "known_shared_noncentered_principal_point_limitation",
        "principal_point_offset_training_resolution_px": [x / 2 for x in render["PINHOLE_principal_point_offset_from_image_center_px"]],
        "absolute_pose_ATE_RPE_and_surface_accuracy": "not_evaluated_missing_independent_reference",
        "LPIPS": "not_evaluated_optional_metric", "single_seed_per_case": True,
        "review_kind": "self_review_not_independent_scientific_review",
        "viewer_url": "http://127.0.0.1:8195/", "viewer_server": read_json(run / "viewer/SERVER_STATE.json"),
        "final_code_snapshot": snapshot,
        "post_run_guard_and_metadata_changes": ["combined_wall_time_guard", "complete_augmented_prior_metadata", "common_viewer_default_capture"],
        "code_change_boundary": "numeric_prior_readback_rechecked_and_code_compiled_no_extra_BA_or_training_repeated",
        "background_training_remaining": False,
        "budget_counts": {"same_frame_BA": 3, "augmented_registration_campaign": 1, "augmented_BA": 1,
                          "same_frame_Gaussian_main": 2, "additional_Gaussian_main": 1}}
    write_json(run / "FINAL_RECEIPT.json", receipt)
    print({"status": receipt["status"], "common_C_p95": common["reprojection_px"]["p95"],
           "original_inputs_preserved": preserved, "active_augmented_priors": len(expected)})


if __name__ == "__main__":
    configure_stdout()
    main()
