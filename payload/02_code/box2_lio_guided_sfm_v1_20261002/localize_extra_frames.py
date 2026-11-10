"""Localize frozen evaluation images without changing any training reconstruction."""

import argparse
import time
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

from augment_views import roi_contract, target_window
from common import Heartbeat, configure_stdout, read_json, write_json
from feature_localization import FeatureBank, estimate_pose, pose_curve
from run_prior_ba import quantiles


def split_correspondences(bank, model, keypoints, descriptors, neighbors):
    even = np.arange(0, len(keypoints), 2)
    odd = np.arange(1, len(keypoints), 2)
    localization, loc_edges, _ = bank.match(model, keypoints[even], descriptors[even], neighbors, geometric=True)
    evaluation, eval_edges, _ = bank.match(model, keypoints[odd], descriptors[odd], neighbors, geometric=False)
    for row in localization:
        row["query"] = int(even[row["query"]])
    for row in evaluation:
        row["query"] = int(odd[row["query"]])
    used_points = {row["point_id"] for row in localization}
    evaluation = [row for row in evaluation if row["point_id"] not in used_points]
    return localization + evaluation, loc_edges, eval_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = run / "05_evaluation"
    root.mkdir(exist_ok=True)
    references = root / "reference_H"
    references.mkdir(exist_ok=True)
    cases = ["A1_visual", "B_w_sigma015", "C_augmented_212"]
    if (root / "H_LOCALIZATION_COMPARISON.json").exists():
        raise FileExistsError("The evaluation campaign is already preserved")
    write_json(root / "H_MECHANICS_RECEIPT.json", {"status": "frozen_before_first_H_localization",
        "H_eval_affects_PnP": False, "H_loc_matching": "even_features_only_including_its_geometric_verification",
        "H_eval_matching": "odd_features_tentative_mutual_ratio_SIFT_without_pose_based_filter",
        "H_eval_3D_points": "exclude_all_points_used_in_H_loc",
        "H_eval_reprojection_boundary": "all_unused_tentative_matches_include_mismatches_not_GT_correspondences",
        "model_update": False, "candidate_selection_from_H": False,
        "render_ROI": "one_A1_localized_PINHOLE_target_rectangle_per_H_image_shared_by_all_cases",
        "render_pose_policy": "own_training_map_PnP_H_loc_pose_combined_localization_and_rendering_metric",
        "recording_status": "previously_reviewed_development_recording_not_strict_blind_test"})
    split = read_json(run / "00_contract/split_manifest.json")
    evaluation_rows = [r for r in split["extra_frames"] if r["role"] == "H"]
    bank, curve, roi = FeatureBank(run), pose_curve(), roi_contract(run)
    comparison = []
    for case in cases:
        output = root / case
        output.mkdir(exist_ok=False)
        heartbeat = Heartbeat(output / "H_LOCALIZATION_STATE.json")
        start = time.monotonic()
        try:
            model_path = run / ("03_coverage_augmented/model" if case.startswith("C_") else f"02_same_180_ba/{case}/model")
            model = pycolmap.Reconstruction(model_path)
            camera = next(iter(model.cameras.values()))
            pinhole = next(iter(pycolmap.Reconstruction(run / "04_downstream" / case / "dataset/sparse/0").cameras.values()))
            original_intrinsics = camera.params.copy()
            training_names = {im.name for im in model.images.values()}
            results = []
            heartbeat.phase = "frozen_H_localization_no_map_update"
            for row in evaluation_rows:
                if row["image_name"] in training_names:
                    raise RuntimeError("An H image leaked into a training map")
                path = run / "00_contract/extra_images" / row["image_name"]
                reference = references / (path.stem + ".png")
                if not reference.exists():
                    rgb = np.asarray(Image.open(path).convert("RGB"))
                    bitmap, undistorted_camera = pycolmap.undistort_image(
                        pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(rgb), camera)
                    if not np.array_equal(undistorted_camera.params, pinhole.params):
                        raise RuntimeError("Evaluation undistortion differs from the training camera")
                    Image.fromarray(bitmap.to_array()).save(reference)
                keypoints, descriptors = bank.extra(path)
                rough = curve.at(row["camera_header_ns"])
                neighbors = bank.neighbors(model, rough, row["camera_header_ns"])
                correspondences, loc_edges, eval_edges = split_correspondences(bank, model, keypoints, descriptors, neighbors)
                result, info, localization, evaluation = estimate_pose(model, keypoints, correspondences, split=True)
                record = {**row, **info, "image_reference": reference, "feature_count": len(keypoints),
                          "localization_edges": loc_edges, "evaluation_edges": eval_edges,
                          "PnP_uses_even_features_only": True, "map_updated": False}
                if result is not None:
                    estimated = {"C_W_m": result["cam_from_world"].inverse().translation.tolist(),
                                 "R_WC": result["cam_from_world"].rotation.matrix().T.tolist()}
                    record["target_window_PINHOLE"] = target_window(pinhole, estimated, roi)
                    record["rough_to_localized_center_distance_m"] = float(np.linalg.norm(np.asarray(rough["C_W_m"]) - estimated["C_W_m"]))
                    if info["loc_eval_3D_overlap"] != 0:
                        raise RuntimeError("Localization and evaluation share a 3D point")
                results.append(record)
                print(f"{case} H {row['camera_frame']}: {info['status']}", flush=True)
            if not np.array_equal(camera.params, original_intrinsics):
                raise RuntimeError("Held-out localization changed training intrinsics")
            localized = [r for r in results if r["status"] == "localized"]
            summary = {"case": case, "all_H_count": len(results), "localized_count": len(localized),
                       "quality_eligible_H_count": sum(r["quality_eligible"] for r in results),
                       "all_H_denominator_kept": True,
                       "unused_correspondence_P95_px_per_frame": quantiles([r["eval_reprojection_px_all_unused_correspondences"]["p95"]
                           for r in localized if r.get("eval_reprojection_px_all_unused_correspondences", {}).get("count", 0)]),
                       "elapsed_seconds": time.monotonic() - start,
                       "geometry_or_pose_accuracy": "not_evaluated_missing_independent_GT"}
            write_json(output / "H_LOCALIZATION.json", {"status": "completed", "summary": summary,
                "rows": results, "map_source": model_path, "map_updated": False,
                "PINHOLE_camera": pinhole.todict(), "H_data_used_for_selection": False})
            comparison.append(summary)
        finally:
            heartbeat.close()
    write_json(root / "H_LOCALIZATION_COMPARISON.json", {"status": "completed", "cases": comparison,
        "boundary": "map_and_PnP_development_evaluation_not_true_pose_error"})


if __name__ == "__main__":
    configure_stdout()
    main()
