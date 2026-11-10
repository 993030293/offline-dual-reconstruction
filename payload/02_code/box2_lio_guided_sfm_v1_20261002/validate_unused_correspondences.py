"""Diagnose unused matches with loc-only epipolar models and unchanged H poses."""

import argparse
from pathlib import Path

import numpy as np
import pycolmap

from common import configure_stdout, read_json, write_json
from feature_localization import FeatureBank, pose_curve
from run_prior_ba import quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    root = run / "05_evaluation"
    write_json(root / "UNUSED_MATCH_DIAGNOSTIC_PROTOCOL.json", {
        "status": "bounded_metric_repair_after_unfiltered_diagnostic",
        "preserved_v1_metric": root / "H_LOCALIZATION_COMPARISON.json",
        "v1_quality_status": "failed_verification_tentative_SIFT_mismatches_not_geometry_validated",
        "threshold_source": run / "00_contract/LOCALIZATION_PROTOCOL.json",
        "epipolar_max_error_px": 4., "threshold_changed": False,
        "F_estimation_uses": "H_loc_even_features_only", "H_eval_in_F_fit": False,
        "H_poses_reestimated": False, "training_or_candidate_selection_changed": False,
        "interpretation": "supplementary_development_consistency_diagnostic_not_a_replaced_frozen_acceptance_score"})
    bank, curve, comparison = FeatureBank(run), pose_curve(), []
    for case in ["A1_visual", "B_w_sigma015", "C_augmented_212"]:
        old = read_json(root / case / "H_LOCALIZATION.json")
        path = run / ("03_coverage_augmented/model" if case.startswith("C_") else f"02_same_180_ba/{case}/model")
        model = pycolmap.Reconstruction(path)
        camera = next(iter(model.cameras.values()))
        results = []
        for row in old["rows"]:
            keypoints, descriptors = bank.extra(run / "00_contract/extra_images" / row["image_name"])
            neighbors = bank.neighbors(model, curve.at(row["camera_header_ns"]), row["camera_header_ns"])
            even, odd = np.arange(0, len(keypoints), 2), np.arange(1, len(keypoints), 2)
            loc, edges, _ = bank.match(model, keypoints[even], descriptors[even], neighbors, geometric=True)
            evaluate, _, _ = bank.match(model, keypoints[odd], descriptors[odd], neighbors, geometric=False)
            if len(loc) != row["loc_correspondences"]:
                raise RuntimeError("Replayed H_loc identity/count differs from the frozen localization")
            loc_ids = {r["point_id"] for r in loc}
            evaluate = [r for r in evaluate if r["point_id"] not in loc_ids]
            matrices = {edge["train_image_id"]: np.asarray(edge["F_from_query_localization_features"])
                        for edge in edges if edge["F_from_query_localization_features"] is not None}
            accepted, rejected = [], 0
            for correspondence in evaluate:
                matrix = matrices.get(correspondence["train_image_id"])
                if matrix is None:
                    rejected += 1
                    continue
                query = int(odd[correspondence["query"]])
                xy = keypoints[query, :2]
                train_xy = bank.original[correspondence["train_image_id"]][1][correspondence["train_index"], :2]
                x, y = np.r_[xy, 1.], np.r_[train_xy, 1.]
                fx, fty = matrix @ x, matrix.T @ y
                denominator = np.sum(fx[:2] ** 2) + np.sum(fty[:2] ** 2)
                error2 = (y @ fx) ** 2 / max(denominator, 1e-20)
                if error2 <= 4.**2:
                    accepted.append((query, correspondence["point_id"]))
                else:
                    rejected += 1
            record = {"image_name": row["image_name"], "tentative_unused_matches": len(evaluate),
                      "accepted_static_epipolar_matches": len(accepted), "rejected_matches": rejected,
                      "H_pose_unchanged": True, "no_loc_eval_point_overlap": True}
            if accepted and row["status"] == "localized":
                xyz = np.asarray([model.points3D[p].xyz for _, p in accepted])
                xy = np.asarray([keypoints[q, :2] for q, _ in accepted])
                projection = camera.img_from_cam(xyz @ np.asarray(row["R_CW"]).T + row["t_CW"])
                record["validated_unused_reprojection_px"] = quantiles(np.linalg.norm(projection - xy, axis=1))
            results.append(record)
        summary = {"case": case, "all_H_count": len(results),
            "frames_with_validated_unused_matches": sum(r["accepted_static_epipolar_matches"] > 0 for r in results),
            "per_frame_P95_px": quantiles([r["validated_unused_reprojection_px"]["p95"] for r in results if "validated_unused_reprojection_px" in r]),
            "tentative_matches": sum(r["tentative_unused_matches"] for r in results),
            "accepted_matches": sum(r["accepted_static_epipolar_matches"] for r in results),
            "rejected_matches": sum(r["rejected_matches"] for r in results)}
        write_json(root / case / "VALIDATED_UNUSED_CORRESPONDENCES.json", {"status": "diagnostic_completed", "summary": summary, "rows": results})
        comparison.append(summary)
    write_json(root / "VALIDATED_UNUSED_COMPARISON.json", {"status": "diagnostic_completed", "cases": comparison,
        "v1_preserved": True, "PnP_map_and_training_unchanged": True,
        "accuracy_boundary": "not_independent_GT_not_a_new_acceptance_gate"})
    print(comparison)


if __name__ == "__main__":
    configure_stdout()
    main()
