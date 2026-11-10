"""Compare the frozen BA cases without consulting the held-out frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from common import configure_stdout, read_json, write_json
from run_prior_ba import quantiles


def similarity(source, target):
    source, target = np.asarray(source), np.asarray(target)
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular * signs) / np.mean(np.sum(x * x, axis=1)))
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    residuals = np.linalg.norm(source @ rotation.T * scale + translation - target, axis=1)
    return {"scale": scale, "rotation": rotation, "translation_m": translation,
            "center_deformation_after_fitted_similarity_m": quantiles(residuals),
            "interpretation": "descriptive_gauge_and_shape_separation_not_GT_accuracy"}


def triangulation_metrics(model):
    angles = []
    centers = {i: im.projection_center() for i, im in model.images.items()}
    for point in model.points3D.values():
        rays = np.asarray([centers[t.image_id] - point.xyz for t in point.track.elements])
        norms = np.linalg.norm(rays, axis=1)
        if len(rays) < 2 or np.any(norms == 0):
            continue
        rays /= norms[:, None]
        angles.append(float(np.rad2deg(np.arccos(np.clip((rays @ rays.T).min(), -1, 1)))))
    return quantiles(angles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run / "02_same_180_ba"
    names = ["A1_visual", "B_w_sigma015", "B_s_sigma005"]
    protocol = read_json(args.run / "00_contract/EVAL_PROTOCOL.json")
    initial = pycolmap.Reconstruction(args.run / "00_contract/initial_model")
    initial_centers = np.asarray([initial.images[i].projection_center() for i in sorted(initial.images)])
    records, poses = [], []
    reference = read_json(root / names[0] / "RECEIPT.json")
    reference_p95 = reference["after"]["reprojection_px"]["p95"]
    for name in names:
        receipt = read_json(root / name / "RECEIPT.json")
        model = pycolmap.Reconstruction(root / name / "model")
        centers = np.asarray([model.images[i].projection_center() for i in sorted(model.images)])
        record = {"case": name, "status": receipt["status"],
                  "reprojection_px": receipt["after"]["reprojection_px"],
                  "negative_depth_observations": receipt["after"]["negative_depth_observations"],
                  "track_length": receipt["after"]["track_length"],
                  "maximum_triangulation_angle_deg": triangulation_metrics(model),
                  "position_prior_consistency_m": receipt["position_prior_consistency_m"],
                  "active_position_priors": receipt["active_position_priors"],
                  "prior_activation_verified": receipt["prior_activation_verified"],
                  "joint_termination": receipt["joint"]["termination_type"],
                  "elapsed_seconds": receipt["elapsed_seconds"],
                  "p95_delta_vs_A1_px": receipt["after"]["reprojection_px"]["p95"] - reference_p95,
                  "fitted_initial_to_final_similarity": similarity(initial_centers, centers)}
        record["development_guard_passed"] = record["p95_delta_vs_A1_px"] <= protocol["common_reprojection_p95_development_tolerance_px"]
        records.append(record)
        for row in receipt["camera_corrections"]:
            poses.append({"case": name, "image_id": row["image_id"], "image_name": row["image_name"],
                          "center_x_m": row["center_m"][0], "center_y_m": row["center_m"][1],
                          "center_z_m": row["center_m"][2], "total_center_delta_m": row["total_center_delta_m"],
                          "joint_center_delta_after_alignment_m": row["joint_center_delta_after_alignment_m"],
                          "rotation_delta_deg": row["rotation_delta_deg"]})
    weak = records[1]
    selected = "B_w_sigma015" if weak["development_guard_passed"] and weak["prior_activation_verified"] else "A1_visual"
    write_json(root / "COMMON_OBSERVATION_METRICS.json", {"records": records,
        "common_observations": reference["after"]["observations"], "filter_applied": False,
        "absolute_accuracy": "not_evaluated_missing_independent_GT"})
    write_json(root / "SELECTION_RECEIPT.json", {"selected_case": selected,
        "comparison_B_case": "B_w_sigma015", "selection_data": "training_contract_and_predeclared_P95_guard_only",
        "H_consulted": False, "D_cal_consulted": False,
        "reason": "default_weak_candidate_contract_valid_and_guard_passed" if selected.startswith("B") else "weak_candidate_failed_guard",
        "method_benefit_accepted": False, "boundary": "guard_pass_is_not_evidence_of_quality_or_absolute_accuracy_improvement"})
    with (root / "POSE_COMPARISON.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(poses[0]))
        writer.writeheader()
        writer.writerows(poses)
    print({"selected_case": selected, "common_observations": reference["after"]["observations"],
           "p95_deltas_px": {r["case"]: r["p95_delta_vs_A1_px"] for r in records}})


if __name__ == "__main__":
    configure_stdout()
    main()
