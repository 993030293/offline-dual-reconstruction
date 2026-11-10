"""Protect existing camera admission and common observations after pair recovery."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
PACKAGE = ROOT/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
sys.path.insert(0, str(V1))
sys.path.insert(0, str(PACKAGE/"code/validated_fresh_runtime"))
from runtime import atomic_json, data_identity
from metric_admission import camera_geometry_metrics
from native_observation_checks import projection_validity_native


def compare_recovery(reference_result, candidate_result):
    reference = json.loads(Path(reference_result).read_text())
    candidate = json.loads(Path(candidate_result).read_text())
    reference_summary = reference["final"] if "final" in reference else max(reference["models"], key=lambda r: r["registered_count"])
    candidate_summary = candidate["final"]
    models = {"baseline": pycolmap.Reconstruction(Path(reference_summary["path"])),
              "candidate": pycolmap.Reconstruction(Path(candidate_summary["path"]))}
    policy_path = PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())["camera_geometry_admission"]
    support = {m: {im.name: camera_geometry_metrics(model, im, policy)
        for im in model.images.values() if im.has_pose} for m, model in models.items()}
    passed = {m: {name for name, r in rows.items() if r["passed"]} for m, rows in support.items()}
    validity = {m: projection_validity_native(model) for m, model in models.items()}
    checks = {"complete1002_denominator_kept": candidate["original_denominator"] == 1002,
        "no_baseline_registered_camera_lost": set(support["baseline"]) <= set(support["candidate"]),
        "no_baseline_admitted_camera_lost": passed["baseline"] <= passed["candidate"],
        "all_candidate_projection_finite_and_positive": validity["candidate"]["valid"],
        "retained_original_observations_not_reduced": candidate_summary["retained_old_observations"] >= reference_summary["retained_old_observations"],
        "mean_reprojection_error_guard": candidate_summary["mean_reprojection_error_px"] <= reference_summary["mean_reprojection_error_px"]+.05,
        "all_recorded_original_pose_and_intrinsics_guards": all(b["original_pose_max_delta"] <= 1e-9
            and b["original_intrinsics_exact"] and not b["summary"]["original_registered_frames_lost"] for b in candidate["batches"]),
        "source_inputs_unchanged": candidate["source_inputs_unchanged"]}
    return {"status": "recovery_geometry_protected" if all(checks.values()) else "failed_geometry_guard",
        "checks": checks, "admission_policy_source": data_identity(policy_path),
        "registered": {m: len(v) for m, v in support.items()}, "admitted": {m: len(v) for m, v in passed.items()},
        "lost_admitted_names": sorted(passed["baseline"]-passed["candidate"]),
        "new_admitted_names": sorted(passed["candidate"]-passed["baseline"]),
        "projection_validity": validity, "support": support,
        "original_observations_retained": {"baseline": reference_summary["retained_old_observations"],
            "candidate": candidate_summary["retained_old_observations"]},
        "total_observations": {"baseline": reference_summary["new_total_observations"],
            "candidate": candidate_summary["new_total_observations"]},
        "mean_reprojection_error_px": {"baseline": reference_summary["mean_reprojection_error_px"],
            "candidate": candidate_summary["mean_reprojection_error_px"]},
        "total_new_observation_count_is_disclosed_not_a_quality_oracle": True,
        "three_pair_speed_and_metric_training_quality_pending": True, "counted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = compare_recovery(args.reference, args.candidate)
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("support", "projection_validity")}), flush=True)
    if not all(result["checks"].values()):
        raise AssertionError("Progressive recovery failed unchanged geometry policy")


if __name__ == "__main__":
    main()
