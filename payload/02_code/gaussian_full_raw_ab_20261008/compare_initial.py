"""Verify every fresh raw decision and output before comparing stage duration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    args = parser.parse_args()
    folders = [args.case_root / side / "initial" for side in ("baseline", "optimized")]
    manifests = [json.loads((folder / "INITIAL_FRAME_MANIFEST.json").read_text()) for folder in folders]
    decisions = [[json.loads(line) for line in (folder / "frame_decisions.jsonl").read_text().splitlines()] for folder in folders]
    contracts = [json.loads((folder / "CONTRACT.json").read_text()) for folder in folders]
    if len(decisions[0]) != len(decisions[1]):
        raise AssertionError("The complete raw decision denominator differs")
    categorical = ("camera_frame", "camera_header_ns", "image_name", "excluded_role", "coarse_pose_valid",
                   "pose_rejection", "decision", "selection_reason", "width", "height", "optical_frame")
    differences = []
    quality_error, pose_error = 0., 0.
    for a, b in zip(*decisions, strict=True):
        if any(a.get(key) != b.get(key) for key in categorical):
            differences.append(a["camera_frame"])
        if set(a["quality"]) != set(b["quality"]):
            raise AssertionError("The original full quality schema differs")
        quality_error = max(quality_error, max(abs(a["quality"][key] - b["quality"][key]) for key in a["quality"]))
        for key in ("C_W_m", "quaternion_WC_xyzw", "R_WC"):
            if key in a:
                pose_error = max(pose_error, float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))))
    selected = [manifest["selected_frames"] for manifest in manifests]
    name_exact = [row["image_name"] for row in selected[0]] == [row["image_name"] for row in selected[1]]
    jpeg_exact = name_exact and all((folders[0] / "images" / row["image_name"]).read_bytes()
        == (folders[1] / "images" / row["image_name"]).read_bytes() for row in selected[0])
    evaluations = [json.loads((folder / "EVALUATION_FRAMES.json").read_text())["rows"] for folder in folders]
    evaluation_exact = ([row["image_name"] for row in evaluations[0]] == [row["image_name"] for row in evaluations[1]]
        and all((folders[0] / "evaluation_images" / row["image_name"]).read_bytes()
            == (folders[1] / "evaluation_images" / row["image_name"]).read_bytes() for row in evaluations[0]))
    checks = {"all_raw_categorical_decisions_exact": not differences,
        "full_original_quality_numerics_within_1e_8": quality_error <= 1e-8,
        "full_coarse_poses_within_1e_9": pose_error <= 1e-9,
        "selected_names_exact": name_exact, "all_selected_JPEG_bytes_exact": jpeg_exact,
        "candidate_pair_file_bytes_exact": (folders[0] / "initial_matching_pairs.txt").read_bytes() == (folders[1] / "initial_matching_pairs.txt").read_bytes(),
        "all_evaluation_JPEG_bytes_exact": evaluation_exact,
        "evaluation_role_frozen_identically": contracts[0]["evaluation_frames"] == contracts[1]["evaluation_frames"],
        "no_historical_caches": all(not contract["quality_cache_enabled"] and not contract["JPEG_cache_enabled"] for contract in contracts),
        "zero_quality_cache_hits": all(manifest["receipt"]["quality_cache_hits"] == 0 for manifest in manifests),
        "no_excluded_training_frames": all(manifest["receipt"]["excluded_overlap"] == 0 for manifest in manifests)}
    times = [manifest["receipt"]["elapsed_seconds"] for manifest in manifests]
    result = {"status": "exact_initial_outputs_verified" if all(checks.values()) else "initial_equivalence_failed",
        "checks": checks, "raw_frames": len(decisions[0]), "selected_count": list(map(len, selected)),
        "max_quality_abs_error": quality_error, "max_pose_abs_error": pose_error,
        "categorical_difference_frames": differences, "baseline_initial_seconds": times[0],
        "optimized_initial_seconds": times[1], "initial_observed_reduction_fraction": 1 - times[1] / times[0],
        "single_pair_stage_measurement": True, "end_to_end_speed_claim": False,
        "all75_modules_verified": False}
    atomic_json(args.case_root / "INITIAL_COMPARISON.json", result)
    print(json.dumps(result))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
