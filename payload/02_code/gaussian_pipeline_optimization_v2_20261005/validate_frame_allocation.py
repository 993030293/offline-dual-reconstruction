"""Check raw-only allocation geometry, exclusion gates and the reproduced gap failure."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from frame_allocation import eligible_rows, pose_distance, temporal_backbone, rank_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.initial/"INITIAL_FRAME_MANIFEST.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.initial/"frame_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    policy, threshold = manifest["policy"], manifest["receipt"]["sharpness_min"]
    eligible = eligible_rows(rows, policy, threshold)
    checks = {}

    def rejects(label, function):
        try:
            function()
        except ValueError:
            checks[label] = True
        else:
            checks[label] = False

    rejects("duplicate_raw_identity_rejected", lambda: eligible_rows(rows+[copy.deepcopy(rows[0])], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["quality"]["quality_score"] = float("nan")
    rejects("nonfinite_quality_rejected", lambda: eligible_rows([sample], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["quaternion_WC_xyzw"] = [0., 0., 0., 2.]
    rejects("invalid_rotation_rejected", lambda: eligible_rows([sample], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["extrapolated"] = True
    rejects("extrapolated_pose_rejected", lambda: eligible_rows([sample], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["source_role"] = "refined_SfM_pose"
    rejects("SfM_pose_provenance_rejected", lambda: eligible_rows([sample], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["quality"]["sharpness"] = threshold-1
    rejects("forged_quality_eligibility_rejected", lambda: eligible_rows([sample], policy, threshold))
    sample = copy.deepcopy(eligible[0])
    sample["excluded_role"] = "development_evaluation"
    checks["protected_evaluation_never_selected"] = eligible_rows([sample], policy, threshold) == []
    sample = copy.deepcopy(eligible[0])
    sample["coarse_pose_valid"] = False
    checks["no_time_support_never_initially_selected"] = eligible_rows([sample], policy, threshold) == []
    distance, position, angle = pose_distance(np.array([[.1, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0, -1], [0, 0, np.sqrt(.5), np.sqrt(.5)]]), np.zeros(3), np.array([0, 0, 0, 1]))
    checks["known_translation_and_sign_invariant_rotation_oracle"] = bool(
        np.allclose(position, [.1, 0], rtol=0, atol=1e-12)
        and np.allclose(angle, [0, 90], rtol=0, atol=1e-12)
        and np.allclose(distance, [1, 9], rtol=0, atol=1e-12))
    baseline_names = {row["image_name"] for row in manifest["selected_frames"]}
    baseline = [row for row in eligible if row["image_name"] in baseline_names]
    origin = min(row["camera_header_ns"] for row in rows)
    required = temporal_backbone(baseline, origin)
    rejects("budget_below_backbone_rejected", lambda: rank_rows(baseline, len(required)-1, required))
    order = rank_rows(baseline, 651, required)
    chosen = sorted((baseline[i] for i in order[:501]), key=lambda row: row["camera_header_ns"])
    checks["observed_1795_to1836_gap_repaired_with_real_pool_frame"] = any(
        1795 < row["camera_frame"] < 1836 for row in chosen)
    new_gaps = []
    for a, b in zip(chosen, chosen[1:]):
        if b["camera_header_ns"]-a["camera_header_ns"] > 800_000_000:
            candidates = [row for row in baseline if a["camera_header_ns"] < row["camera_header_ns"] < b["camera_header_ns"]]
            if candidates:
                new_gaps.append([a["camera_frame"], b["camera_frame"]])
    checks["no_unfilled_gap_with_actual_fixed_pool_support"] = not new_gaps
    recorded = json.loads((args.audit/"pool1002_initial651/ALLOCATION.json").read_text(encoding="utf-8"))
    checks["fresh_ranking_reproduces_recorded_actual651_names"] = {baseline[i]["image_name"] for i in order} == {row["image_name"] for row in recorded["initial"]}
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "source": data_identity(Path(__file__)), "implementation": data_identity(Path(__file__).with_name("frame_allocation.py")),
              "actual_initial_count": len(order), "backbone_count": len(required), "counted": False,
              "scope": "selection regression only; no SfM/quality acceptance"}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise AssertionError("Allocation validation failed")


if __name__ == "__main__":
    main()
