"""Measure a fresh complete original metric node under isolated branch scheduling."""

import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
from metric_branch_dag import COMMON, ROOT, V1, run_dag, run_original
import pycolmap
from runtime import atomic_json, data_identity

V1OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"
MODEL = V1OUT / "p23_track_completion_001/metric_filtered_model"
INITIAL = V1OUT / "p1_raw_combined_001"
GAPS = V1OUT / "p7_pose_gap_recovery_001/POSE_GAP_DECISIONS.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serial", "dag"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("metric_branch_dag.py"), V1 / "run_metric_refinement.py",
        V1 / "runtime.py", V1 / "pose_gap_candidates.py", COMMON / "common.py", COMMON / "run_prior_ba.py",
        Path(pycolmap._core.__file__)]
    inputs = [MODEL/name for name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")]
    inputs += [INITIAL/"INITIAL_FRAME_MANIFEST.json", INITIAL/"frame_decisions.jsonl", GAPS]
    source_ids, input_ids = [data_identity(p) for p in sources], [data_identity(p) for p in inputs]
    model = pycolmap.Reconstruction(MODEL)
    counts = {"registered_images": model.num_reg_images(), "points": model.num_points3D(), "observations": model.compute_num_observations()}
    if counts != {"registered_images": 1021, "points": 75846, "observations": 959881}:
        raise AssertionError("Complete current new-map metric inputs changed")
    del model
    atomic_json(args.output/"CONTRACT.json", {"sources": source_ids, "inputs": input_ids, "counts": counts,
        "mode": args.mode, "complete_original_metric_node": True, "no_GT_or_old_metric_transform": True,
        "full_downstream_quality_not_yet_run": True, "counted": False})
    arguments = ["--model", str(MODEL), "--initial-frames", str(INITIAL), "--output", str(args.output/"node"),
        "--threads", "8", "--include-raw-candidates", "--pose-gap-decisions", str(GAPS)]
    if args.mode == "dag":
        run_dag(arguments)
    else:
        run_original(arguments)
    result = json.loads((args.output/"node/RESULT.json").read_text(encoding="utf-8"))
    guards = {"original_status": result["status"] == "metric_interface_verified",
        "weak_selected": result["selected"] == "weak_position_prior", "original_P95_admission": result["development_guard_passed"]
        and result["common_observation_p95_delta_px"] <= .05}
    for label, row in result["results"].items():
        guards[label+"_full_original_observations"] = row["expected_visual_residuals"] == 1919762 and row["common_observations_preserved"]
        guards[label+"_all_registered"] = row["registered_images"] == 1021
        guards[label+"_native_factor_activation"] = row["factor_count_verified"] and row["solver"]["usable"]
        guards[label+"_prior_count"] = row["active_priors"] == (0 if label == "visual_metric_aligned" else 1015)
        guards[label+"_pose_gaps"] = len(row["pose_gap_images_without_priors"]) == 6 and row["pose_gap_poses_finite"]
    guards["sources_unchanged"] = source_ids == [data_identity(p) for p in sources]
    guards["inputs_unchanged"] = input_ids == [data_identity(p) for p in inputs]
    complete_phase = time.perf_counter()-STARTED
    atomic_json(args.output/"RESULT.json", {"status": "complete_metric_node_passed_downstream_and_cost_pending" if all(guards.values()) else "required_complete_metric_guard_failed",
        "guards": guards, "original_selected": result["selected"], "P95_delta_px": result["common_observation_p95_delta_px"],
        "complete_affected_phase_seconds": complete_phase, "whole_child_seconds": time.perf_counter()-STARTED,
        "both_branch_semantics": {label: {"elapsed_seconds": row["elapsed_seconds"], "final_cost": row["solver"]["final_cost"],
            "native_solver": row["solver"]["native_summary"], "reprojection": row["metrics"]["reprojection_px"]}
            for label, row in result["results"].items()}, "counted": False})
    if not all(guards.values()):
        raise AssertionError("A required complete original metric guard failed")


if __name__ == "__main__":
    main()
