"""Audit and export a fresh R129-selected map through unchanged native admission."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

STARTED = time.perf_counter()
from cluster_ba import COMMON, ROOT, V1
import audit_camera_roles
import export_admitted_map
import common
from runtime import atomic_json, data_identity

V1OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"
INITIAL = V1OUT / "p1_raw_combined_001"
GAPS = V1OUT / "p7_pose_gap_recovery_001/POSE_GAP_DECISIONS.json"
IMAGES = V1OUT / "e2_geometry_support_001/images"
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ba001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def invoke(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [str(Path(module.__file__)), *map(str, arguments)]
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-admission", type=Path)
    args = parser.parse_args()
    args.metric.resolve().relative_to(OWNED.resolve())
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    metric = read(args.metric / "RESULT.json")
    if (metric["status"] != "metric_interface_verified" or not metric["development_guard_passed"]
            or metric["selected"] != "weak_position_prior" or metric["common_observation_p95_delta_px"] > .05):
        raise ValueError("Own admission requires the unchanged full weak metric guard")
    selected_model = Path(metric["results"]["weak_position_prior"]["model"])
    sources = [Path(__file__), V1/"audit_camera_roles.py", V1/"export_admitted_map.py", V1/"run_metric_refinement.py",
        V1/"pose_gap_candidates.py", COMMON/"common.py", common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [args.metric/"RESULT.json", *sorted(selected_model.glob("*.bin")), INITIAL/"INITIAL_FRAME_MANIFEST.json",
        INITIAL/"frame_decisions.jsonl", GAPS, *sorted(p for p in IMAGES.iterdir() if p.is_file())]
    if args.control_admission:
        inputs.append(args.control_admission)
    source_ids, input_ids = [data_identity(p) for p in sources], [data_identity(p) for p in inputs]
    atomic_json(args.output/"CONTRACT.json", {"source": source_ids, "inputs": input_ids,
        "own_complete_selected_map": str(selected_model), "original_policy_unchanged": True,
        "all_control_admitted_cameras_required": args.control_admission is not None, "counted": False})
    invoke(audit_camera_roles, ["--model", selected_model, "--initial-frames", INITIAL, "--pose-gap-decisions", GAPS,
        "--output", args.output/"roles", "--phase", "after_BA"])
    roles = read(args.output/"roles/TRAINING_ROLES.json")
    if (roles["registered_count"] != 1021 or roles["position_prior_count"] != 1015
            or len(roles["pose_gap_names"]) != 6 or roles["initial_count"] != 1002
            or roles["raw_candidate_count"] != 7338 or roles["protected_development_exclusion_count"] != 48):
        raise AssertionError("The complete original initial/map/prior/role denominators changed")
    invoke(export_admitted_map, ["--ba", args.metric, "--roles", args.output/"roles/TRAINING_ROLES.json",
        "--images", IMAGES, "--output", args.output/"export"])
    admission = read(args.output/"export/TRAINING_ADMISSION.json")
    names = sorted(row["image_name"] for row in admission["rows"] if row["training_role"] == "admitted")
    checks = {"original_complete_export": admission["status"] == "export_and_admission_verified_training_pending",
        "complete_registered_denominator": admission["full_registered_denominator"] == 1021,
        "full_initial_denominator": admission["initial_count"] == 1002,
        "original_final_support": all(row["passed"] for row in admission["final_pinhole_support"]),
        "original_full_positive_finite_projection": admission["final_projection"]["valid"],
        "sources_unchanged": source_ids == [data_identity(p) for p in sources],
        "inputs_unchanged": input_ids == [data_identity(p) for p in inputs],
        "own_group_under_30GiB_free_20GiB": sum(p.stat().st_size for p in OWNED.rglob("*") if p.is_file()) <= 30*1024**3
            and shutil.disk_usage(OWNED).free >= 20*1024**3}
    if args.control_admission:
        control = read(args.control_admission)
        control_names = {row["image_name"] for row in control["rows"] if row["training_role"] == "admitted"}
        checks["all_control_qualified_training_cameras_retained"] = control_names.issubset(names)
    atomic_json(args.output/"RESULT.json", {"status": "own_admission_passed_localization_quality_pending" if all(checks.values()) else "own_admission_required_guard_failed",
        "checks": checks, "training_names": names, "training_count": admission["training_count"],
        "training_points": admission["training_points"], "training_observations": admission["training_observations"],
        "export": str(args.output/"export"), "dataset": admission["dataset"], "complete_seconds": time.perf_counter()-STARTED,
        "independent_pose_accuracy_claim": False, "counted": False})
    if not all(checks.values()):
        raise AssertionError("An unchanged complete own admission or retained camera guard failed")


if __name__ == "__main__":
    main()


