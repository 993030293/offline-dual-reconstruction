"""Consolidate the prespecified paired seeds and actual display verification."""

import argparse
import json
from pathlib import Path
import statistics

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks, records = {}, []
    expected_sources = expected_inputs = expected_evaluation = None
    for seed, run in enumerate(args.runs):
        pair = read(run / "PAIR_STATE.json")
        evaluation = read(run / "evaluation/EVALUATION.json")
        evaluation_contract = read(run / "evaluation/EVALUATION_CONTRACT.json")
        visual = read(run / "VISUAL_REVIEW.json")
        sources = sorted((Path(row["path"]).name, row["sha256"]) for row in pair["frozen_training_sources"])
        if expected_sources is None:
            expected_sources = sources
        protocol = {key: evaluation_contract[key] for key in ["localization_source", "K", "images", "frame_count"]}
        if expected_evaluation is None:
            expected_evaluation = protocol
        checks[f"seed{seed}_same_sources_and_evaluation"] = sources == expected_sources and protocol == expected_evaluation
        checks[f"seed{seed}_identity_and_completion"] = (pair["seed"] == seed and pair["candidate"] == "gsplat"
            and pair["status"] == "training_completed_pending_evaluation" and pair["control_iterations"] == 10000
            and pair["candidate_iterations"] == 12000 and pair["projection"] == "full")
        expected_order = ["gsplat", "reference"] if seed == 1 else ["reference", "gsplat"]
        checks[f"seed{seed}_prespecified_order"] = [row["label"] for row in pair["runs"]] == expected_order
        deltas = evaluation["candidate_minus_control"]
        checks[f"seed{seed}_all_quality_guards"] = evaluation["development_metric_guard_passed"] and all(
            row["mean_PSNR_dB"] >= -.1 and row["mean_SSIM"] >= -.002 for row in deltas.values())
        checks[f"seed{seed}_actual_six_view_self_review"] = (visual["new_obvious_target_hole_seen"] is False
            and visual["self_reviewed_image"] == "evaluation/SIX_VIEW_COMPARISON.jpg" and visual["independent_review"] is False)
        states = {}
        for label, iterations in [("reference", 10000), ("gsplat", 12000)]:
            state = read(run / label / "STATE.json")
            inputs = read(run / label / "INPUT_MANIFEST.json")
            identities = [(row["path"], row["bytes"], row["sha256"]) for row in inputs["files"]]
            if expected_inputs is None:
                expected_inputs = identities
            checks[f"seed{seed}_{label}_fixed_inputs"] = inputs["image_count"] == 212 and identities == expected_inputs
            checks[f"seed{seed}_{label}_complete_exposure"] = (state["status"] == "completed" and state["iteration"] == iterations
                and len(state["sample_counts"]) == 212 and sum(state["sample_counts"].values()) == iterations)
            summary = evaluation["results"][label]["summary"]
            checks[f"seed{seed}_{label}_full_evaluation_denominator"] = (
                evaluation["all_frames_included_in_denominator"] == 24 and all(summary[region]["count"] == 24 for region in ["full_image", "target"]))
            model = run / label / f"model/point_cloud/iteration_{iterations}/point_cloud.ply"
            checks[f"seed{seed}_{label}_evaluated_model_identity"] = data_identity(model)["sha256"] == evaluation["results"][label]["model"]["sha256"]
            states[label] = {"training_seconds": state["training_call_seconds"], "gaussians": state["gaussian_count"],
                "mean_samples": state["mean_samples"], "minimum_samples": min(state["sample_counts"].values()),
                "maximum_samples": max(state["sample_counts"].values()), "maximum_cuda_reserved_bytes": state["maximum_cuda_reserved_bytes"],
                "maximum_rss_bytes": state["maximum_rss_bytes"], "model_bytes": model.stat().st_size}
        records.append({"seed": seed, "run": str(run), "models": states, "quality_deltas": deltas,
                        "paired_training_reduction": 1-states["gsplat"]["training_seconds"]/states["reference"]["training_seconds"]})
    export = read(args.viewer / "EXPORT_RECEIPT.json")
    browser = read(args.viewer / "BROWSER_VALIDATION.json")
    viewer_visual = read(args.viewer / "VISUAL_REVIEW.json")
    checks["actual_freeview_navigation_and_projection"] = browser["status"] == "passed_native_PLY_and_free_navigation" and all(row["passed"] for row in browser["checks"])
    checks["actual_freeview_visual_self_review"] = viewer_visual["new_obvious_target_hole_seen"] is False and viewer_visual["independent_review"] is False
    checks["viewer_exports_same_seed0_models"] = export["seed"] == 0 and all(
        row["source"]["sha256"] == read(args.runs[0] / "evaluation/EVALUATION.json")["results"][row["case"]]["model"]["sha256"]
        and row["source"]["sha256"] == row["full"]["sha256"] for row in export["assets"])
    medians = {label: statistics.median(row["models"][label]["training_seconds"] for row in records) for label in ["reference", "gsplat"]}
    reduction = 1-medians["gsplat"]/medians["reference"]
    checks["three_seed_median_at_least10percent_faster"] = reduction >= .1
    passed = all(checks.values())
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "accepted_stage_optimization" if passed else "failed_cohort_guards",
        "mechanism": "P6_GSPLAT_BACKEND", "configuration": "gsplat1.5.3_12000steps_vs_INRIA10000steps_fixed212_fullK",
        "checks": checks, "records": records, "median_training_seconds": medians,
        "median_training_reduction": reduction, "counted": passed, "count_units": 1 if passed else 0,
        "same10k_candidate_remains_rejected": True, "strict_blind_test": False,
        "freeview_validation_seed": 0, "full_route_combination_accepted": False,
        "independent_geometry_or_pose_accuracy_claim": False,
        "cost_boundary": "prebuilt_environment_training_call_includes_scene_load_and_model_saves_excludes_initial_backend_build",
        "source": data_identity(Path(__file__)), "viewer_browser_receipt": data_identity(args.viewer / "BROWSER_VALIDATION.json")}
    atomic_json(args.output / "ACCEPTANCE.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise ValueError("The unchanged cohort guards did not all pass")


if __name__ == "__main__":
    main()
