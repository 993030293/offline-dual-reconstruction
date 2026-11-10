"""Aggregate frozen three-seed quality, resource and actual viewer evidence."""

import argparse
import json
from pathlib import Path
import statistics

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks, records = {}, []
    baseline_inputs = baseline_sources = baseline_eval = None
    for seed in (0, 1, 2):
        folder = args.root/f"p17_target_density_pair_s{seed}_001"
        pair = read(folder/"PAIR_STATE.json")
        evaluation = read(folder/"evaluation/EVALUATION.json")
        visual = read(folder/"evaluation/VISUAL_REVIEW.json")
        sources = [r["sha256"] for r in pair["sources"]]
        if baseline_sources is None:
            baseline_sources = sources
        checks[f"seed{seed}_frozen_sources"] = sources == baseline_sources
        checks[f"seed{seed}_completion"] = (pair["seed"] == seed and pair["status"] == "training_completed_pending_evaluation"
            and pair["candidate"] == "P17_TARGET_DENSITY" and pair["iterations"] == 12000 and pair["multiplier"] == 2)
        checks[f"seed{seed}_order"] = [r["label"] for r in pair["runs"]] == (["target_density", "reference"] if seed == 1 else ["reference", "target_density"])
        delta = evaluation["candidate_minus_control"]
        checks[f"seed{seed}_quality"] = (evaluation["development_metric_guard_passed"] and delta["target"]["mean_PSNR_dB"] >= .15
            and delta["full_image"]["mean_PSNR_dB"] >= -.1 and min(delta[r]["mean_SSIM"] for r in ("full_image", "target")) >= -.002)
        checks[f"seed{seed}_capture_review"] = visual["status"] == "six_capture_views_passed_free_views_pending" and visual["review_type"] == "visual_self_review"
        models = {}
        sample_sha = None
        for label in ("reference", "target_density"):
            state = read(folder/label/"STATE.json")
            inputs = read(folder/label/"INPUT_MANIFEST.json")
            identity = [(r["bytes"], r["sha256"]) for r in inputs["files"]]
            if baseline_inputs is None:
                baseline_inputs = identity
            checks[f"seed{seed}_{label}_input"] = identity == baseline_inputs and inputs["image_count"] == 212 and inputs["input_version"] == "C212_frozen"
            checks[f"seed{seed}_{label}_training"] = (state["status"] == "completed" and state["projection"] == "full"
                and state["iteration"] == 12000 and len(state["sample_counts"]) == 212 and sum(state["sample_counts"].values()) == 12000)
            if sample_sha is None:
                sample_sha = state["sample_sequence_sha256"]
            checks[f"seed{seed}_{label}_sampling"] = sample_sha == state["sample_sequence_sha256"]
            result = evaluation["results"][label]
            eval_identity = [(r["image_name"], r["target_window"], r["quality_eligible"]) for r in result["rows"]]
            if baseline_eval is None:
                baseline_eval = eval_identity
            checks[f"seed{seed}_{label}_all24"] = (eval_identity == baseline_eval and len(eval_identity) == 24
                and evaluation["all_frames_included_in_denominator"] and not evaluation["strict_blind_test"])
            model = data_identity(folder/label/"model/point_cloud/iteration_12000/point_cloud.ply")
            checks[f"seed{seed}_{label}_evaluated_model"] = model["sha256"] == result["model"]["sha256"]
            models[label] = {"training_seconds": state["training_call_seconds"], "gaussians": state["gaussian_count"],
                "mean_samples": state["mean_samples"], "model": model,
                "maximum_cuda_reserved_bytes": state["maximum_cuda_reserved_bytes"], "maximum_rss_bytes": state["maximum_rss_bytes"]}
        time_ratio = models["target_density"]["training_seconds"]/models["reference"]["training_seconds"]
        count_ratio = models["target_density"]["gaussians"]/models["reference"]["gaussians"]
        checks[f"seed{seed}_cost"] = time_ratio <= 1.25 and count_ratio <= 1.5 and pair["resource_ratio_guard_passed"]
        records.append({"seed": seed, "models": models, "delta": delta, "time_ratio": time_ratio, "gaussian_ratio": count_ratio,
            "pair": data_identity(folder/"PAIR_STATE.json"), "evaluation": data_identity(folder/"evaluation/EVALUATION.json")})
    median_target_gain = statistics.median(r["delta"]["target"]["mean_PSNR_dB"] for r in records)
    checks["three_seed_median_target_gain"] = median_target_gain >= .2
    viewer = args.root/"p17_target_density_pair_s0_001/viewer"
    browser = read(viewer/"BROWSER_VALIDATION.json")
    visual = read(viewer/"VISUAL_REVIEW.json")
    exported = read(viewer/"EXPORT_RECEIPT.json")
    checks["real_navigation_and_projection"] = len(browser["checks"]) >= 47 and all(r["passed"] for r in browser["checks"]) and not browser["errors"]
    checks["free_view_visual_self_review"] = visual["passed"] and visual["review_type"] == "visual_self_review"
    checks["same_exported_models"] = all(r["source"]["sha256"] == records[0]["models"][r["case"]]["model"]["sha256"] for r in exported["assets"])
    passed = all(checks.values())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/"ACCEPTANCE.json", {"status": "accepted_stage_optimization" if passed else "failed_verification",
        "mechanism": "P17_TARGET_DENSITY", "checks": checks, "records": records, "counted": passed,
        "median_target_PSNR_gain_dB": median_target_gain,
        "median_time_ratio": statistics.median(r["time_ratio"] for r in records),
        "median_gaussian_ratio": statistics.median(r["gaussian_ratio"] for r in records),
        "evaluation_role": "previously_viewed_C212_development", "independent_geometry_accuracy_claim": False,
        "freeview_reference_available": False, "new_route_combination_accepted": False,
        "limitations": "Existing off-path floating splats and insufficiently observed surfaces remain; ROI belongs only to C212 metric world",
        "entrypoint": "train_case.py --gsplat-build <verified-build> --target-density-roi <same-world-frozen-ROI> --iterations 12000",
        "fallback": "omit target-density-roi; retain the accepted P6 configuration",
        "source": data_identity(Path(__file__))})
    print(json.dumps({"passed": passed, "checks": checks, "median_target_PSNR_gain_dB": median_target_gain}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
