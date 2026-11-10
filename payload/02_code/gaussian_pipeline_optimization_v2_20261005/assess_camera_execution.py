"""Accept independent camera execution gains only after full training and visual review."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resources", "training", "visual-review", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    resources, training, visual = read(args.resources/"RESULT.json"), read(args.training/"RESULT.json"), read(args.visual_review)
    checks = {"three_fresh_paired_full12000_seed_runs": training["status"] == "three_fresh_seed_quality_and_actual_recovery_passed_visual_pending"
            and sorted(row["seed"] for row in training["seed_checks"]) == [0, 1, 2],
        "all_seeds_original_full24_PSNR_SSIM_protection": all(row["all24_quality_guards_passed"] for row in training["seed_checks"]),
        "all1016_inputs_and12000_samples_exact": all(row["input_and_all12000_samples_exact"] for row in training["seed_checks"]),
        "actual_same_prefix_exit86_full_state_and_quality_passed": all(training["same_prefix_recovery"]["checks"].values()),
        "three_independent_resource_trials_each": all(len(row["pairs"]) == 3 and all(p["all_protected"] for p in row["pairs"])
            and row["stage_reduction"] >= .1 for row in resources["results"].values()),
        "actual_three_seed_six_view_and_recovery_visual_self_review": visual.get("method") == "actual_render_image_self_review"
            and visual.get("all_three_seed_six_views_reviewed") is True and visual.get("recovery_six_views_reviewed") is True
            and visual.get("material_target_regression_observed") is False,
        "source_identities_unchanged": training["sources_unchanged"] and resources["sources_unchanged"]}
    for row in training["seed_checks"]:
        checks["seed"+str(row["seed"])+"_complete_training_time_protected"] = row["actual_training_seconds"]["combined"] <= 1.1*row["actual_training_seconds"]["baseline"]
    result = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed_verification", "checks": checks,
        "mechanisms": ["Q58_OPAQUE_RGB_ALPHA_ALLOCATION", "Q59_DIRECT_CALIBRATED_PROJECTION_CONSTANT_REUSE", "Q60_NORMALIZED_RGB_CLAMP_COPY_ELISION"],
        "stage_accepted_count": 3 if all(checks.values()) else 0, "full_route_combination_pending": True,
        "individual_benefits": {name: {key: value[key] for key in ("median", "stage_reduction", "absolute_phase_seconds_saved")}
            for name, value in resources["results"].items()},
        "actual_training_median_seconds": {name: statistics.median(row["actual_training_seconds"][name] for row in training["seed_checks"])
            for name in ("baseline", "combined")}, "all_three_quality_deltas": [row["candidate_minus_control"] for row in training["seed_checks"]],
        "visual_review": visual, "receipts": [str(args.resources/"RESULT.json"), str(args.training/"RESULT.json"), str(args.visual_review)],
        "assessor_source": data_identity(Path(__file__)), "quality_improvement_not_claimed_from_stochastic_deltas": True}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    if not all(checks.values()):
        raise AssertionError("A full actual camera execution acceptance protection failed")
    print({key: result[key] for key in ("status", "stage_accepted_count", "actual_training_median_seconds")}, flush=True)


if __name__ == "__main__":
    main()
