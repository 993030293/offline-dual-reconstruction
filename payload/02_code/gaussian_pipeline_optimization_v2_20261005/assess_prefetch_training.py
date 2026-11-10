"""Consolidate actual loading, training and image-review guards without rerunning them."""

import argparse
import json
from pathlib import Path
import statistics
import sys

V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resources", "smoke", "training", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    resource = read(args.resources)
    training = read(args.training/"STATE.json")
    if training["status"] != "three_seed_training_and_metrics_passed_visual_and_combination_pending":
        raise ValueError("All three fixed-seed training/evaluation pairs must finish first")
    smoke, smoke_ply = read(args.smoke/"STATE.json"), read(args.smoke/"PLY_SERIALIZATION.json")
    checks = {"three_input_and_resource_pairs": all(resource["checks"].values()) and len(resource["pair_checks"]) == 3
        and all(all(row.values()) for row in resource["pair_checks"]),
        "actual1100step_growth_smoke": smoke["status"] == "completed" and smoke["iteration"] == 1100
            and smoke["image_count"] == 1016 and smoke["gaussian_count"] > 75767,
        "smoke_native_serialization_exact": smoke_ply["verify_native"] and all(event["native_byte_oracle"]["identical"] for event in smoke_ply["events"]),
        "all9_training_and_evaluation_processes_complete": len(training["runs"]) == 9
            and all(row["status"] == "completed" and row["returncode"] == 0 for row in training["runs"]),
        "original_seeds_preserved": [row["seed"] for row in training["seed_checks"]] == [0, 1, 2],
        "training_bound_sources_unchanged": all(data_identity(Path(row["path"])) == row for row in training["sources"])}
    summaries = []
    for seed in range(3):
        pair = args.training/f"seed{seed}"
        reference, candidate = [read(pair/label/"STATE.json") for label in ("reference", "prefetch")]
        evaluation = read(pair/"evaluation/EVALUATION.json")
        review = read(pair/"evaluation/VISUAL_REVIEW.json")
        adapter = read(pair/"prefetch/PREFETCH_ADAPTER.json")
        control_input, candidate_input = [read(pair/label/"INPUT_MANIFEST.json") for label in ("reference", "prefetch")]
        row = training["seed_checks"][seed]
        complete = all(s["status"] == "completed" and s["iteration"] == 12000 and s["image_count"] == 1016 for s in (reference, candidate))
        checks[f"seed{seed}_complete_same_inputs_and_sample_sequence"] = (complete
            and reference["sample_sequence_sha256"] == candidate["sample_sequence_sha256"]
            and [f["sha256"] for f in control_input["files"]] == [f["sha256"] for f in candidate_input["files"]])
        checks[f"seed{seed}_original_full_and_target_metric_guards"] = evaluation["development_metric_guard_passed"] and row["all24_metric_guards_passed"]
        checks[f"seed{seed}_training_time_protected"] = candidate["training_call_seconds"]/reference["training_call_seconds"] <= 1.1
        checks[f"seed{seed}_actual_bounded_prefetch_used"] = (adapter["status"] == "actual_training_completed"
            and len(adapter["prefetch"]) == 1 and adapter["prefetch"][0]["consumed"] == 1016
            and adapter["prefetch"][0]["maximum_pending"] <= 8)
        checks[f"seed{seed}_actual_six_view_self_review"] = (review["type"] == "actual_image_self_review"
            and review["independent_reviewer"] is False and review["rows_reviewed"] == [734, 1457, 2136, 2806, 3440, 4319]
            and review["acquisition_view_protection_passed"] and not review["new_obvious_target_holes"]
            and not review["new_obvious_target_edge_or_handle_failure"])
        for label in ("reference", "prefetch"):
            ply = read(pair/label/"PLY_SERIALIZATION.json")
            process = read(args.training/f"seed{seed}_{label}_process.json")
            checks[f"seed{seed}_{label}_native_PLY_and_memory_protection"] = (ply["verify_native"]
                and len(ply["events"]) > 0 and all(event["native_byte_oracle"]["identical"] for event in ply["events"])
                and process["peak_tree_RSS_bytes"] <= 32*1024**3)
        summaries.append({"seed": seed, "reference_seconds": reference["training_call_seconds"],
            "prefetch_seconds": candidate["training_call_seconds"], "candidate_minus_control": row["candidate_minus_control"],
            "actual_six_view_self_review": str(pair/"evaluation/VISUAL_REVIEW.json")})
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed_verification",
        "mechanism": "Q31_BOUNDED_TRAINING_IMAGE_PREFETCH", "checks": checks,
        "median_Scene_load_seconds": resource["median_Scene_load_seconds"], "Scene_stage_reduction": resource["stage_time_reduction"],
        "median_training_call_seconds": {"reference": statistics.median(row["reference_seconds"] for row in summaries),
            "prefetch": statistics.median(row["prefetch_seconds"] for row in summaries)},
        "seeds": summaries, "accepted_scope": "same1016 CPU RGB images, four decoding workers, eight pending images, original serial native cameras and training",
        "counted_stage_only": all(checks.values()), "counted_final_round2": False,
        "checkpoint_recovery_combination_validated": False, "free_flight_combination_validated": False,
        "quality_improvement_claim": False, "GT_accuracy_claim": False,
        "inputs": {"resources": str(args.resources), "smoke": str(args.smoke), "training": str(args.training)},
        "assessor_source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print(json.dumps(result))
    if not all(checks.values()):
        raise AssertionError("A prefetch stage guard failed")


if __name__ == "__main__":
    main()
