"""Audit the complete prespecified residency protocol before stage acceptance."""

import argparse
import json
from pathlib import Path
import statistics

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    resource = read(args.validation)
    groups = resource["pairs"]
    checks = {
        "three_resource_pairs": len(groups) == 3,
        "all_numeric_oracles": len(resource["numeric_checks"]) == 14 and all(resource["numeric_checks"].values()),
        "resource_status": resource["status"] == "input_and_resource_guards_passed_training_pending",
        "all212_resource_values_layout_and_geometry_exact": all(len(g["cpu"]["rows"]) == 212
            and g["cpu"]["rows"] == g["cuda"]["rows"] for g in groups),
        "zero_CPU_image_payload": all(g["cuda"]["payload_bytes_by_device"]["cpu"] == 0 for g in groups),
        "resource_budgets": all(g[label]["peak_cuda_reserved_bytes"] <= g[label]["gpu_budget_bytes"]
            and g[label]["peak_rss_bytes"] <= 32*1024**3 for g in groups for label in ["cpu", "cuda"])}
    med_rss = {label: statistics.median(g[label]["rss_after_load_bytes"] for g in groups)
               for label in ["cpu", "cuda"]}
    reduction = 1-med_rss["cuda"]/med_rss["cpu"]
    checks["host_RSS_reduction_at_least20percent"] = reduction >= .2
    times = {"reference": [], "gpu_resident": []}
    records, source_sets = [], []
    localization_sources, training_inputs = [], []
    for seed, root in enumerate(args.pairs):
        pair = read(root/"PAIR_STATE.json")
        evaluation = read(root/"evaluation/EVALUATION.json")
        contract = read(root/"evaluation/EVALUATION_CONTRACT.json")
        visual = read(root/"evaluation/VISUAL_REVIEW.json")
        source_sets.append([row["sha256"] for row in pair["sources"]])
        localization_sources.append(contract["localization_source"]["sha256"])
        checks[f"seed{seed}_training_status"] = (pair["seed"] == seed
            and pair["status"] == "training_completed_pending_evaluation"
            and pair["identical_input_and_sampling"] and pair["iterations"] == 12000)
        checks[f"seed{seed}_source_content_preserved"] = all(
            data_identity(Path(row["path"]))["sha256"] == row["sha256"] for row in pair["sources"])
        expected_order = ["gpu_resident", "reference"] if seed == 1 else ["reference", "gpu_resident"]
        checks[f"seed{seed}_registered_order"] = [run["label"] for run in pair["runs"]] == expected_order
        checks[f"seed{seed}_all24_quality"] = (evaluation["all_frames_included_in_denominator"] == 24
            and evaluation["development_metric_guard_passed"]
            and all(row["mean_PSNR_dB"] >= -.1 and row["mean_SSIM"] >= -.002
                    for row in evaluation["candidate_minus_control"].values())
            and all(evaluation["results"][label]["summary"][region]["count"] == 24
                    for label in times for region in ["full_image", "target"]))
        checks[f"seed{seed}_visual_self_review"] = (visual["passed"]
            and not visual["independent_review"] and (root/"evaluation/SIX_VIEW_COMPARISON.jpg").is_file())
        configs = {label: read(root/label/"CONFIG.json") for label in times}
        states = {label: read(root/label/"STATE.json") for label in times}
        inputs = {label: [r["sha256"] for r in read(root/label/"INPUT_MANIFEST.json")["files"]] for label in times}
        training_inputs.append(inputs["reference"])
        checks[f"seed{seed}_input_and_sampling_identity"] = (inputs["reference"] == inputs["gpu_resident"]
            and states["reference"]["sample_sequence_sha256"] == states["gpu_resident"]["sample_sequence_sha256"])
        parameters = {label: {key: value for key, value in config["parameters"].items()
                             if key not in ("model_path", "data_device")}
                      for label, config in configs.items()}
        checks[f"seed{seed}_only_input_device_changed"] = (parameters["reference"] == parameters["gpu_resident"]
            and configs["reference"]["parameters"]["data_device"] == "cpu"
            and configs["gpu_resident"]["parameters"]["data_device"] == "cuda"
            and all(configs[label]["trim_host_after_load"] for label in times))
        residency = states["gpu_resident"]["scene_timing"]["input_residency"]
        checks[f"seed{seed}_actual_complete_GPU_residency"] = (residency["camera_count"] == 212
            and residency["payload_bytes_by_device"] == {"cpu": 0, "cuda": 1707668480})
        for label in times:
            child, config = states[label], configs[label]
            times[label].append(child["training_call_seconds"])
            checks[f"seed{seed}_{label}_budget_and_model"] = (child["status"] == "completed"
                and child["iteration"] == 12000 and sum(child["sample_counts"].values()) == 12000
                and child["maximum_cuda_reserved_bytes"] <= config["gpu_budget_bytes"]
                and child["maximum_rss_bytes"] <= 32*1024**3
                and child["elapsed_seconds"] <= 1800
                and data_identity(root/label/"model/point_cloud/iteration_12000/point_cloud.ply")["sha256"]
                    == evaluation["results"][label]["model"]["sha256"])
        records.append({"seed": seed, "root": str(root), "pair": data_identity(root/"PAIR_STATE.json"),
            "evaluation": data_identity(root/"evaluation/EVALUATION.json"),
            "visual": data_identity(root/"evaluation/VISUAL_REVIEW.json"),
            "quality_deltas": evaluation["candidate_minus_control"]})
    med_time = {label: statistics.median(values) for label, values in times.items()}
    ratio = med_time["gpu_resident"]/med_time["reference"]
    checks["one_frozen_implementation_for_all_seeds"] = all(value == source_sets[0] for value in source_sets)
    checks["one_frozen_localization_for_all_seeds"] = len(set(localization_sources)) == 1
    checks["one_frozen_training_input_for_all_seeds"] = all(value == training_inputs[0] for value in training_inputs)
    checks["median_training_ratio_at_most1_10"] = ratio <= 1.10
    result = {"status": "accepted_stage_optimization" if all(checks.values()) else "failed_stage_acceptance",
        "mechanism": "P26_INPUT_RESIDENCY", "fixed_seeds": [0, 1, 2], "checks": checks,
        "resource_receipt": data_identity(args.validation), "median_host_RSS_bytes": med_rss,
        "host_RSS_reduction": reduction, "training_seconds": times, "median_training_seconds": med_time,
        "median_training_ratio": ratio, "records": records, "counted": all(checks.values()),
        "count_units": 1 if all(checks.values()) else 0, "same_storage_family_P16_P24_not_counted": True,
        "benefit": "Host RAM pressure reduction at an explicit GPU input-storage cost",
        "total_memory_reduction_claim": False, "training_speed_optimization_claim": False,
        "strict_blind_test": False, "full_route_combination_accepted": False,
        "source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print(json.dumps({key: result[key] for key in ["status", "host_RSS_reduction", "median_training_ratio", "count_units"]}))
    if not all(checks.values()):
        raise ValueError("At least one original residency guard failed")


if __name__ == "__main__":
    main()
