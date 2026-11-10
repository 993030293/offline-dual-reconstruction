"""Accept exact final artifact conversion after three complete native evaluations."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, oracle = read(args.campaign/"RESULT.json"), read(args.oracle/"RESULT.json")
    records = [*campaign["sources"], *campaign["inputs"], *oracle["sources"]]
    oracle_keys = {"all21_shapes_layouts_256_values_tie_neighbors_and_extremes_exact",
        "torch.float16_preselected_original_exact", "torch.float16_native_route",
        "torch.float64_preselected_original_exact", "torch.float64_native_route", "CPU_original_route_exact",
        "nonfinite_preselected_original_pixels_and_warning_preserved", "explicit_budget_preselects_original_without_pixel_drop",
        "global_evaluation_restored_after_error", "all_rng_states_preserved", "actual_CPU_transfer_dtype_and_bytes_are_uint8"}
    calls = [read(args.campaign/f"trial{trial}"/mode/"pair/evaluation/ARTIFACTS.json")
        for trial in range(3) for mode in ("native", "device")]
    checks = {"all_three_complete_original24_metrics_and18_artifact_guards": len(campaign["pairs"]) == 3
            and all(campaign["checks"].values()) and all(all(row["checks"].values()) for row in campaign["pairs"]),
        "all11_named_format_rounding_layout_warning_and_cleanup_oracles": set(oracle["checks"]) == oracle_keys
            and oracle["status"] == "passed" and all(oracle["checks"].values()),
        "complete_artifact_stage_reduces10percent": campaign["reductions"]["complete_artifact_seconds"] >= .1,
        "actual_returned_tensor_transfer_reduces70percent": campaign["reductions"]["actual_device_to_host_tensor_bytes"] >= .7,
        "whole_evaluation_cost_not_regressed": campaign["reductions"]["complete_makespan_seconds"] >= 0,
        "same_native_conversion_oracle_and_complete_evaluator_sources": campaign["sources"][2] == oracle["sources"][1]
            and campaign["sources"][4] == oracle["sources"][2],
        "all_sources_models_and_actual_entry_sources_unchanged": all(row == data_identity(Path(row["path"])) for row in records)
            and all(all(row == data_identity(Path(row["path"])) for row in cost["sources"]) for cost in calls),
        "training_metrics_and_native_render_unchanged": all(not cost["training_or_rendering_math_changed"] for cost in calls)}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R100_EXACT_FINAL_ARTIFACT_DEVICE_QUANTIZATION", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "median": campaign["median"], "reductions": campaign["reductions"],
        "actual_transferred_tensor_bytes_not_peak_memory_or_FPS": True,
        "scope": "pinned_Torch2.7.1_NumPy1.26.4_finite_CUDA_float32_CHW_RGB_final_display_artifacts_only",
        "all24_fixed_C212_metrics_and18_artifact_pixels_exact": True,
        "current1016_new_map_full_evaluation_not_yet_run": True,
        "generic_contract_checked_new_object_whole_pipeline_pending": True,
        "independent_review": False, "campaign": str(args.campaign), "oracle": str(args.oracle),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "checks", "additional_accepted_count", "median")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original final display artifact acceptance guard failed")


if __name__ == "__main__":
    main()
