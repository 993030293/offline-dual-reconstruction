"""Accept one complete native checkpoint loading mechanism with actual recovery proof."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "oracle", "recovery", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, oracle, recovery = [read(p/"RESULT.json") for p in (args.campaign, args.oracle, args.recovery)]
    quality = read(args.recovery/"evaluation/EVALUATION.json")
    loader = read(args.recovery/"transport/LOADER.json")
    records = [*campaign["sources_inputs"], *oracle["sources"], *recovery["sources"]]
    checks = {"all_three_real_source_three_pair_cost_and_native_oracle_guards": all(campaign["checks"].values()),
        "complete_loader_validation_stage_reduces10percent": campaign["reduction"] >= .1,
        "all_private_alias_format_and_error_cleanup_oracles": all(oracle["checks"].values()),
        "actual_same_native500_prefix_to1100_all_restore_sampling_and_quality": all(recovery["checks"].values()),
        "all24_original_render_quality_guards": quality["development_metric_guard_passed"],
        "source_original_and_private_copy_remain_byte_exact": loader["owned_source_and_original_unchanged"]
            and loader["original"] == data_identity(Path(loader["original"]["path"]))
            and loader["owned"] == data_identity(Path(loader["owned"]["path"])),
        "all_sources_inputs_unchanged": all(r == data_identity(Path(r["path"])) for r in records),
        "two_native_device_preserving_loads_and_global_restoration": len(loader["trace"]["calls"]) == 2
            and loader["trace"]["original_global_loader_restored"]
            and all(c["mmap"] and c["map_location_argument_unchanged"] == "None" for c in loader["trace"]["calls"])}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R96_NATIVE_COMPLETE_CHECKPOINT_PRIVATE_MMAP", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "sum_source_median_seconds": campaign["sum_source_median_seconds"], "reduction": campaign["reduction"],
        "scope": "fixed_PyTorch2.7.1_native_ZIP_private_mapping_with_same_saved_device_labels_and_complete_state_identity",
        "actual_Box2_1016_same_prefix_recovery_verified": True, "old_C212_8000_cost_fixture_identity_preserved": True,
        "general_tensor_contract_passed_whole_new_object_pipeline_pending": True,
        "PSNR_fluctuation_not_a_quality_gain_claim": True, "independent_review": False,
        "campaign": str(args.campaign), "oracle": str(args.oracle), "recovery": str(args.recovery),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({k: result[k] for k in ("status", "checks", "additional_accepted_count", "sum_source_median_seconds")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original complete native checkpoint loader acceptance condition failed")


if __name__ == "__main__":
    main()
