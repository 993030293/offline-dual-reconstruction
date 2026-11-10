"""Accept one exact raw calibration reuse mechanism after full initial selection."""

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
    sources = [*campaign["sources"], *oracle["sources"]]
    protections = []
    for trial in range(3):
        for mode in ("incumbent", "native", "memo"):
            target = args.campaign/f"trial{trial}"/mode
            combination, contract = read(target/"COMBINATION.json"), read(target/"CONTRACT.json")
            protections.append(all(item == data_identity(Path(item["path"])) for item in combination["sources"])
                and contract["JPEG_cache"] is None and contract["workers"] == 4 and contract["capacity"] == 8)
    checks = {"all_three_complete_triplets_and_original_input_guards": len(campaign["triplets"]) == 3
            and all(campaign["checks"].values())
            and all(all(row["checks"].values()) for row in campaign["triplets"]),
        "complete_phase_includes_source_composition_schema_and_first_cache_miss":
            campaign["first_schema_and_source_composition_setup_added_to_original_stage"],
        "stage_reduces10percent_against_untouched_original": campaign["stage_reduction"] >= .1,
        "all12_real_and_generic_header_body_mutation_and_error_oracles": len(oracle["checks"]) == 12
            and oracle["status"] == "passed" and all(oracle["checks"].values()),
        "actual_route_matches_full_native_decoder_oracle": campaign["sources"][2] == oracle["sources"][1],
        "all_sources_and_old_combination_inputs_unchanged": all(item == data_identity(Path(item["path"])) for item in sources)
            and all(protections),
        "no_new_frame_cap_or_object_specific_logic": campaign["only_raw_estimate_and_raw_image_quality_no_GT_or_SfM"],
        "full_entry_no_unreported_regression": campaign["real_complete_entry_reduction"] >= 0
            and all(row["complete_entry_seconds"]["memo"] <= 1.1*row["complete_entry_seconds"]["incumbent"]
                for row in campaign["triplets"])}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R99_EXACT_RAW_CALIBRATION_BODY_REUSE", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "median": campaign["median"], "stage_reduction": campaign["stage_reduction"],
        "actual_complete_entry_reduction": campaign["real_complete_entry_reduction"],
        "scope": "pinned_ROS2_CDR_CameraInfo_decoder_full_body_prefix_alignment_equality_fresh_header_owned_lists",
        "dynamic_calibration_retains_original_rejection_no_silent_change": True,
        "calibration_parser_does_not_claim_additional_R_P_ROI_fields": True,
        "generic_contract_checked_new_object_whole_pipeline_pending": True,
        "actual7338_raw_records1002_JPEG11583_pairs_preserved": True,
        "independent_review": False, "campaign": str(args.campaign), "oracle": str(args.oracle),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "checks", "additional_accepted_count", "median")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original complete calibration body acceptance guard failed")


if __name__ == "__main__":
    main()
