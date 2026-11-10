"""Accept exact gray moments only after all cold raw outputs and cache bytes agree."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "oracle", "boundary", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, oracle, boundary = [read(folder/"RESULT.json") for folder in (args.campaign, args.oracle, args.boundary)]
    verified = []
    before = time.perf_counter()
    for trial in range(3):
        folder = args.campaign/f"trial{trial}"
        left, right = folder/"native_quality", folder/"histogram_quality"
        names = sorted(path.name for path in left.iterdir() if path.is_file())
        checked = len(names) == 7338 and names == sorted(path.name for path in right.iterdir() if path.is_file())
        checked = checked and all((left/name).read_bytes() == (right/name).read_bytes() for name in names)
        verified.append({"trial": trial, "all7338_complete_original_cache_records_bytes_exact": checked})
    cache_verification_seconds = time.perf_counter()-before
    lifecycle = []
    for trial in range(3):
        for mode in ("native", "histogram"):
            folder = args.campaign/f"trial{trial}"/mode
            item, combination, contract = read(folder/"HISTOGRAM.json"), read(folder/"COMBINATION.json"), read(folder/"CONTRACT.json")
            lifecycle.append(item["cache_did_not_exist_before_run"] and item["trace"]["aliases_restored"]
                and item["same_current_Q78_Q50_Q53_P18_R99_JPEG_cache_off"]
                and contract["JPEG_cache"] is None and contract["workers"] == 4 and contract["capacity"] == 8
                and all(source == data_identity(Path(source["path"])) for source in [*item["sources"], *combination["sources"]]))
    checks = {"all_three_actual_complete_cold_raw_pairs_and_original_denominators":
            campaign["status"] == "all_actual_cold_quality_guards_passed_acceptance_pending"
            and len(campaign["pairs"]) == 3 and all(campaign["checks"].values())
            and all(all(row["checks"].values()) for row in campaign["pairs"]),
        "complete_cold_quality_phase_reduces10percent_with_all_new_lifecycle":
            campaign["reductions"]["complete_cold_quality_stage_seconds"] >= .1
            and campaign["complete_quality_phase_includes_source_setup_raw_IO_decode_workers_first_cache_write_and_close"],
        "all_original_cache_payloads_exact_and_all_six_lifecycles_protected": all(lifecycle)
            and all(row["all7338_complete_original_cache_records_bytes_exact"] for row in verified),
        "original_generic277_complete_fields_errors_and_actual128_pilot_passed":
            oracle["status"] == "pilot_passed_full_entry_pending" and len(oracle["checks"]) == 6
            and all(oracle["checks"].values()),
        "real_oversized_image_routed_to_original_before_histogram": boundary["status"] == "passed"
            and len(boundary["checks"]) == 4 and all(boundary["checks"].values()),
        "actual_source_bound_to_both_original_complete_oracles_and_unchanged":
            campaign["sources"][2] == oracle["sources"][1] == boundary["sources"][1]
            and all(source == data_identity(Path(source["path"])) for source in [*campaign["sources"], *oracle["sources"], *boundary["sources"]]),
        "whole_median_and_all_each_entry_cost_protected": campaign["reductions"]["whole_seconds"] >= 0
            and all(row["whole_seconds"]["histogram"] <= 1.1*row["whole_seconds"]["native"] for row in campaign["pairs"]),
        "no_quality_policy_pixel_feature_frame_or_object_specific_selection_change":
            campaign["same_current_Q78_Q50_Q53_P18_R99_and_original_formula_and_policy"]
            and campaign["raw_quality_and_coarse_estimate_only_no_GT_or_SfM"]}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R104_EXACT_GRAY_HISTOGRAM_QUALITY_MOMENTS", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "median": campaign["median"], "reductions": campaign["reductions"],
        "additional_formal_cache_byte_verification_seconds": cache_verification_seconds, "cache_verification": verified,
        "scope": "pinned_OpenCV4_13_uint8_BGR_up_to2power24_pixels_full_gray_brightness_and_clipping_moments_only",
        "cold_first_score_benefit_existing_score_cache_hits_do_not_recompute": True,
        "no_new_geometry_training_or_object_quality_claim": True, "new_object_whole_pipeline_pending": True,
        "independent_review": False, "campaign": str(args.campaign), "oracle": str(args.oracle), "boundary": str(args.boundary),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "additional_accepted_count", "checks", "median")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original complete cold quality acceptance guard failed")


if __name__ == "__main__":
    main()
