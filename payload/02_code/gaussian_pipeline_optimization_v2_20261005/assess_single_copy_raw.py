"""Accept direct payload streaming only with three complete exact raw-entry comparisons."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "validation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, validation = read(args.campaign/"RESULT.json"), read(args.validation/"RESULT.json")
    checks = {"three_actual_complete_entry_pairs": len(campaign["pairs"]) == 3,
        "original_complete_scoring_phase_benefit_at_least10pct": campaign["scoring_wall_reduction"] >= .1,
        "all7338_decisions1002_JPEG11583_pairs_and_odometry_exact": all(row["all_protected"] for row in campaign["pairs"]),
        "all_complete_entry_time_protected": all(row["complete_time_protected"] for row in campaign["pairs"]),
        "all7338_native_payload_metadata_and_twelve_boundary_cases_passed": validation["status"] == "lifecycle_passed"
            and len(validation["checks"]) == 12 and all(validation["checks"].values()),
        "native_stream_reader_and_campaign_unchanged": campaign["sources_unchanged"]}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed_verification",
        "mechanism": "Q78_SINGLE_COPY_INDEXED_MCAP_PAYLOAD_STREAM", "checks": checks,
        "stage_accepted_count": 1 if all(checks.values()) else 0, "full_route_combination_pending": True,
        "median": campaign["median"], "stage_reduction": campaign["scoring_wall_reduction"],
        "absolute_scoring_seconds_saved": campaign["absolute_scoring_seconds_saved"],
        "warm_quality_cache_and_serial_JPEG_native_export": True, "cold_uncached_quality_cost_not_claimed": True,
        "selected_frame_count_and_units_unchanged": True, "receipt_paths": [str(args.campaign/"RESULT.json"), str(args.validation/"RESULT.json")],
        "assessor_source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    if not all(checks.values()):
        raise AssertionError("An actual single-copy raw speed or output protection failed")
    print({key: result[key] for key in ("status", "median", "stage_reduction")}, flush=True)


if __name__ == "__main__":
    main()
