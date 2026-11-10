"""Bind lossless codec costs, complete recovery and the inspected RGB result."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from zstd_checkpoint import verified_expand


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "recovery", "entry", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    campaign, recovery = read(args.campaign/"RESULT.json"), read(args.recovery/"RESULT.json")
    visual = read(args.recovery/"evaluation/VISUAL_REVIEW.json")
    entry = read(args.entry/"ARCHIVE.json")
    expanded = args.output/"verified_native.pth"
    verified = verified_expand(args.entry/"checkpoint.archive", expanded)
    checks = {
        "three_native_sizes_three_paired_costs_and_full_payloads": all(campaign["checks"].values()),
        "actual_model_optimizer_rng_queue_and_all24_quality": all(recovery["checks"].values()),
        "six_actual_RGB_views_inspected_and_no_new_regression": visual["all_six_views_inspected"]
            and visual["no_new_visible_regression_in_inspected_views"],
        "actual_entry_uses_complete_original_state": entry["native_receipt"]["status"] == "complete"
            and verified["native_payload"] == entry["native_payload"],
        "current_codec_source_matches_measured_source": entry["decoder_source"] == data_identity(Path(__file__).with_name("zstd_checkpoint.py"))
    }
    expanded.unlink()
    result = {"status": "accepted_one_new_mechanism_recovery_and_combination_complete" if all(checks.values()) else "failed_verification",
        "mechanism": "R84_EXISTING_SYSTEM_ZSTD_COMPLETE_STATE_TRANSPORT",
        "additional_accepted_count": int(all(checks.values())), "checks": checks,
        "campaign": str(args.campaign), "recovery": str(args.recovery), "entry": str(args.entry),
        "quality_gain_claim": False, "new_training_algorithm": False,
        "native_checkpoint_loader_and_algorithm_identity_unchanged": True,
        "assessor": data_identity(Path(__file__)), "independent_review": False}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({"status": result["status"], "additional_accepted_count": result["additional_accepted_count"], "checks": checks})
    if not all(checks.values()):
        raise ValueError("A complete-state codec acceptance requirement failed")


if __name__ == "__main__":
    main()
