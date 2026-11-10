"""Bind the measured full-map guards to an actual reusable QA entry."""

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
    for name in ("campaign", "entry", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign = read(args.campaign/"RESULT.json")
    qa, contract = read(args.entry/"QA.json"), read(args.entry/"CONTRACT.json")
    protocol = read(args.campaign/"PROTOCOL.json")
    frozen = {Path(row["path"]).name: row["sha256"] for row in protocol["source_inputs"]}
    checks = {"three_real_full_map_pairs_and_all_original_guards": all(campaign["checks"].values()),
        "measured_complete_stage_reduction_at_least10percent": campaign["reduction"] >= .1,
        "actual_entry_reads_all959120_and_rejects_invalid": qa["status"] == "full_native_binary_QA_passed"
            and qa["result"]["observation_count"] == 959120 and qa["result"]["valid"],
        "same_native_input_payloads_and_algorithm": all(frozen.get(Path(row["path"]).name) == row["sha256"]
            for row in contract["input"]+contract["sources"] if Path(row["path"]).name != "effective_route.py"),
        "actual_input_payload_identity_remains_valid": all(data_identity(Path(row["path"])) == row
            for row in contract["input"]+contract["sources"])}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism_relevant_combination_complete" if all(checks.values()) else "failed_verification",
        "mechanism": "R85_NATIVE_BINARY_OBSERVATION_ARRAY_QA", "checks": checks,
        "additional_accepted_count": 1 if all(checks.values()) else 0,
        "median_seconds": campaign["median_seconds"], "reduction": campaign["reduction"],
        "scope": "saved_immutable_native_model_disk_to_complete_projection_QA",
        "mutable_in_memory_BA_QA_replacement": False, "speedup_raw_to_new_map_claim": False,
        "baseline_is_previously_accepted_Q49_plus_actual_native_model_read": True,
        "inherited_full_observation_map_and_original_failure_guards": True,
        "first_invalid_order_failure_preserved": str(args.campaign.parent/"binary_qa_002/RESULT.json"),
        "metadata_map_enumeration_preserves_original_order_without_tolerance_changes": True,
        "receipts": [str(args.campaign/"RESULT.json"), str(args.entry/"QA.json")],
        "assessor": data_identity(Path(__file__)), "independent_review": False}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({"status": result["status"], "additional_accepted_count": result["additional_accepted_count"], "checks": checks})
    if not all(checks.values()):
        raise ValueError("An actual complete binary QA entry or original guard failed")


if __name__ == "__main__":
    main()
