"""Bind preserved primary timing to completed QA and every recorded recovery cost."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = args.pair / "formal_sparse_001"
    side = campaign / "box1/optimized"
    original = read_json(campaign / "FAILURE.json")
    contract = read_json(campaign / "CONTRACT.json")
    outer = read_json(args.pair / "formal_sparse_001_process.json")
    local = side / "localization_recovery_002/LOCALIZATION.json"
    recovery = read_json(side / "localization_recovery_002/RECOVERY_RECEIPT.json")
    evaluation = read_json(side / "evaluation/RESULT.json")
    scale = read_json(side / "scale_consistency/RESULT.json")
    if not original["primary_pipeline_finished"] or original["current_stage"] != "localization":
        raise ValueError("Only completed fresh primary timing followed by this QA failure can be reused")
    if read_json(local)["status"] != "all24_localized" or recovery["resumed_count"]+recovery["newly_completed_count"] != 24:
        raise ValueError("The full24 QA localization gate is incomplete")
    if evaluation["summary"]["full_image"]["rendered_count"] != 24 or scale["selected_scan_denominator"] != 64 or not scale["source_unchanged"]:
        raise ValueError("Full24 rendering and full64 sensor checks remain required")
    raw = Path(contract["raw"])
    unchanged = (raw.stat().st_size, raw.stat().st_mtime_ns) == (contract["raw_bytes"], contract["raw_mtime_ns"])
    if not unchanged:
        raise ValueError("The raw recording identity changed")
    receipts = []
    for name, expected in (("localization_recovery_001", "failed_child"), ("localization_recovery_002", "completed"),
                           ("candidate_evaluation_recovery_001", "completed"), ("candidate_scale_recovery_001", "completed")):
        path = args.pair / (name+"_process.json")
        record = read_json(path)
        if record["status"] != expected:
            raise ValueError("An expected preserved recovery receipt differs: "+name)
        receipts.append({"stage": name, "receipt": str(path), "seconds": record["elapsed_seconds"], "status": record["status"]})
    diagnosis = {"stage": "one_query_NaN_diagnosis", "seconds": 1.5410349,
                 "method": "captured_unified_exec_command_wall_time", "chunk_id": "65738f"}
    seconds = outer["elapsed_seconds"] + sum(row["seconds"] for row in receipts)+diagnosis["seconds"]
    result = {**original, "status": "complete_raw_model_preview_and_QA_comparison_pending",
        "raw_identity_unchanged": unchanged, "raw_to_preview_plus_full_QA_seconds": seconds,
        "localization_path": str(local), "evaluation_path": str(side / "evaluation/RESULT.json"),
        "scale_path": str(side / "scale_consistency/RESULT.json"),
        "recovered_current_campaign_only": True, "original_failure": str(campaign / "FAILURE.json"),
        "original_failed_outer_receipt": str(args.pair / "formal_sparse_001_process.json"),
        "outer_original_seconds": outer["elapsed_seconds"], "recovery_costs": receipts, "diagnostic_cost": diagnosis,
        "primary_timing_kind": "unchanged_uninterrupted_fresh_raw_to_native_assets_and_viewer_readiness",
        "full_QA_timing_kind": "recorded_original_attempt_plus_all_recovery_and_diagnostic_execution_costs",
        "full_QA_is_not_contiguous_operator_wall_time": True,
        "operator_investigation_editing_and_idle_not_counted_as_algorithm_time": True,
        "all_actual_localization_failures_and_retries_included": True,
        "historical_other_campaign_model_feature_or_image_cache_used": False,
        "model_and_matching_algorithms_unchanged_during_QA_recovery": True}
    atomic_json(args.output / "RESULT.json", result)
    print({"primary": result["raw_to_preview_seconds"], "full_including_failures_and_recovery": seconds})


if __name__ == "__main__":
    main()
