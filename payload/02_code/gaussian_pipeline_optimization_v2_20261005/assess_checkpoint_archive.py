"""Keep storage benefits separate from measured compression and native recovery costs."""

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
    for name in ("archive", "recovery", "visual-review", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    archive, recovery, visual = read(args.archive/"RESULT.json"), read(args.recovery/"RESULT.json"), read(args.visual_review)
    checks = {"three_distinct_actual_complete_states": len(archive["rows"]) == 3,
        "all_storage_payload_and_malformed_archive_guards": all(archive["checks"].values()),
        "actual_native_full_state_queue_sampler_and_quality_recovery": all(recovery["checks"].values()),
        "owned_temporary_expansion_removed_after_complete_recovery": recovery["transport"]["expanded_copy_removed"],
        "actual_six_view_visual_self_review": visual["method"] == "actual_render_image_self_review"
            and visual["six_actual_development_views_reviewed"] and not visual["material_target_regression_observed"]}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed_verification",
        "mechanism": "Q76_LOSSLESS_COMPLETE_CHECKPOINT_TRANSPORT_ARCHIVE", "checks": checks,
        "stage_accepted_count": 1 if all(checks.values()) else 0, "full_route_combination_pending": True,
        "deployment": "optional archived recovery artifacts; original evidence checkpoints preserved",
        "bytes_at_rest_including_new_manifest": [{"native_bytes": row["source"]["bytes"], "stored_bytes": row["payload_and_sidecar_bytes"],
            "reduction": row["reduction_including_new_sidecar"], "compression_seconds": row["compression_seconds"],
            "native_read_seconds": row["native_read_seconds"], "direct_gzip_random_read": row["archive_read"]} for row in archive["rows"]],
        "recommended_recovery_read": "verified sequential expansion then unchanged native full-state loader",
        "verified_sequential_transport": recovery["transport"], "all24_quality_delta": recovery["candidate_minus_control"],
        "whole_workspace_disk_reduction_not_claimed": True, "raw_checkpoint_read_speedup_not_claimed": True,
        "receipts": [str(args.archive/"RESULT.json"), str(args.recovery/"RESULT.json"), str(args.visual_review)],
        "assessor_source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    if not all(checks.values()):
        raise AssertionError("A complete-state archived recovery resource or quality guard failed")
    print({"status": result["status"], "stage_accepted_count": result["stage_accepted_count"],
        "minimum_three_actual_storage_reduction": min(row["reduction"] for row in result["bytes_at_rest_including_new_manifest"])}, flush=True)


if __name__ == "__main__":
    main()
