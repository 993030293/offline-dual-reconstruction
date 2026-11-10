"""Accept one complete DAG mechanism with output, overlap and lifecycle evidence."""

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
    for name in ("campaign", "failure", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    proof, failure = read(args.campaign/"RESULT.json"), read(args.failure/"RESULT.json")
    rows = []
    for trial in range(3):
        pair = args.campaign/f"trial{trial}"
        results = {mode: read(pair/mode/"RESULT.json") for mode in ("serial", "parallel")}
        protocols = {mode: read(pair/mode/"PROTOCOL.json") for mode in results}
        overlap = {}
        for mode, result in results.items():
            tasks = list(result["tasks"].values())
            overlap[mode] = min(t["end_seconds"] for t in tasks)-max(t["start_seconds"] for t in tasks)
        artifacts = {}
        for mode in results:
            evaluation = read(pair/mode/"pair/evaluation/EVALUATION.json")
            archive = read(pair/mode/"archive/ARCHIVE.json")
            verified = read(pair/mode/"archive/VERIFIED.json")
            artifacts[mode] = {"metrics": {k: v["rows"] for k, v in evaluation["results"].items()},
                "complete_payload": archive["native_payload"], "verified_payload": verified["native_payload"],
                "all24": evaluation["all_frames_included_in_denominator"] == 24 and evaluation["development_metric_guard_passed"]}
        row_checks = {
            "actual_parallel_overlap_and_serial_dependency": overlap["parallel"] > 0 and overlap["serial"] <= 0,
            "same_frozen_sources_inputs_and_task_graph": all(protocols["serial"][k] == protocols["parallel"][k]
                for k in ("sources", "input", "DAG", "input_version")),
            "all24_metrics_and_verified_complete_state_equal": artifacts["serial"] == artifacts["parallel"] and artifacts["serial"]["all24"],
            "both_completed_before_local_publication": all(r["publication_created"] and r["status"] == "complete_both_validated_before_publication"
                and all(t["status"] == "complete" and t["returncode"] == 0 for t in r["tasks"].values()) for r in results.values()),
            "current_input_and_task_sources_unchanged": all(record == data_identity(Path(record["path"]))
                for key in ("sources", "input") for record in protocols["parallel"][key])
        }
        rows.append({"trial": trial, "checks": row_checks, "actual_task_overlap_seconds": overlap})
    checks = {
        "three_full_cost_pairs_and_original_output_guards": len(proof["pairs"]) == 3 and all(proof["checks"].values()),
        "each_actual_pair_complete_and_current": all(all(r["checks"].values()) for r in rows),
        "restart_reuses_only_exact_completed_dependency": all(proof["lifecycle"].values()),
        "concurrent_failure_cancels_owned_peer_before_publication": all(failure["checks"].values())
            and not failure["unrelated_processes_or_existing_services_modified"],
        "cost_reduction_at_least10percent": proof["reduction"] >= proof["minimum_whole_makespan_reduction"],
        "campaign_and_fault_assessor_sources_unchanged": proof["sources"] == [data_identity(Path(r["path"])) for r in proof["sources"]]
            and failure["assessor"] == data_identity(Path(failure["assessor"]["path"]))
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R91_POSTTRAINING_EVALUATION_ARCHIVE_DAG", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "actual_pairs": rows, "median_makespan_seconds": proof["median_makespan_seconds"],
        "reduction": proof["reduction"], "scope": "existing_Box2_C212_completed_two12000_models_and_last_retained8000_complete_checkpoint",
        "generalization_pending": True, "whole_pipeline_or_quality_gain_claim": False,
        "archive_is_not_final12000_state": True, "GPU_heavy_tasks": 1,
        "independent_review": False, "campaign": str(args.campaign), "failure": str(args.failure),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({k: result[k] for k in ("status", "checks", "additional_accepted_count", "median_makespan_seconds")})
    if not all(checks.values()):
        raise ValueError("A complete posttraining DAG acceptance condition failed")


if __name__ == "__main__":
    main()
