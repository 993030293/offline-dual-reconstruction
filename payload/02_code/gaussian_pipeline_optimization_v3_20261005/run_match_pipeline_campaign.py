"""Compare actual incumbent, common serial and one-task overlap localization."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("paired_match_pipeline.py"), Path(__file__).with_name("localize_match_pipeline.py")]
    identity = [data_identity(p) for p in paths]
    protocol = {"sources": identity, "all24_and48_exact": True, "three_complete_triplets": True,
        "only_new_parallel_mechanism_compared_with_same_serial_preparation": True, "fixture_worker": "previously_accepted_legacy_R90_current_inputs_only",
        "minimum_complete_match_phase_reduction_vs_both_controls": .1, "whole_entry_cannot_regress_vs_both_controls": True,
        "first_preparation_startup_threadpool_and_shutdown_cost_in_phase_and_whole_process": True,
        "unsupported_shapes_use_native_route_before_execution_no_feature_deletion": True,
        "future_inputs_generic_worker_option_must_be_verified_separately": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    pairs = []
    try:
        orders = (("incumbent", "serial", "parallel"), ("parallel", "serial", "incumbent"), ("serial", "incumbent", "parallel"))
        for trial, order in enumerate(orders):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in order:
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "150", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_match_pipeline.py")), "--mode", mode,
                    "--distance-worker", "legacy", "--output", str(pair/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache", "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                subprocess.run(command, check=True)
            entries = {m: pair/m/"entry/entry/distance_entry/reference_entry/query_entry/entry" for m in order}
            checks = {}
            for mode in ("serial", "parallel"):
                protected = compare(entries["incumbent"], entries[mode], entries["incumbent"]/"NEIGHBOR_TRACE.json", entries[mode]/"NEIGHBOR_TRACE.json")
                atomic_json(pair/(mode+"_INVARIANTS.json"), protected)
                checks[mode+"_original24_48_and_poses"] = protected["all_protected"]
            traces = {m: read(pair/m/"entry/entry/INTEGER_SIFT_TRACE.json")["trace"] for m in order}
            checks["all24_complete_split_costs_observed"] = all(len(t["complete_current_frame_split_seconds"]) == 24 for t in traces.values())
            checks["original4GiB_and_complete_shutdown"] = all((t.get("worker", t))["returncode"] == 0
                and (t.get("worker", t))["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for t in traces.values())
            checks["actual_parallel_overlap_and_serial_no_overlap"] = sum(r["overlap_seconds"] > 0 for r in traces["parallel"]["splits"]) > 0
            checks["serial_distance_after_geometry"] = all(r["overlap_seconds"] <= 0 for r in traces["serial"]["splits"])
            row = {"trial": trial, "checks": checks,
                "whole_seconds": {m: read(pair/(m+"_process.json"))["elapsed_seconds"] for m in order},
                "complete_match_phase_seconds": {m: t["complete_match_phase_seconds"] for m, t in traces.items()},
                "actual_parallel_overlap_pairs": sum(r["overlap_seconds"] > 0 for r in traces["parallel"]["splits"])}
            pairs.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs, "counted": False})
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An actual whole pair pipeline output, cost coverage or overlap guard failed")
        medians = {stage: {m: statistics.median(r[stage][m] for r in pairs) for m in ("incumbent", "serial", "parallel")}
            for stage in ("whole_seconds", "complete_match_phase_seconds")}
        reductions = {stage: {m: 1-v["parallel"]/v[m] for m in ("incumbent", "serial")} for stage, v in medians.items()}
        checks = {"all_three_complete_triplets_and_original_guards": len(pairs) == 3 and all(all(r["checks"].values()) for r in pairs),
            "complete_matching_phase_reduces10percent_vs_both": min(reductions["complete_match_phase_seconds"].values()) >= .1,
            "complete_actual_entry_has_no_regression": min(reductions["whole_seconds"].values()) >= 0,
            "sources_unchanged": identity == [data_identity(p) for p in paths]}
        result = {"status": "actual_pair_overlap_benefit_passed_lifecycle_acceptance_pending" if all(checks.values()) else "failed_verification",
            "checks": checks, "pairs": pairs, "median_seconds": medians, "reductions": reductions,
            "independent_geometry_or_generalization_claim": False, "sources": identity, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "median_seconds", "reductions")}, flush=True)
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed_no_acceptance", "error": repr(error), "pairs": pairs, "counted": False})
        raise


if __name__ == "__main__":
    main()
