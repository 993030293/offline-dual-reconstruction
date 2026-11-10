"""Compare full localization and digest costs while byte immutability is enforced."""

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
    for name in ("oracle", "output", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if not all(oracle["checks"].values()):
        raise ValueError("The owned byte immutability and SQLite oracle is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("localize_with_locked_inputs.py"),
        Path(__file__).with_name("locked_input_identity.py")]
    identity = [data_identity(p) for p in paths]
    protocol = {"sources": identity, "control": "accepted_R90_Q28_Q47_Q48_Q51_actual_entry",
        "primary_stage": "complete_source_and_input_identity_checks", "minimum_digest_reduction": .1,
        "minimum_actual_whole_entry_reduction": .05, "three_full_pairs": True,
        "original_quality_and_all24_48_correspondences_and_poses_preserved": True,
        "parameters_frozen_before_cost_results": True, "Windows_scope": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    pairs = []
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "locked"] if trial % 2 == 0 else ["locked", "baseline"]):
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "150", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_with_locked_inputs.py")),
                    "--snapshot-mode", mode, "--output", str(pair/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache",
                    "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                subprocess.run(command, check=True)
            entries = {mode: pair/mode/"entry/distance_entry/reference_entry/query_entry/entry" for mode in ("baseline", "locked")}
            invariant = compare(entries["baseline"], entries["locked"], entries["baseline"]/"NEIGHBOR_TRACE.json", entries["locked"]/"NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", invariant)
            traces = {m: read(pair/m/"SNAPSHOT_TRACE.json") for m in entries}
            row = {"trial": trial, "all_protected": invariant["all_protected"],
                "all_handles_closed": all(r["no_held_handles_after_context"] for r in traces.values()),
                "whole_seconds": {m: read(pair/(m+"_process.json"))["elapsed_seconds"] for m in entries},
                "identity_phase_seconds": {m: r["identity_phase_seconds"] for m, r in traces.items()},
                "snapshot": traces["locked"]["snapshot"]}
            pairs.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs, "counted": False})
            print(row, flush=True)
            if not row["all_protected"] or not row["all_handles_closed"]:
                raise ValueError("An original full localization or owned snapshot lifecycle guard failed")
        medians = {stage: {m: statistics.median(p[stage][m] for p in pairs) for m in ("baseline", "locked")}
            for stage in ("whole_seconds", "identity_phase_seconds")}
        reductions = {stage: 1-r["locked"]/r["baseline"] for stage, r in medians.items()}
        checks = {"three_all24_48_and_pose_pairs_protected": len(pairs) == 3 and all(p["all_protected"] and p["all_handles_closed"] for p in pairs),
            "original_identity_stage_reduction_at_least10percent": reductions["identity_phase_seconds"] >= .1,
            "whole_actual_entry_reduction_at_least5percent": reductions["whole_seconds"] >= .05,
            "same_oracle_and_sources": oracle["sources"] == [data_identity(Path(r["path"])) for r in oracle["sources"]]
                and identity == [data_identity(p) for p in paths]}
        result = {"status": "actual_snapshot_cost_and_protection_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            "checks": checks, "pairs": pairs, "median_seconds": medians, "reductions": reductions,
            "Windows_localization_scope": True, "new_geometry_accuracy_or_quality_claim": False, "counted": False,
            "sources": identity, "oracle": str(args.oracle)}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "median_seconds", "reductions")}, flush=True)
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed_no_acceptance", "error": repr(error), "pairs": pairs, "counted": False})
        raise


if __name__ == "__main__":
    main()
