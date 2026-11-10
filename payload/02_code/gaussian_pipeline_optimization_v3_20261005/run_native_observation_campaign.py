"""Measure full localization and native-index build and validation over three pairs."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
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
    sources = [Path(__file__), Path(__file__).with_name("native_observation_index.py"), Path(__file__).with_name("localize_native_observations.py")]
    identity = [data_identity(p) for p in sources]
    atomic_json(args.output/"PROTOCOL.json", {"sources": identity, "three_actual24_48_pairs": True,
        "CPU_geometry_includes_cold_native_snapshot_and_complete_end_verify": True,
        "minimum_complete_stage_reduction": .1, "whole_entry_cannot_regress": True,
        "one_unaccepted_Q29_index_family_variant_not_recounting_R85_QA": True, "counted": False})
    pairs = []
    try:
        for trial in range(3):
            target = args.output/f"trial{trial}"
            target.mkdir()
            order = ("baseline", "index") if trial % 2 == 0 else ("index", "baseline")
            for mode in order:
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(target/(mode+"_process.json")),
                    "--log", str(target/(mode+".log")), "--maximum-seconds", "150", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_native_observations.py")), "--mode", mode,
                    "--output", str(target/mode), "--reference-mode", "cache", "--reference-cache-root", str(args.reference_cache),
                    "--feature-mode", "cache", "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                subprocess.run(command, check=True)
            entries = {m: target/m/"entry/entry/distance_entry/reference_entry/query_entry/entry" for m in order}
            invariant = compare(entries["baseline"], entries["index"], entries["baseline"]/"NEIGHBOR_TRACE.json", entries["index"]/"NEIGHBOR_TRACE.json")
            atomic_json(target/"INVARIANTS.json", invariant)
            traces = {m: read(target/m/"INDEX_TRACE.json")["trace"] for m in order}
            checks = {"original24_48_and_poses": invariant["all_protected"],
                "all48_stage_calls": all(v["calls"] == 48 for v in traces.values()),
                "complete_native_map_unchanged": traces["index"]["index"]["complete_native_state_unchanged"],
                "all48_indexed_without_feature_deletion": all(r["route"] == "indexed" for r in traces["index"]["routes"])}
            row = {"trial": trial, "checks": checks, "whole_seconds": {m: read(target/(m+"_process.json"))["elapsed_seconds"] for m in order},
                "complete_CPU_geometry_seconds": {m: v["CPU_geometry_seconds_including_all_snapshot_and_verify"] for m, v in traces.items()},
                "index": traces["index"]["index"]}
            pairs.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs, "counted": False})
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete localization or native map ownership guard failed")
        medians = {stage: {m: statistics.median(p[stage][m] for p in pairs) for m in ("baseline", "index")}
            for stage in ("whole_seconds", "complete_CPU_geometry_seconds")}
        reductions = {stage: 1-values["index"]/values["baseline"] for stage, values in medians.items()}
        checks = {"all_three_full_original_pairs_protected": len(pairs) == 3 and all(all(p["checks"].values()) for p in pairs),
            "complete_stage_reduces10percent": reductions["complete_CPU_geometry_seconds"] >= .1,
            "whole_entry_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": identity == [data_identity(p) for p in sources]}
        result = {"status": "actual_native_index_benefit_passed_oracles_acceptance_pending" if all(checks.values()) else "failed_verification",
            "checks": checks, "pairs": pairs, "median_seconds": medians, "reductions": reductions, "sources": identity, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "median_seconds", "reductions")}, flush=True)
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed_no_acceptance", "error": repr(error), "pairs": pairs, "counted": False})
        raise


if __name__ == "__main__":
    main()
