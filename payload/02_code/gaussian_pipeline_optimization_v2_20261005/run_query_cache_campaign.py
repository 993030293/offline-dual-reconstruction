"""Run three fresh-process pairs and protect the complete localization outputs."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from compare_actual_localization import compare

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "cache", "output", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cold = read(args.cold/"FEATURE_CACHE_TRACE.json")
    if cold["status"] != "all24_actual_queries_completed" or cold["cache"]["misses"] != 24 or cold["cache"]["hits"] != 0:
        raise ValueError("The actual24-query cold fill is incomplete")
    sources = [Path(__file__), Path(__file__).with_name("localize_with_cached_queries.py"),
               Path(__file__).with_name("query_feature_cache.py"), Path(__file__).with_name("compare_actual_localization.py"),
               Path(__file__).with_name("localize_with_neighbors.py"), Path(__file__).with_name("frozen_neighbors.py"),
               V1/"localize_new_map_evaluation.py", V1/"sift_feature_store.py"]
    identities = [data_identity(p) for p in sources]
    snapshots = args.output/"sources"
    snapshots.mkdir()
    for p in sources:
        shutil.copyfile(p, snapshots/p.name)
    atomic_json(args.output/"PROTOCOL.json", {"sources": identities, "cold": data_identity(args.cold/"FEATURE_CACHE_TRACE.json"),
        "three_pairs": True, "full24_and_all48_match_outputs": True, "pose_delta_max": 1e-9,
        "minimum_stage_reduction": .1, "maximum_process_time_ratio": 1.1,
        "maximum_seconds_per_run": 900, "maximum_RSS_GiB": 8, "maximum_cache_bytes": 512*1024**2,
        "cold_and_repeated_costs_separate": True, "map_independent_features_only": True, "counted": False})
    state = {"status": "running", "stages": [], "pairs": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "cache"] if trial % 2 == 0 else ["cache", "baseline"]):
                if identities != [data_identity(p) for p in sources]:
                    raise ValueError("A bound source changed between fresh processes")
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["stages"].append(row)
                atomic_json(args.output/"STATE.json", state)
                command = [sys.executable, str(V1/"run_bounded_task.py"),
                    "--receipt", str(pair/(mode+"_process.json")), "--log", str(pair/(mode+".log")),
                    "--maximum-seconds", "900", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_with_cached_queries.py")),
                    "--feature-mode", mode, "--feature-cache-root", str(args.cache), "--output", str(pair/mode)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["controller_pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("A complete query cache entry failed")
            left, right = pair/"baseline", pair/"cache"
            guard = compare(left/"entry", right/"entry", left/"entry/NEIGHBOR_TRACE.json", right/"entry/NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", guard)
            traces = {m: read(pair/m/"FEATURE_CACHE_TRACE.json") for m in ("baseline", "cache")}
            processes = {m: read(pair/(m+"_process.json")) for m in ("baseline", "cache")}
            item = {"trial": trial, "all24_invariants": guard["all_protected"],
                "all24_hits": traces["cache"]["cache"]["hits"] == 24 and traces["cache"]["cache"]["misses"] == 0,
                "query_feature_seconds": {m: t["query_feature_seconds"] for m, t in traces.items()},
                "complete_process_seconds": {m: p["elapsed_seconds"] for m, p in processes.items()},
                "peak_process_RSS_bytes": {m: p["peak_tree_RSS_bytes"] for m, p in processes.items()},
                "sources_unchanged": identities == [data_identity(p) for p in sources]}
            item["process_time_protected"] = item["complete_process_seconds"]["cache"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[k] for k in ("all24_invariants", "all24_hits", "sources_unchanged", "process_time_protected")):
                raise ValueError("A real query cache pair failed an original protection")
        medians = {kind: {m: statistics.median(p[kind][m] for p in state["pairs"]) for m in ("baseline", "cache")}
                   for kind in ("query_feature_seconds", "complete_process_seconds")}
        reduction = 1-medians["query_feature_seconds"]["cache"]/medians["query_feature_seconds"]["baseline"]
        checks = {"three_real_pairs": len(state["pairs"]) == 3,
                  "all_original_invariants": all(p["all24_invariants"] for p in state["pairs"]),
                  "all24_warm_hits_in_each_pair": all(p["all24_hits"] for p in state["pairs"]),
                  "sources_unchanged": identities == [data_identity(p) for p in sources],
                  "query_stage_reduces10percent": reduction >= .1,
                  "complete_process_time_protected": all(p["process_time_protected"] for p in state["pairs"])}
        result = {"status": "three_actual_pairs_passed_lifecycle_pending" if all(checks.values()) else "failed",
            "checks": checks, "pairs": state["pairs"], "median": medians, "query_stage_reduction": reduction,
            "cold_query_feature_seconds": cold["query_feature_seconds"], "cache_bytes": cold["cache"]["cache_bytes"],
            "deployment": "optional_repeated_localization_only", "counted": False,
            "quality_and_new_pose_accuracy_claim": False, "round2_combination_accepted": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
