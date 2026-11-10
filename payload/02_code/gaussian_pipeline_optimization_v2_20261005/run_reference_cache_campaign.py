"""Measure three complete paired localizations with exact calibrated references."""

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
    for name in ("cold", "cache", "feature-cache", "output", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cold = read(args.cold/"REFERENCE_CACHE_TRACE.json")
    if cold["cache"]["hits"] != 0 or cold["cache"]["misses"] != 24:
        raise ValueError("The complete actual24 calibrated-reference cold fill is required")
    sources = [Path(__file__), Path(__file__).with_name("calibrated_reference_cache.py"),
        Path(__file__).with_name("localize_with_cached_references.py"),
        Path(__file__).with_name("localize_with_cached_queries.py"), Path(__file__).with_name("query_feature_cache.py"),
        Path(__file__).with_name("localize_with_neighbors.py"), Path(__file__).with_name("frozen_neighbors.py"),
        V1/"localize_new_map_evaluation.py", V1/"sift_feature_store.py"]
    identities = [data_identity(p) for p in sources]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for p in sources:
        shutil.copyfile(p, snapshot/p.name)
    atomic_json(args.output/"PROTOCOL.json", {"sources": identities,
        "three_actual_full24_pairs": True, "all48_match_outputs_and_reference_PNG_bytes_exact": True,
        "pose_delta_max": 1e-9, "minimum_reference_stage_reduction": .1, "maximum_process_time_ratio": 1.1,
        "query_feature_cache_on_both_modes": True, "maximum_seconds_per_entry": 900,
        "maximum_RSS_GiB": 8, "maximum_cache_bytes": 512*1024**2,
        "cold_and_repeat_separate": True, "counted": False})
    state = {"status": "running", "runs": [], "pairs": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "cache"] if trial % 2 == 0 else ["cache", "baseline"]):
                if identities != [data_identity(p) for p in sources]:
                    raise ValueError("A bound localization source changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "900", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_with_cached_references.py")),
                    "--reference-mode", mode, "--reference-cache-root", str(args.cache), "--output", str(pair/mode),
                    "--feature-mode", "cache", "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                if code:
                    raise RuntimeError("An actual reference-cache localization entry failed")
            entries = {m: pair/m/"query_entry/entry" for m in ("baseline", "cache")}
            invariant = compare(entries["baseline"], entries["cache"],
                entries["baseline"]/"NEIGHBOR_TRACE.json", entries["cache"]/"NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", invariant)
            traces = {m: read(pair/m/"REFERENCE_CACHE_TRACE.json") for m in ("baseline", "cache")}
            processes = {m: read(pair/(m+"_process.json")) for m in ("baseline", "cache")}
            item = {"trial": trial, "all24_and48_protected": invariant["all_protected"],
                "actual24_hits": traces["cache"]["cache"]["hits"] == 24 and traces["cache"]["cache"]["misses"] == 0,
                "reference_stage_seconds": {m: r["reference_stage_seconds"] for m, r in traces.items()},
                "complete_process_seconds": {m: r["elapsed_seconds"] for m, r in processes.items()},
                "peak_RSS_bytes": {m: r["peak_tree_RSS_bytes"] for m, r in processes.items()}}
            item["full_process_protected"] = item["complete_process_seconds"]["cache"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[k] for k in ("all24_and48_protected", "actual24_hits", "full_process_protected")):
                raise ValueError("A reference cache pair failed its original complete-entry guard")
        median = {kind: {m: statistics.median(p[kind][m] for p in state["pairs"]) for m in ("baseline", "cache")}
            for kind in ("reference_stage_seconds", "complete_process_seconds")}
        reduction = 1-median["reference_stage_seconds"]["cache"]/median["reference_stage_seconds"]["baseline"]
        result = {"status": "three_actual_pairs_passed_lifecycle_pending" if reduction >= .1 else "failed_speed_guard",
            "pairs": state["pairs"], "median": median, "reference_stage_reduction": reduction,
            "cold_reference_seconds": cold["reference_stage_seconds"], "cache_bytes": cold["cache"]["cache_bytes"],
            "sources_unchanged": identities == [data_identity(p) for p in sources],
            "deployment": "optional_repeated_calibrated_reference_export", "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
