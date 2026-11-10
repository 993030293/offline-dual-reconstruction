"""Compare three real24 byte-SIFT localization pairs and actual descriptor memory."""

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
    for name in ("output", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("byte_sift_matching.py"),
        Path(__file__).with_name("localize_with_byte_sift.py"), Path(__file__).with_name("localize_with_cached_references.py"),
        Path(__file__).with_name("calibrated_reference_cache.py"), Path(__file__).with_name("localize_with_cached_queries.py"),
        Path(__file__).with_name("query_feature_cache.py"), Path(__file__).with_name("localize_with_neighbors.py"),
        Path(__file__).with_name("frozen_neighbors.py"), V1/"sift_feature_store.py", V1/"localize_new_map_evaluation.py"]
    identities = [data_identity(p) for p in sources]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for p in sources:
        shutil.copyfile(p, snapshot/p.name)
    state = {"status": "running", "sources": identities, "pairs": [], "runs": [], "counted": False,
        "three_actual_pairs": True, "all24_and48_outputs_exact": True, "pose_delta_max": 1e-9,
        "minimum_descriptor_store_and_transient_peak_reduction": .1, "maximum_complete_process_time_ratio": 1.1,
        "capacity32_same_and_no_new_image_quantization": True, "original_float_arrays_recover_exactly": True,
        "native_matcher_is_original_float32": True,
        "previous_native_uint8_matcher_failed_latency_guard": "q51_campaign_001/STATE.json",
        "maximum_seconds_per_entry": 900, "maximum_RSS_GiB": 8}
    atomic_json(args.output/"PROTOCOL.json", state)
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "bytes"] if trial % 2 == 0 else ["bytes", "baseline"]):
                if identities != [data_identity(p) for p in sources]:
                    raise ValueError("A frozen byte matching implementation changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "900", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_with_byte_sift.py")),
                    "--descriptor-mode", mode, "--output", str(pair/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache",
                    "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["runs"].append(row)
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                if code:
                    raise RuntimeError("An actual byte-SIFT pair failed")
            entries = {m: pair/m/"reference_entry/query_entry/entry" for m in ("baseline", "bytes")}
            invariants = compare(entries["baseline"], entries["bytes"],
                entries["baseline"]/"NEIGHBOR_TRACE.json", entries["bytes"]/"NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", invariants)
            stores = {m: read(p/"localization/FEATURE_COST.json")["final_store"] for m, p in entries.items()}
            processes = {m: read(pair/(m+"_process.json")) for m in entries}
            traces = {m: read(pair/m/"BYTE_SIFT_TRACE.json") for m in entries}
            item = {"trial": trial, "all24_and48_protected": invariants["all_protected"],
                "peak_feature_store_array_bytes": {m: r["maximum_cached_array_bytes"] for m, r in stores.items()},
                "peak_store_plus_two_transient_upper_bound_bytes": {m: r["maximum_cached_array_bytes"]+
                    2*traces[m]["mechanism_calls"]["maximum_transient_descriptor_bytes"] for m, r in stores.items()},
                "maximum_transient_descriptor_bytes": {m: r["mechanism_calls"]["maximum_transient_descriptor_bytes"] for m, r in traces.items()},
                "resident_feature_store_array_bytes": {m: r["cached_array_bytes"] for m, r in stores.items()},
                "match_seconds": {m: r["match_seconds"] for m, r in traces.items()},
                "complete_process_seconds": {m: r["elapsed_seconds"] for m, r in processes.items()},
                "process_peak_RSS_bytes": {m: r["peak_tree_RSS_bytes"] for m, r in processes.items()},
                "same_feature_cache_access_and_capacity": all(stores["baseline"][k] == stores["bytes"][k]
                    for k in ("capacity", "hits", "misses", "evictions", "maximum_cached_images")),
                "all48_real_matching_calls": all(r["all48_match_calls"] == 48 for r in traces.values())}
            item["complete_time_protected"] = item["complete_process_seconds"]["bytes"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[k] for k in ("all24_and48_protected", "same_feature_cache_access_and_capacity", "all48_real_matching_calls", "complete_time_protected")):
                raise ValueError("A real byte-SIFT pair failed its original preservation guard")
        medians = {kind: {m: statistics.median(p[kind][m] for p in state["pairs"]) for m in ("baseline", "bytes")}
            for kind in ("peak_feature_store_array_bytes", "peak_store_plus_two_transient_upper_bound_bytes", "match_seconds", "complete_process_seconds")}
        reduction = 1-medians["peak_store_plus_two_transient_upper_bound_bytes"]["bytes"]/medians["peak_store_plus_two_transient_upper_bound_bytes"]["baseline"]
        result = {"status": "three_actual_pairs_passed_lifecycle_pending" if reduction >= .1 else "failed_memory_guard",
            "pairs": state["pairs"], "median": medians, "feature_store_memory_reduction": reduction,
            "sources_unchanged": identities == [data_identity(p) for p in sources], "counted": False,
            "new_pose_accuracy_or_quality_claim": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
