"""Measure complete actual localization with exact outputs and equal native thread budget."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("parallel_descriptor_distances.py"),
        Path(__file__).with_name("localize_with_parallel_matching.py"), Path(__file__).with_name("byte_sift_matching.py")]
    sources = [data_identity(p) for p in paths]
    (args.output/"sources").mkdir()
    import shutil
    for path in paths:
        shutil.copyfile(path, args.output/"sources"/path.name)
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [], "counted": False,
        "minimum_matching_wall_reduction": .1, "complete_entry_time_ratio_max": 1.1,
        "native_control_OpenCV_threads": 8, "candidate_workers": 2, "candidate_OpenCV_threads_per_job": 4,
        "geometric_RANSAC_and_neighbor_order_serial": True, "all24_and48_outputs_exact": True,
        "PnP_pose_delta_max": 1e-9, "same_common_Q28_Q47_Q48_Q51": True,
        "maximum_seconds_each": 180, "maximum_RSS_GiB": 8, "three_actual_pairs": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "parallel"] if trial % 2 == 0 else ["parallel", "baseline"]):
                if sources != [data_identity(p) for p in paths]:
                    raise ValueError("A guarded descriptor implementation changed during actual matching")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_with_parallel_matching.py")),
                    "--matching-mode", mode, "--output", str(pair/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache", "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row = {"trial": trial, "mode": mode, "status": "running", "pid": process.pid}
                    state["runs"].append(row)
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(returncode=code, status="completed" if code == 0 else "failed")
                if code:
                    raise RuntimeError("A full actual bounded matching entry failed")
            entries = {mode: pair/mode/"distance_entry/reference_entry/query_entry/entry" for mode in ("baseline", "parallel")}
            invariant = compare(entries["baseline"], entries["parallel"], entries["baseline"]/"NEIGHBOR_TRACE.json", entries["parallel"]/"NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", invariant)
            processes = {mode: read(pair/(mode+"_process.json")) for mode in entries}
            costs = {mode: read(pair/mode/"distance_entry/BYTE_SIFT_TRACE.json") for mode in entries}
            trace = read(pair/"parallel/PARALLEL_MATCHING_TRACE.json")["trace"]
            item = {"trial": trial, "all_protected": invariant["all_protected"],
                "matching_wall_seconds": {mode: value["match_seconds"] for mode, value in costs.items()},
                "complete_process_seconds": {mode: value["elapsed_seconds"] for mode, value in processes.items()},
                "bounded_queue": trace["maximum_pending"] <= 2 and trace["submitted_neighbors"] == trace["consumed_neighbors"] and trace["actual_calls"] == 48,
                "trace": trace}
            item["complete_time_protected"] = item["complete_process_seconds"]["parallel"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[key] for key in ("all_protected", "bounded_queue", "complete_time_protected")):
                raise ValueError("An exact actual matching or complete time guard failed")
        median = {key: {mode: statistics.median(p[key][mode] for p in state["pairs"]) for mode in ("baseline", "parallel")}
            for key in ("matching_wall_seconds", "complete_process_seconds")}
        reduction = 1-median["matching_wall_seconds"]["parallel"]/median["matching_wall_seconds"]["baseline"]
        result = {"status": "stage_pairs_passed_lifecycle_pending" if reduction >= .1 else "rejected_insufficient_benefit",
            "median": median, "matching_wall_reduction": reduction, "pairs": state["pairs"], "sources": sources,
            "sources_unchanged": sources == [data_identity(p) for p in paths], "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
