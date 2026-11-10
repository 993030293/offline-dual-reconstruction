"""Measure three full raw exports with bounded ordered preparation and unchanged outputs."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "quality-cache", "validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--jpeg-cache", type=Path)
    args = parser.parse_args()
    adapter = Path(__file__).with_name("parallel_raw_export.py")
    lifecycle = read(args.validation)
    if lifecycle["status"] != "lifecycle_passed" or lifecycle["sources"][1] != data_identity(adapter):
        raise ValueError("The raw export queue lifecycle or exact adapter identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), adapter, Path(__file__).with_name("run_initial_with_export_queue.py"),
        Path(__file__).with_name("raw_initial_invariants.py"), Path(__file__).with_name("run_initial_with_jpeg_cache.py"),
        Path(__file__).with_name("encoded_jpeg_cache.py")]
    paths += [V1/name for name in ("run_initial_pipeline.py", "runtime.py", "quality_engine.py", "selection_engine.py", "pose_interpolation.py")]
    sources = [data_identity(path) for path in paths]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshot/path.name)
    state = {"status": "running", "sources": sources, "pairs": [], "runs": [], "counted": False,
        "minimum_full_export_stage_reduction": .1, "complete_entry_time_ratio_max": 1.1,
        "queue_workers": 4, "queue_capacity": 8, "OpenCV_threads": 1,
        "all7338_decisions1002_JPEG_bytes11583_pairs_and_raw_odometry_exact": True,
        "JPEG_cache_mode_common_on_both_sides": bool(args.jpeg_cache),
        "first_export_without_cache": args.jpeg_cache is None, "maximum_seconds_each": 180,
        "maximum_RSS_GiB": 8, "maximum_group_disk_GiB": 8, "three_actual_pairs": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "queue"] if trial % 2 == 0 else ["queue", "baseline"]):
                if sources != [data_identity(p) for p in paths]:
                    raise ValueError("A frozen ordered raw export implementation changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_initial_with_export_queue.py")),
                    "--export-mode", mode, "--output", str(pair/mode), "--quality-cache", str(args.quality_cache)]
                if args.jpeg_cache:
                    command += ["--jpeg-cache", str(args.jpeg_cache)]
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["runs"].append(row)
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("An actual full raw export queue entry failed")
            roots = {mode: pair/mode/"raw_entry/initial" for mode in ("baseline", "queue")}
            invariants = compare_initial_outputs(roots["baseline"], roots["queue"])
            atomic_json(pair/"INVARIANTS.json", invariants)
            receipts = {mode: read(root/"SELECTION_RECEIPT.json") for mode, root in roots.items()}
            processes = {mode: read(pair/(mode+"_process.json")) for mode in roots}
            traces = {mode: read(pair/mode/"EXPORT_QUEUE_TRACE.json")["records"][0] for mode in roots}
            item = {"trial": trial, "all_protected": invariants["all_protected"],
                "export_seconds": {mode: r["timings"]["decode_encode_write_seconds"] for mode, r in receipts.items()},
                "complete_entry_seconds": {mode: r["elapsed_seconds"] for mode, r in receipts.items()},
                "complete_process_seconds": {mode: r["elapsed_seconds"] for mode, r in processes.items()},
                "process_peak_RSS_bytes": {mode: r["peak_tree_RSS_bytes"] for mode, r in processes.items()},
                "queue_traces": traces, "all_queues_consumed1002_and_closed": all(r["submitted"] == r["consumed"] == 1002 and
                    r["closed"] and r["maximum_pending"] <= 8 for r in traces.values())}
            item["complete_time_protected"] = item["complete_process_seconds"]["queue"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[key] for key in ("all_protected", "all_queues_consumed1002_and_closed", "complete_time_protected")):
                raise ValueError("A complete original raw export guard did not pass")
            if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 8*1024**3:
                raise MemoryError("The ordered export group output budget was exceeded")
        medians = {kind: {mode: statistics.median(p[kind][mode] for p in state["pairs"]) for mode in ("baseline", "queue")}
            for kind in ("export_seconds", "complete_entry_seconds", "complete_process_seconds", "process_peak_RSS_bytes")}
        reduction = 1-medians["export_seconds"]["queue"]/medians["export_seconds"]["baseline"]
        result = {"status": "stage_guards_passed_combination_pending" if reduction >= .1 else "rejected",
            "median": medians, "export_stage_reduction": reduction,
            "median_absolute_export_seconds_saved": medians["export_seconds"]["baseline"]-medians["export_seconds"]["queue"],
            "pairs": state["pairs"], "sources": sources, "lifecycle": str(args.validation),
            "sources_unchanged": sources == [data_identity(p) for p in paths], "counted": False,
            "JPEG_cache_common": bool(args.jpeg_cache), "whole_route_or_quality_improvement_claim": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
