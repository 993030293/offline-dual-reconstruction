"""Measure exact spatial indexing through three full raw-only initial entry pairs."""

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


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "quality-cache", "jpeg-cache", "validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    lifecycle = read(args.validation)
    pair_source = Path(__file__).with_name("indexed_candidate_pairs.py")
    if lifecycle["status"] != "lifecycle_passed" or lifecycle["sources"][1] != data_identity(pair_source):
        raise ValueError("The exact pair-index lifecycle or implementation identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), pair_source, Path(__file__).with_name("run_initial_with_pair_index.py"),
        Path(__file__).with_name("run_initial_with_jpeg_cache.py"), Path(__file__).with_name("encoded_jpeg_cache.py")]
    paths += [V1/name for name in ("run_initial_pipeline.py", "quality_engine.py", "selection_engine.py", "pose_interpolation.py")]
    sources = [data_identity(p) for p in paths]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshot/path.name)
    state = {"status": "running", "sources": sources, "pairs": [], "runs": [], "counted": False,
        "stage_reduction_minimum": .1, "complete_entry_time_ratio_max": 1.1,
        "all7338_decisions1002_image_bytes11583_pairs_exact": True,
        "coarse_pose_and_quality_only_for_initial_selection": True,
        "tree_build_is_included": True, "three_actual_pairs": True,
        "maximum_seconds_each": 180, "maximum_RSS_GiB": 8, "maximum_group_disk_GiB": 8}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "indexed"] if trial % 2 == 0 else ["indexed", "baseline"]):
                if sources != [data_identity(p) for p in paths]:
                    raise ValueError("A frozen initial-pair implementation changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_initial_with_pair_index.py")),
                    "--pair-mode", mode, "--output", str(pair/mode), "--quality-cache", str(args.quality_cache),
                    "--jpeg-cache", str(args.jpeg_cache)]
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["runs"].append(row)
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                if code:
                    raise RuntimeError("An actual full raw-pair entry failed")
            entries = {mode: pair/mode/"raw_entry/initial" for mode in ("baseline", "indexed")}
            manifests = {mode: read(p/"INITIAL_FRAME_MANIFEST.json") for mode, p in entries.items()}
            traces = {mode: read(pair/mode/"PAIR_INDEX_TRACE.json") for mode in entries}
            receipts = {mode: read(p/"SELECTION_RECEIPT.json") for mode, p in entries.items()}
            process_receipts = {mode: read(pair/(mode+"_process.json")) for mode in entries}
            exact = all(manifests["baseline"][key] == manifests["indexed"][key]
                for key in ("selected_frames", "camera_intrinsics", "policy"))
            canonical = ("frame_decisions.jsonl", "initial_matching_pairs.txt", "EXCLUDED_EVALUATION_IDS.json", "RAW_SENSOR_CONTRACT.json")
            exact &= all((entries["baseline"]/name).read_bytes() == (entries["indexed"]/name).read_bytes() for name in canonical)
            names = sorted(p.name for p in (entries["baseline"]/"images").iterdir())
            exact &= names == sorted(p.name for p in (entries["indexed"]/"images").iterdir())
            exact &= all((entries["baseline"]/"images"/name).read_bytes() == (entries["indexed"]/"images"/name).read_bytes() for name in names)
            denominator = all((r["raw_frame_count"], r["selected_count"], r["matching_candidate_pair_count"], r["excluded_overlap"]) ==
                (7338, 1002, 11583, 0) for r in receipts.values())
            item = {"trial": trial, "all_actual_outputs_exact": bool(exact), "denominator_protected": denominator,
                "pair_stage_seconds": {mode: t["records"][0]["seconds"] for mode, t in traces.items()},
                "complete_entry_seconds": {mode: r["elapsed_seconds"] for mode, r in receipts.items()},
                "complete_process_seconds": {mode: r["elapsed_seconds"] for mode, r in process_receipts.items()},
                "ambiguous_boundary_fallback_calls": traces["indexed"]["records"][0]["ambiguous_boundary_full_reference_calls"]}
            item["complete_time_protected"] = item["complete_process_seconds"]["indexed"] <= 1.1*item["complete_process_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not all(item[key] for key in ("all_actual_outputs_exact", "denominator_protected", "complete_time_protected")):
                raise ValueError("An original complete raw input guard did not pass")
            if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 8*1024**3:
                raise MemoryError("The paired full raw output budget was exceeded")
        medians = {kind: {mode: statistics.median(p[kind][mode] for p in state["pairs"]) for mode in ("baseline", "indexed")}
            for kind in ("pair_stage_seconds", "complete_entry_seconds", "complete_process_seconds")}
        reduction = 1-medians["pair_stage_seconds"]["indexed"]/medians["pair_stage_seconds"]["baseline"]
        result = {"status": "stage_guards_passed_combination_pending" if reduction >= .1 else "rejected",
            "sources": sources, "pairs": state["pairs"], "median": medians, "pair_stage_reduction": reduction,
            "median_absolute_pair_seconds_saved": medians["pair_stage_seconds"]["baseline"]-medians["pair_stage_seconds"]["indexed"],
            "counted": False, "whole_route_speed_or_quality_claim": False,
            "tree_build_included": True, "lifecycle": str(args.validation),
            "sources_unchanged": sources == [data_identity(p) for p in paths]}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
