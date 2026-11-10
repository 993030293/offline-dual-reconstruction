"""Measure isolated full raw runtime choices against the same original keyed cache."""

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
    parser.add_argument("--candidate", choices=["columnar", "mapped"], required=True)
    parser.add_argument("--score-table", type=Path)
    args = parser.parse_args()
    if (args.candidate == "columnar") != bool(args.score_table):
        raise ValueError("The numeric-score candidate requires its explicit immutable table only")
    if read(args.validation)["status"] != "lifecycle_passed":
        raise ValueError("The exact raw runtime candidate lifecycle is incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_initial_with_runtime_choices.py"),
        Path(__file__).with_name("columnar_quality_scores.py"), Path(__file__).with_name("raw_initial_invariants.py"),
        Path(__file__).with_name("windowed_raw_reader.py")]
    paths += [V1/name for name in ("run_initial_pipeline.py", "mapped_raw_reader.py", "quality_engine.py")]
    sources = [data_identity(p) for p in paths]
    (args.output/"sources").mkdir()
    for path in paths:
        shutil.copyfile(path, args.output/"sources"/path.name)
    state = {"status": "running", "candidate": args.candidate, "sources": sources, "runs": [], "pairs": [],
        "three_actual_pairs": True, "minimum_quality_scoring_wall_reduction_including_table_startup": .1,
        "complete_entry_time_ratio_max": 1.1, "all7338_decisions1002_JPEG11583_pairs_and_odometry_exact": True,
        "quality_cache_mode": "same complete warm image-only cache; no JPEG cache", "counted": False,
        "table_build_receipt": read(args.score_table/"BUILD_RECEIPT.json") if args.score_table else None,
        "first_run_includes_table_build_in_cold_cost": bool(args.score_table), "maximum_group_disk_GiB": 8,
        "maximum_seconds_each": 180, "maximum_RSS_GiB": 8}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]):
                if sources != [data_identity(p) for p in paths]:
                    raise ValueError("An isolated raw runtime source changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_initial_with_runtime_choices.py")),
                    "--output", str(pair/mode), "--quality-cache", str(args.quality_cache),
                    "--reader-mode", "mapped" if mode == "candidate" and args.candidate == "mapped" else "indexed"]
                if mode == "candidate" and args.score_table:
                    command += ["--score-table", str(args.score_table)]
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row = {"trial": trial, "mode": mode, "pid": process.pid, "status": "running"}
                    state["runs"].append(row)
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(returncode=code, status="completed" if code == 0 else "failed")
                if code:
                    raise RuntimeError("A full raw runtime entry failed")
            roots = {mode: pair/mode/"initial" for mode in ("baseline", "candidate")}
            invariant = compare_initial_outputs(roots["baseline"], roots["candidate"])
            atomic_json(pair/"INVARIANTS.json", invariant)
            receipts = {mode: read(root/"SELECTION_RECEIPT.json") for mode, root in roots.items()}
            traces = {mode: read(pair/mode/"RUNTIME_TRACE.json") for mode in roots}
            startup = traces["candidate"]["numeric_score_cache"]["initialization_seconds"] if args.score_table else 0.
            item = {"trial": trial, "all_protected": invariant["all_protected"],
                "scoring_wall_seconds": {mode: value["timings"]["raw_image_scoring_wall_seconds"]+
                    (startup if mode == "candidate" else 0.) for mode, value in receipts.items()},
                "complete_entry_seconds": {mode: value["elapsed_seconds"] for mode, value in receipts.items()},
                "table_initialization_seconds_included": startup,
                "original_cache_hits": {mode: value["timings"]["raw_image_scoring_detail"]["cache_hits"] for mode, value in receipts.items()},
                "all_runtime_traces": traces}
            item["complete_time_protected"] = item["complete_entry_seconds"]["candidate"] <= 1.1*item["complete_entry_seconds"]["baseline"]
            if args.score_table:
                item["all_exact_table_hits"] = traces["candidate"]["numeric_score_cache"]["hits"] == 7338 and traces["candidate"]["numeric_score_cache"]["misses"] == 0
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps({key: item[key] for key in ("trial", "all_protected", "scoring_wall_seconds", "complete_entry_seconds", "complete_time_protected")}), flush=True)
            if not item["all_protected"] or not item["complete_time_protected"] or not item.get("all_exact_table_hits", True):
                raise ValueError("The raw runtime output or complete time guard failed")
            if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 8*1024**3:
                raise MemoryError("The raw runtime group disk budget was exceeded")
        median = {key: {mode: statistics.median(p[key][mode] for p in state["pairs"]) for mode in ("baseline", "candidate")}
            for key in ("scoring_wall_seconds", "complete_entry_seconds")}
        reduction = 1-median["scoring_wall_seconds"]["candidate"]/median["scoring_wall_seconds"]["baseline"]
        cold = median["complete_entry_seconds"]["candidate"]+(state["table_build_receipt"]["build_seconds"] if args.score_table else 0.)
        result = {"status": "stage_guards_passed_combination_pending" if reduction >= .1 else "rejected_insufficient_wall_benefit",
            "median": median, "scoring_wall_reduction": reduction, "first_derived_artifact_complete_cost_seconds": cold,
            "cold_cost_uses_existing_warm_quality_cache": True, "pairs": state["pairs"], "sources": sources,
            "sources_unchanged": sources == [data_identity(p) for p in paths], "counted": False,
            "table_build_receipt": state["table_build_receipt"], "absolute_scoring_seconds_saved": median["scoring_wall_seconds"]["baseline"]-median["scoring_wall_seconds"]["candidate"]}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
