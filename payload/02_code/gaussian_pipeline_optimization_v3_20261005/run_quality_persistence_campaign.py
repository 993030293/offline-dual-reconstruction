"""Measure three full cold raw pairs including every original durable cache commit."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "oracle", "reference-initial"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The original durable-write and lifecycle oracle did not pass")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_raw_quality_persistence.py"),
        Path(__file__).with_name("quality_persistence_pipeline.py"),
        Path(__file__).with_name("quality_histogram.py"), Path(__file__).with_name("run_raw_calibration.py"),
        Path(__file__).with_name("calibration_body_cache.py"), V1/"quality_engine.py", V1/"run_initial_pipeline.py",
        V2/"run_effective_initial.py", V2/"raw_initial_invariants.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1] or sources[6] != oracle["sources"][2]:
        raise ValueError("A current writer or original cache source differs from the full oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"),
        "pairs": [], "runs": [], "counted": False,
        "minimum_complete_cold_quality_stage_reduction": .1, "maximum_each_whole_entry_ratio": 1.1,
        "whole_entry_median_cannot_regress": True, "maximum_group_seconds": 1800,
        "same_R104_R99_Q78_Q50_Q53_P18_and_all_original_raw_denominators": True,
        "six_new_empty_private_quality_caches_JPEG_cache_off": True,
        "all_queue_waits_cache_writes_fsync_atomic_replace_and_shutdown_inside_quality_wall": True,
        "no_GT_SfM_object_shape_or_pose_accuracy_input": True,
        "full_cache_byte_verification_wall_reported_separately_from_entry": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        for trial in range(3):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            order = ("native", "pipeline") if trial % 2 == 0 else ("pipeline", "native")
            duration, costs, details = {}, {}, {}
            for mode in order:
                if time.perf_counter()-started > 1800 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The registered schedule or source changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"),
                    "--receipt", str(folder/(mode+"_process.json")), "--log", str(folder/(mode+".log")),
                    "--maximum-seconds", "600", "--maximum-rss-gib", "8", "--", sys.executable,
                    str(Path(__file__).with_name("run_raw_quality_persistence.py")), "--write-mode", mode,
                    "--output", str(folder/mode), "--quality-cache", str(folder/(mode+"_quality"))]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                run.update(returncode=code, status="completed" if code == 0 else "failed")
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("A complete durable quality entry failed")
                costs[mode] = read(folder/mode/"PERSISTENCE.json")
                details[mode] = read(folder/mode/"initial/SELECTION_RECEIPT.json")
                duration[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
            before = time.perf_counter()
            protections = {"pair": compare_initial_outputs(folder/"native/initial", folder/"pipeline/initial"),
                "current_to_native": compare_initial_outputs(args.reference_initial, folder/"native/initial"),
                "current_to_pipeline": compare_initial_outputs(args.reference_initial, folder/"pipeline/initial")}
            names = sorted(path.name for path in (folder/"native_quality").iterdir())
            caches_exact = (len(names) == 7338 and all(name.endswith(".json") for name in names)
                and names == sorted(path.name for path in (folder/"pipeline_quality").iterdir())
                and all((folder/"native_quality"/name).read_bytes() ==
                    (folder/"pipeline_quality"/name).read_bytes() for name in names))
            verification_seconds = time.perf_counter()-before
            atomic_json(folder/"INVARIANTS.json", {**protections, "all7338_original_cache_files_exact": caches_exact,
                "verification_seconds_separate_from_runtime": verification_seconds})
            writer = costs["pipeline"]["writer_trace"]
            checks = {"all_current_original_raw_outputs_and_complete_pair_exact": all(row["all_protected"] for row in protections.values()),
                "all7338_original_cache_files_exact_no_pending_temps": caches_exact,
                "all7338_original_cache_misses_both_routes": all(row["timings"]["raw_image_scoring_detail"]["cache_hits"] == 0
                    and row["timings"]["raw_image_scoring_detail"]["submitted"] == 7338 for row in details.values()),
                "all7338_durable_writes_drained_and_queue_bounded": len(writer["calls"]) == 1
                    and writer["calls"][0]["submitted"] == writer["calls"][0]["finished"] == writer["calls"][0]["committed"] == 7338
                    and writer["calls"][0]["maximum_pending"] <= 64 and writer["calls"][0]["closed"],
                "all_actual_bindings_restored_and_original_formula_preserved": all(row["writer_trace"]["original_functions_restored"]
                    and row["histogram_trace"]["aliases_restored"] and row["histogram_trace"]["histogram_calls"] == 7338 for row in costs.values()),
                "complete_entry_pair_cost_protected": duration["pipeline"] <= 1.1*duration["native"]}
            row = {"trial": trial, "order": order, "checks": checks, "whole_seconds": duration,
                "complete_cold_quality_stage_seconds": {mode: details[mode]["timings"]["raw_image_scoring_wall_seconds"]
                    +costs[mode]["additional_quality_lifecycle_seconds"] for mode in order},
                "complete_quality_details": {mode: details[mode]["timings"]["raw_image_scoring_detail"] for mode in order},
                "writer_trace": writer, "verification_seconds_separate_from_runtime": verification_seconds}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print({key: value for key, value in row.items() if key not in ("complete_quality_details", "writer_trace")}, flush=True)
            if not all(checks.values()):
                raise ValueError("An original cold entry output, commit or cost guard failed")
        median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"])
            for mode in ("native", "pipeline")} for key in ("whole_seconds", "complete_cold_quality_stage_seconds")}
        reductions = {key: 1-value["pipeline"]/value["native"] for key, value in median.items()}
        checks = {"all_three_complete_cold_raw_pairs_and_protections": len(state["pairs"]) == 3
                and all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_cold_quality_stage_reduces10percent": reductions["complete_cold_quality_stage_seconds"] >= .1,
            "whole_entry_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="all_actual_cold_quality_guards_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median=median, reductions=reductions, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
