"""Require all raw decisions and the complete cold quality stage to remain protected."""

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
    for name in ("output", "oracle", "boundary", "reference-initial"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle, boundary = read(args.oracle/"RESULT.json"), read(args.boundary/"RESULT.json")
    if (oracle["status"] != "pilot_passed_full_entry_pending" or not all(oracle["checks"].values())
            or boundary["status"] != "passed" or not all(boundary["checks"].values())):
        raise ValueError("The original generic and full-scorer diagnostic gates did not pass")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_raw_histogram_quality.py"),
        Path(__file__).with_name("quality_histogram.py"), Path(__file__).with_name("run_raw_calibration.py"),
        Path(__file__).with_name("calibration_body_cache.py"), V1/"quality_engine.py", V1/"run_initial_pipeline.py",
        V2/"run_effective_initial.py", V2/"raw_initial_invariants.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1] or sources[2] != boundary["sources"][1] or sources[5] != oracle["sources"][2]:
        raise ValueError("The actual quality source differs from the original complete oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"),
        "boundary": data_identity(args.boundary/"RESULT.json"), "pairs": [], "runs": [],
        "minimum_complete_cold_quality_stage_reduction": .1, "maximum_each_whole_entry_ratio": 1.1,
        "whole_entry_median_cannot_regress": True, "maximum_group_seconds": 1800,
        "original_all7338_6797_1002_11583_and_all48_exclusions_unchanged": True,
        "all_caches_cold_new_private_dirs_and_JPEG_cache_off": True,
        "complete_quality_phase_includes_source_setup_raw_IO_decode_workers_first_cache_write_and_close": True,
        "same_current_Q78_Q50_Q53_P18_R99_and_original_formula_and_policy": True,
        "raw_quality_and_coarse_estimate_only_no_GT_or_SfM": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        for trial in range(3):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            order = ("native", "histogram") if trial % 2 == 0 else ("histogram", "native")
            durations, costs, details = {}, {}, {}
            for mode in order:
                if time.perf_counter()-started > 1800 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The bounded cold quality schedule or sources changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "600", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_raw_histogram_quality.py")),
                    "--quality-mode", mode, "--output", str(folder/mode), "--quality-cache", str(folder/(mode+"_quality"))]
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
                    raise RuntimeError("An actual cold complete quality entry failed")
                costs[mode] = read(folder/mode/"HISTOGRAM.json")
                details[mode] = read(folder/mode/"initial/SELECTION_RECEIPT.json")
                durations[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
            protections = {"pair": compare_initial_outputs(folder/"native/initial", folder/"histogram/initial"),
                "current_to_native": compare_initial_outputs(args.reference_initial, folder/"native/initial"),
                "current_to_histogram": compare_initial_outputs(args.reference_initial, folder/"histogram/initial")}
            atomic_json(folder/"INVARIANTS.json", protections)
            trace = costs["histogram"]["trace"]
            checks = {"all_current_original_raw_outputs_and_complete_pair_exact": all(row["all_protected"] for row in protections.values()),
                "all7338_original_quality_cache_misses_both_routes": all(row["timings"]["raw_image_scoring_detail"]["cache_hits"] == 0
                    and row["timings"]["raw_image_scoring_detail"]["submitted"] == 7338 for row in details.values()),
                "all7338_actual_histogram_calls_no_feature_pixel_or_frame_loss": trace["histogram_calls"] == 7338 and trace["native_calls"] == 0,
                "all_actual_sources_and_quality_aliases_restored": all(row["trace"]["aliases_restored"] for row in costs.values()),
                "complete_entry_pair_cost_protected": durations["histogram"] <= 1.1*durations["native"]}
            row = {"trial": trial, "order": order, "checks": checks, "whole_seconds": durations,
                "complete_cold_quality_stage_seconds": {mode: details[mode]["timings"]["raw_image_scoring_wall_seconds"]
                    +costs[mode]["additional_quality_lifecycle_seconds"] for mode in order},
                "complete_quality_details": {mode: details[mode]["timings"]["raw_image_scoring_detail"] for mode in order},
                "histogram_trace": trace}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print({key: value for key, value in row.items() if key != "complete_quality_details"}, flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete cold quality cost or output guard failed")
        median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "histogram")}
            for key in ("whole_seconds", "complete_cold_quality_stage_seconds")}
        reductions = {key: 1-value["histogram"]/value["native"] for key, value in median.items()}
        checks = {"all_three_complete_cold_raw_pairs_and_protections": len(state["pairs"]) == 3 and all(all(row["checks"].values()) for row in state["pairs"]),
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
