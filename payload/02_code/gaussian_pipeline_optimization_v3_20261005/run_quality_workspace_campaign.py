"""Measure original full RAW consumers with a frozen resumable three-pair plan."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "pilot", "reference-initial"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--through", choices=(1, 3), type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001")
    generic, pilot = load(args.generic / "RESULT.json"), load(args.pilot / "RESULT.json")
    if generic["status"] != "generic_passed_pilot_and_complete_stage_pending" or not all(generic["checks"].values()) or pilot[
            "status"] != "component_screen_positive_complete_stage_pending" or pilot["reduction"] < .05:
        raise ValueError("A necessary frozen generic or complete component screen failed")
    paths = [Path(__file__), Path(__file__).with_name("run_raw_quality_workspace.py"),
             Path(__file__).with_name("native_quality_workspace.py"), Path(__file__).with_name("run_raw_histogram_quality.py"),
             Path(__file__).with_name("quality_histogram.py"), Path(__file__).with_name("run_raw_calibration.py"),
             Path(__file__).with_name("calibration_body_cache.py"), V1 / "quality_engine.py", V1 / "run_initial_pipeline.py",
             V2 / "run_effective_initial.py", V2 / "raw_initial_invariants.py", V1 / "run_bounded_task.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != generic["sources"][1] or sources[2] != pilot["sources"][1]:
        raise ValueError("A candidate source differs from its complete generic and pilot")
    if output.exists():
        protocol = load(output / "PROTOCOL.json")
        state = load(output / "STATE.json")
        if args.through != 3 or protocol["sources"] != sources or state["status"] != "first_complete_pair_positive_resume_same_frozen_schedule" or len(state["pairs"]) != 1:
            raise ValueError("Only the same positive complete first pair can resume without remeasurement")
        state["status"] = "running"
    else:
        output.mkdir(parents=True, exist_ok=False)
        protocol = {"sources": sources, "generic": str(args.generic), "pilot": str(args.pilot),
                    "reference_initial": str(args.reference_initial), "schedule": [["native", "workspace"], ["workspace", "native"], ["native", "workspace"]],
                    "raw": 7338, "eligible": 6797, "selected": 1002, "pairs": 11583,
                    "minimum_complete_phase_reduction": .05, "each_whole_ratio": 1.1, "whole_median_nonregression": True,
                    "all_known_first_failed_generic_pilot_external_review_publication_fees_divided_by6_for_final_acceptance": True,
                    "first_positive_pair_may_resume_without_repeating_it": True,
                    "any_first_real_science_or_cost_failure_stops_contingent": True, "counted": False}
        atomic_json(output / "PROTOCOL.json", protocol)
        snapshots = output / "source_snapshot"
        snapshots.mkdir()
        for index, path in enumerate(paths):
            (snapshots / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
        state = {"status": "running", "sources": sources, "pairs": [], "runs": [], "invocations": [], "counted": False}
    invocation = {"through": args.through, "first_pair_before_invocation": len(state["pairs"]), "status": "running"}
    state["invocations"].append(invocation)
    started = time.perf_counter()
    leaf_total = 0.
    try:
        for trial in range(len(state["pairs"]), args.through):
            folder = output / ("t" + str(trial))
            folder.mkdir()
            order = protocol["schedule"][trial]
            results = {}
            for mode in order:
                if time.perf_counter() - started > 1800 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The original whole campaign time or source guard changed")
                command = [sys.executable, str(V1 / "run_bounded_task.py"), "--receipt", str(folder / (mode + "_PROCESS.json")),
                           "--log", str(folder / (mode + ".log")), "--maximum-seconds", "600", "--maximum-rss-gib", "8", "--",
                           sys.executable, str(Path(__file__).with_name("run_raw_quality_workspace.py")), "--mode", mode,
                           "--output", str(folder / mode), "--quality-cache", str(folder / (mode + "_quality"))]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(output / "STATE.json", state)
                before = time.perf_counter()
                with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run.update(pid=process.pid, creation_time=psutil.Process(process.pid).create_time())
                    atomic_json(output / "STATE.json", state)
                    code = process.wait()
                whole = time.perf_counter() - before
                leaf_total += whole
                run.update(returncode=code, actual_complete_child_close_seconds=whole, status="completed" if code == 0 else "failed")
                atomic_json(output / "STATE.json", state)
                if code:
                    raise RuntimeError("An original complete cold RAW consumer failed")
                workspace = load(folder / mode / "WORKSPACE.json")
                histogram = load(folder / mode / "HISTOGRAM.json")
                detail = load(folder / mode / "initial/SELECTION_RECEIPT.json")
                results[mode] = {"whole_seconds": whole,
                    "phase_seconds": detail["timings"]["raw_image_scoring_wall_seconds"] + workspace["additional_workspace_lifecycle_seconds"]
                                      + histogram["additional_quality_lifecycle_seconds"],
                    "workspace": workspace, "histogram": histogram, "detail": detail}
            before = time.perf_counter()
            invariants = {"pair": compare_initial_outputs(folder / "native/initial", folder / "workspace/initial"),
                          "native_reference": compare_initial_outputs(args.reference_initial, folder / "native/initial"),
                          "workspace_reference": compare_initial_outputs(args.reference_initial, folder / "workspace/initial")}
            left, right = folder / "native_quality", folder / "workspace_quality"
            names = sorted(path.name for path in left.iterdir())
            cache_exact = len(names) == 7338 and all(name.endswith(".json") for name in names) and names == sorted(
                path.name for path in right.iterdir()) and all((left / name).read_bytes() == (right / name).read_bytes() for name in names)
            comparison = time.perf_counter() - before
            workspace = results["workspace"]["workspace"]["trace"]
            checks = {"all_original7338_decisions1002_JPEG_pairs_K_poses_and_current_reference_exact": all(
                result["all_protected"] for result in invariants.values()), "all7338_original_cache_files_bytes_exact": cache_exact,
                "all_raw_cold_denominators": all((result["detail"]["raw_frame_count"], result["detail"]["quality_and_coarse_pose_eligible_count"],
                    result["detail"]["selected_count"], result["detail"]["matching_candidate_pair_count"], result["detail"]["timings"]["raw_image_scoring_detail"]["cache_hits"])
                    == (7338, 6797, 1002, 11583, 0) for result in results.values()),
                "all_original_quality_aliases_and_thread_planes_closed": all(result["histogram"]["trace"]["aliases_restored"]
                    and result["workspace"]["trace"]["original_builder_restored"] for result in results.values())
                    and workspace["workspace"]["closed"] and workspace["workspace"]["owned_plane_bytes_after_close"] == 0
                    and workspace["workspace"]["calls"] == 3 * 7338 and workspace["workspace"]["thread_count"] <= 4,
                "each_complete_whole_at_most1point1": results["workspace"]["whole_seconds"] <= 1.1 * results["native"]["whole_seconds"]}
            pair = {"trial": trial, "order": order, "results": results, "comparison_seconds_charged_each_side": comparison,
                    "invariants": invariants, "checks": checks,
                    "phase_seconds": {mode: results[mode]["phase_seconds"] + comparison for mode in order},
                    "whole_seconds": {mode: results[mode]["whole_seconds"] + comparison for mode in order}}
            state["pairs"].append(pair)
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": pair["phase_seconds"], "whole_seconds": pair["whole_seconds"], "checks": checks}, flush=True)
            if not all(checks.values()):
                raise ValueError("A necessary complete original initial pipeline guard failed")
            if trial == 0 and (1 - pair["phase_seconds"]["workspace"] / pair["phase_seconds"]["native"] < .05
                              or pair["whole_seconds"]["workspace"] > pair["whole_seconds"]["native"]):
                state["status"] = "first_complete_pair_negative_stop_no_formal_acceptance"
                break
        if state["status"] == "running":
            state["status"] = "first_complete_pair_positive_resume_same_frozen_schedule" if args.through == 1 else "three_complete_fresh_pairs_finished_all_fees_supplement_and_review_pending"
        if len(state["pairs"]) == 3:
            state["medians"] = {name: {mode: statistics.median(pair[name][mode] for pair in state["pairs"])
                                      for mode in ("native", "workspace")} for name in ("phase_seconds", "whole_seconds")}
            state["reductions_before_all_shared_fees"] = {name: 1 - value["workspace"] / value["native"] for name, value in state["medians"].items()}
        invocation.update(status="completed", actual_invocation_seconds=time.perf_counter() - started,
                          measured_leaf_envelopes_seconds=leaf_total, nonleaf_seconds=time.perf_counter() - started - leaf_total)
        atomic_json(output / "STATE.json", state)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "complete_pairs": len(state["pairs"]), "counted": False}, flush=True)
    except BaseException as error:
        invocation.update(status="failed", actual_invocation_seconds=time.perf_counter() - started, measured_leaf_envelopes_seconds=leaf_total)
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(output / "STATE.json", state)
        atomic_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
