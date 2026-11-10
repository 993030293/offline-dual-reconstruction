"""Compare complete current localization and three actual neighborhood scheduling triplets."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def entry(folder):
    found = list(folder.rglob("H_LOCALIZATION.json"))
    if len(found) != 1:
        raise ValueError("The owned actual pose entry must be unambiguous")
    if found[0].parent.name != "localization":
        raise ValueError("The original actual pose output directory changed")
    return found[0].parent.parent


def concurrency(jobs):
    events = sorted((value, change) for row in jobs for value, change in ((row["start"], 1), (row["end"], -1)))
    active, peak = 0, 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "oracle", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()) or len(oracle["rows"]) != 180:
        raise ValueError("The complete original heterogeneous geometry oracle did not pass")
    paths = [Path(__file__), Path(__file__).with_name("parallel_neighbor_geometry.py"),
        Path(__file__).with_name("localize_neighbor_geometry.py"), Path(__file__).with_name("native_observation_index.py"),
        Path(__file__).with_name("localize_native_observations.py"), ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py"]
    sources = [data_identity(path) for path in paths]
    if sources[1] != oracle["sources"][1] or sources[5] != oracle["sources"][2]:
        raise ValueError("The current complete geometry differs from the full original oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"),
        "runs": [], "triplets": [], "minimum_complete_CPU_geometry_reduction_against_incumbent": .1,
        "whole_entry_median_cannot_regress": True, "maximum_each_complete_entry_ratio": 1.1,
        "cold_snapshot_full_verification_first_pool_setup_and_shutdown_in_stage": True,
        "same_current_R90_R92_R94_and_all24_48_inputs": True, "F_parameters_seed_PnP_and_neighbor_order_unchanged": True,
        "maximum_group_seconds": 600, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        orders = (("incumbent", "serial", "parallel"), ("parallel", "incumbent", "serial"), ("serial", "parallel", "incumbent"))
        for trial, order in enumerate(orders):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            duration, costs = {}, {}
            for mode in order:
                if time.perf_counter()-started > 600 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The actual geometry source or bounded schedule changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_neighbor_geometry.py")), "--geometry-mode", mode,
                    "--output", str(folder/mode), "--reference-mode", "cache", "--reference-cache-root", str(args.reference_cache),
                    "--feature-mode", "cache", "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                run.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("An actual complete geometry pose entry failed")
                duration[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
                costs[mode] = read(folder/mode/"GEOMETRY.json")
            entries = {mode: entry(folder/mode) for mode in order}
            protections = {mode: compare(entries["incumbent"], entries[mode], entries["incumbent"]/"NEIGHBOR_TRACE.json", entries[mode]/"NEIGHBOR_TRACE.json")
                for mode in ("serial", "parallel")}
            atomic_json(folder/"INVARIANTS.json", protections)
            trace = costs["parallel"]["trace"]
            checks = {"both_new_routes_all_original24_48_matches_F_and_poses_preserved": all(row["all_protected"] for row in protections.values()),
                "complete_native_map_and_all48_index_calls_preserved": all(cost["complete_original_index_trace"]["index"]["complete_native_state_unchanged"]
                    and cost["complete_original_index_trace"]["calls"] == 48 for cost in costs.values()),
                "owned_pool_and_all48_parallel_admissions_closed": trace["closed"] and len(trace["routes"]) == 48
                    and all(row["route"] == "prepared_parallel" for row in trace["routes"]),
                "actual_independent_F_intervals_overlap": concurrency(trace["geometry_jobs"]) >= 2,
                "explicit_queue_capacity_preserved": trace["peak_pending"] <= 11,
                "complete_entry_pair_cost_protected": duration["parallel"] <= 1.1*duration["incumbent"]}
            row = {"trial": trial, "order": order, "checks": checks, "whole_seconds": duration,
                "complete_CPU_geometry_seconds": {mode: cost["complete_CPU_geometry_seconds"] for mode, cost in costs.items()},
                "parallel_jobs": len(trace["geometry_jobs"]), "actual_peak_overlapping_F_intervals": concurrency(trace["geometry_jobs"])}
            state["triplets"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete geometry or actual ownership cost guard failed")
        medians = {key: {mode: statistics.median(row[key][mode] for row in state["triplets"]) for mode in ("incumbent", "serial", "parallel")}
            for key in ("whole_seconds", "complete_CPU_geometry_seconds")}
        reductions = {key: 1-values["parallel"]/values["incumbent"] for key, values in medians.items()}
        checks = {"all_three_actual_complete_triplets_preserved": len(state["triplets"]) == 3
                and all(all(row["checks"].values()) for row in state["triplets"]),
            "complete_CPU_stage_reduces10percent_against_untouched_incumbent": reductions["complete_CPU_geometry_seconds"] >= .1,
            "complete_entry_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="all_actual_neighbor_geometry_guards_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median=medians, reductions=reductions, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
