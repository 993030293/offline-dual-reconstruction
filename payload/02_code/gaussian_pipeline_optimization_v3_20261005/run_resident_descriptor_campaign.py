"""Measure all current pose outputs and the complete resident-worker lifecycle."""

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


def entry(root):
    found = list(root.rglob("H_LOCALIZATION.json"))
    if len(found) != 1 or found[0].parent.name != "localization":
        raise ValueError("The original complete pose output layout is ambiguous")
    return found[0].parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "oracle", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The full native descriptor residency oracle did not pass")
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in ("descriptor_residency_plan.py",
        "resident_sift_worker.py", "resident_sift_matching.py", "integer_sift_worker.py", "integer_sift_matching.py",
        "generic_sift_worker.py", "generic_sift_matching.py", "localize_resident_descriptors.py",
        "query_preparation_pipeline.py", "localize_prepared_queries.py"))]
    sources = [data_identity(path) for path in paths]
    oracle_sources = {row["path"]: row for row in oracle["sources"]}
    if any(source != oracle_sources[source["path"]] for source in sources[1:8]):
        raise ValueError("The actual resident worker source differs from the complete native oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"),
        "pairs": [], "runs": [], "minimum_complete_worker_stage_reduction": .1,
        "whole_entry_median_cannot_regress": True, "maximum_each_whole_entry_ratio": 1.1,
        "same_current_R90_R92_R94_R102_all24_48_no_feature_reduction": True,
        "maximum_CUDA_bytes": 4*1024**3, "maximum_resident_bytes": 64*1024**2,
        "includes_original_numerical_validation_and_all_new_cache_lifecycle": True,
        "maximum_group_seconds": 600, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        for trial in range(3):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            order = ("incumbent", "resident") if trial % 2 == 0 else ("resident", "incumbent")
            costs, durations = {}, {}
            for mode in order:
                if time.perf_counter()-started > 600 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The bounded descriptor residency schedule or sources changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_resident_descriptors.py")),
                    "--transport-mode", mode, "--output", str(folder/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache",
                    "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                run = {"trial": trial, "mode": mode, "command": command, "status": "running"}
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
                    raise RuntimeError("An actual complete resident descriptor entry failed")
                costs[mode] = read(folder/mode/"RESIDENCY.json")
                durations[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
            left, right = entry(folder/"incumbent"), entry(folder/"resident")
            protections = compare(left, right, left/"NEIGHBOR_TRACE.json", right/"NEIGHBOR_TRACE.json")
            atomic_json(folder/"INVARIANTS.json", protections)
            worker, shutdown = costs["resident"]["worker"], costs["resident"]["worker"]["shutdown"]
            checks = {"all_original24_48_matches_F_poses_features_PNG_inputs_exact": protections["all_protected"],
                "all48_original_batches_kept_both_workers_closed": all(len(row["worker"]["calls"]) == 48
                    and row["worker"]["returncode"] == 0 for row in costs.values()),
                "actual_content_reuse_reduces_descriptor_wire_bytes": shutdown["wire_descriptor_bytes"] < shutdown["original_wire_descriptor_bytes"]
                    and any(row["plan"]["missing_count"] == 0 for row in worker["calls"]),
                "all_original_GPU_budgets_and_residency_cleared": all(row["worker"]["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for row in costs.values())
                    and shutdown["peak_resident_descriptor_bytes"] <= 64*1024**2
                    and shutdown["resident_entries_after_clear"] == worker["client_resident_entries_after_close"] == 0,
                "all_original_worker_factories_restored": all(row["adapter"]["original_worker_factory_restored"] for row in costs.values()),
                "complete_entry_pair_cost_protected": durations["resident"] <= 1.1*durations["incumbent"]}
            row = {"trial": trial, "order": order, "checks": checks, "whole_seconds": durations,
                "complete_worker_stage_seconds": {mode: item["complete_matching_worker_stage_seconds"] for mode, item in costs.items()},
                "resident_shutdown": shutdown}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete resident descriptor output, cost or resource guard failed")
        medians = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("incumbent", "resident")}
            for key in ("whole_seconds", "complete_worker_stage_seconds")}
        reductions = {key: 1-value["resident"]/value["incumbent"] for key, value in medians.items()}
        checks = {"all_three_actual_complete_pairs_and_protections": len(state["pairs"]) == 3
                and all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_worker_stage_reduces10percent": reductions["complete_worker_stage_seconds"] >= .1,
            "whole_entry_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="all_actual_residency_guards_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
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
