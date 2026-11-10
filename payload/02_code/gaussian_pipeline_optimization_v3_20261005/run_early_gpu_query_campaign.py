"""Measure resumable fresh complete CPU/GPU pairs without replacing the first pair."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001"
OWN = OLD / "bootstrap_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_sift_process_campaign_005 import compare_pair


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_pair(cpu_path, gpu_path):
    began = time.perf_counter()
    cpu, gpu = (read(path / "RESULT.json") for path in (cpu_path, gpu_path))
    oracle = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001/c1/t0/c"
    cpu_exact = compare_pair(oracle, cpu_path)
    hc, hg = (read(value["localization"]) for value in (cpu, gpu))
    previous = read(OLD / "g0/RESULT.json")
    current = read(gpu_path / "gpu_feature_batch/RESULT.json")
    prior_rows = {row["name"]: row for row in previous["rows"]}
    rows = []
    for row in current["rows"]:
        before = Path(prior_rows[row["name"]]["feature_artifact"])
        after = Path(row["feature_artifact"])
        with np.load(before, allow_pickle=False) as left, np.load(after, allow_pickle=False) as right:
            fields = {name: left[name].dtype == right[name].dtype and left[name].shape == right[name].shape
                      and np.array_equal(left[name], right[name]) for name in ("keypoints", "descriptors")}
        rows.append({"name": row["name"], "checks": fields,
                     "before": data_identity(before), "after": data_identity(after)})
    old_inputs = {row["path"]: row for row in previous["inputs"]}
    checks = {"current_CPU2_R120_all_original_outputs_exact": cpu_exact["all_protected"],
        "both_original_complete_full_entry_guards": all(all(value["checks"].values()) for value in (cpu, gpu)),
        "both_common48_prepared_geometry_routes": all(len(value["common_array_geometry_trace"]["routes"]) == 48
            and all(route == "owned_exact_array" for route in value["common_array_geometry_trace"]["routes"])
            and value["common_array_geometry_trace"]["binding_restored"] for value in (cpu, gpu)),
        "same_complete_localization_inputs": cpu["inputs"] == gpu["inputs"],
        "same_all24_names_and_order": len(hc["rows"]) == len(hg["rows"]) == 24
            and [row["image_name"] for row in hc["rows"]] == [row["image_name"] for row in hg["rows"]],
        "same_original_camera_and_target_ROIs": hc["PINHOLE_camera"] == hg["PINHOLE_camera"]
            and [row["target_window_PINHOLE"] for row in hc["rows"]] == [row["target_window_PINHOLE"] for row in hg["rows"]],
        "all24_CPU_GPU_loc_eval_3D_disjoint": all(row["loc_eval_3D_overlap"] == 0 for result in (hc, hg) for row in result["rows"]),
        "same_original_GPU_parameters_and_pixels": current["original_CPU_options"] == previous["original_CPU_options"]
            and current["original_query_paths"] == previous["original_query_paths"] and current["gray_rows"] == previous["gray_rows"],
        "same_all24_original_GPU_feature_array_bits": len(rows) == 24 and all(all(row["checks"].values()) for row in rows),
        "same_all_original_GPU_source_inputs": all(old_inputs.get(row["path"]) == row for row in current["inputs"]),
        "actual_original_GPU_producer_success": all(current["checks"].values()) and current["CLI_returncode"] == 0,
        "both_original_argv_and_producer_closed": all(read(path / "BOOTSTRAP_CLOSE.json")["argv_restored"]
            and read(path / "BOOTSTRAP_CLOSE.json")["producer_drained_returncode"] in (None, 0) for path in (cpu_path, gpu_path))}
    trace = read(gpu_path / "BOOTSTRAP.json")["trace"]
    checks["producer_actually_started_before_heavy_imports_and_first_use"] = (
        trace["producer_start_monotonic"] < read(gpu_path / "BOOTSTRAP.json")["producer_started_before_heavy_imports_monotonic"]
        <= trace["first_use_monotonic"] and trace["broker"]["delivered"] == 24
        and trace["binding_restored"] and trace["broker"]["closed"])
    centers = [float(np.linalg.norm(np.asarray(a["camera_center_m"]) - np.asarray(b["camera_center_m"])))
               for a, b in zip(hc["rows"], hg["rows"])]
    return {"checks": checks, "all_protected": all(checks.values()), "CPU_current_oracle": cpu_exact,
        "all_GPU_feature_records": rows, "GPU_trace": trace,
        "CPU_GPU_center_differences_m_not_GT": {"maximum": max(centers), "mean": statistics.mean(centers)},
        "complete_compare_seconds": time.perf_counter() - began}


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWN.resolve())
    paths = [Path(__file__), CODE / "early_gpu_query_provider.py", CODE / "localize_early_gpu_query.py",
        CODE / "check_early_gpu_query_schedule.py", CODE / "localize_gpu_query_sift.py",
        CODE / "profile_gpu_query_sift.py", CODE / "gpu_query_backend.py", CODE / "gpu_query_node.py",
        CODE / "gpu_query_node_config.py", CODE / "run_bounded_journal_task.py", CODE / "array_match_geometry.py",
        CODE / "native_sift_process.py", CODE / "run_sift_process_campaign_005.py"]
    sources = [data_identity(path) for path in paths]
    generic = read(OWN / "g0/RESULT.json")
    if not all(generic["checks"].values()) or generic["sources"] != [data_identity(CODE / name) for name in
            ("check_early_gpu_query_schedule.py", "early_gpu_query_provider.py", "localize_early_gpu_query.py")]:
        raise ValueError("The exact current scheduling generic must pass before complete measurement")
    if args.output.exists():
        state = read(args.output / "STATE.json")
        if args.through != 3 or state["status"] != "first_complete_cost_positive_pending_three_pairs_and_quality" or state["sources"] != sources:
            raise ValueError("Only the same frozen successful first pair may resume")
    else:
        args.output.mkdir(parents=True)
        snapshots = args.output / "source_snapshot"
        snapshots.mkdir()
        for number, path in enumerate(paths):
            shutil.copyfile(path, snapshots / (str(number) + "_" + path.name))
        state = {"status": "running", "sources": sources, "orders": [["cpu_process", "gpu"], ["gpu", "cpu_process"], ["cpu_process", "gpu"]],
            "pairs": [], "runs": [], "invocations": [], "new_optimization_count": 0,
            "old_negative_measured_family_fees_seconds": 75.030935,
            "first_generic_fee_seconds": read(OWN / "GENERIC_LEASE_001.json")["child_elapsed_seconds"],
            "phase_and_whole_are_complete_parent_Popen_through_true_exit": True,
            "minimum_raw_first_complete_reduction": .05, "minimum_final_complete_reduction": .05,
            "other_quality_source_recovery_and_resource_guards_unchanged": True}
        atomic_json(args.output / "PROTOCOL.json", state)
    try:
        state["status"] = "running"
        atomic_json(args.output / "STATE.json", state)
        for trial in range(len(state["pairs"]), args.through):
            folder = args.output / ("t" + str(trial))
            folder.mkdir()
            order = state["orders"][trial]
            nodes, seconds = {}, {}
            for mode in order:
                if sources != [data_identity(path) for path in paths] or time.perf_counter() - began > 1700:
                    raise RuntimeError("The frozen source or complete campaign bound changed")
                name = "cpu" if mode == "cpu_process" else "gpu"
                node, output = folder / ("n_" + name), folder / name
                command = [sys.executable, "-B", str(CODE / "gpu_query_node.py"), "--kind", "localization", "--mode", mode,
                    "--node", str(node), "--output", str(output), "--", sys.executable, "-B",
                    str(CODE / "localize_early_gpu_query.py"), "--mode", mode, "--output", str(output)]
                row = {"trial": trial, "mode": mode, "command": command, "status": "running"}
                state["runs"].append(row)
                atomic_json(args.output / "STATE.json", state)
                launched = time.perf_counter()
                with (folder / (name + "_controller.log")).open("xb") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row.update(pid=process.pid, creation_time=psutil.Process(process.pid).create_time())
                    atomic_json(args.output / "STATE.json", state)
                    code = process.wait(timeout=1800)
                seconds[mode] = time.perf_counter() - launched
                row.update(returncode=code, status="completed" if code == 0 else "failed", complete_parent_exit_seconds=seconds[mode])
                atomic_json(args.output / "STATE.json", state)
                nodes[mode] = read(node / "NODE.json")
                if code or nodes[mode]["status"] != "completed":
                    raise RuntimeError("The complete original cold localization node failed")
            protected = scientific_pair(folder / "cpu", folder / "gpu")
            atomic_json(folder / "INVARIANTS.json", protected)
            reduction = 1 - seconds["gpu"] / seconds["cpu_process"]
            checks = {"all_scientific_invariants_preserved": protected["all_protected"],
                "both_actual_owned_Windows_Linux_gone": all(not node["live_Windows_after_wait"]
                    and not any(row["alive"] for row in node["actual_Linux_after_wait"]) for node in nodes.values()),
                "candidate_whole_within1_1": seconds["gpu"] <= 1.1 * seconds["cpu_process"]}
            pair = {"trial": trial, "order": order, "checks": checks, "raw_complete_seconds": seconds,
                "raw_complete_reduction": reduction, "complete_compare_seconds": protected["complete_compare_seconds"]}
            state["pairs"].append(pair)
            atomic_json(args.output / "STATE.json", state)
            print(json.dumps(pair), flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete scientific or whole-cost guard failed")
            if trial == 0 and reduction < .05:
                state.update(status="closed_negative_first_raw_complete_cost_screen", quality_formal_external_recovery="not_run_after_cost_screen_negative")
                break
        if state["status"] == "running":
            state["status"] = "first_complete_cost_positive_pending_three_pairs_and_quality" if len(state["pairs"]) == 1 else "three_complete_pairs_science_passed_original_quality_and_final_fees_pending"
        state["current_sources_unchanged"] = sources == [data_identity(path) for path in paths]
        if not state["current_sources_unchanged"]:
            raise ValueError("The measured complete source changed")
    except BaseException as error:
        state.update(status="failed_verification_complete_current_entry", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["invocations"].append({"through": args.through, "parent_observed_before_last_publication_seconds": time.perf_counter() - began})
        atomic_json(args.output / "STATE.json", state)
        atomic_json(args.output / "RESULT.json", state)


if __name__ == "__main__":
    main()
