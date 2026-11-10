"""Measure three complete fresh save pairs with retained first fees and owners."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from streamed_ply_node_config import ROOT, CODE, OUT, V1, read, atomic_json, data_identity

MODEL = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0/combined/model/point_cloud/iteration_12000/point_cloud.ply"
RUNTIME = ROOT / "04_outputs/box2_3dgs_display_v1_20260930"
LINUX = "/opt/e26_thin_surface_v1/venv/bin/python"


def mounted(path):
    return "/mnt/" + str(path)[0].lower() + str(path)[2:].replace("\\", "/")


def file_exact(left, right):
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            x, y = a.read(1024**2), b.read(1024**2)
            if x != y: return False
            if not x: return True


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    if not all(read(OUT / "g1/RESULT.json")["checks"].values()) or read(OUT / "ng1/NODE.json")["status"] != "completed":
        raise ValueError("A complete original generic and owned closure is required")
    paths = [Path(__file__), CODE / "streamed_gaussian_ply_oracle_002.py", CODE / "profile_streamed_gaussian_ply_002.py",
        CODE / "streamed_ply_node_002.py", CODE / "streamed_ply_node_config.py", CODE / "run_bounded_journal_task.py",
        V1 / "ply_serialization.py", V1 / "runtime.py", MODEL]
    identities = [data_identity(path) for path in paths]
    fee_names = ("GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json", "EVENT_LEASE_001.json", "GENERIC_LEASE_002.json", "EVENT_LEASE_002.json")
    fee = sum(read(OUT / name)["child_elapsed_seconds"] for name in fee_names) + .2883809
    state = {"status": "running", "sources_inputs": identities, "pairs": [], "first_generic_fee_seconds": fee,
        "orders": [["native", "stream"], ["stream", "native"], ["native", "stream"]],
        "minimum_complete_phase_reduction": .1, "maximum_each_whole_ratio": 1.1,
        "separate_side_whole_median_must_not_regress": True, "counted": False}
    atomic_json(args.output / "PROTOCOL.json", state)
    try:
        for trial, order in enumerate(state["orders"]):
            sides, folders = {}, {}
            for mode in order:
                begin = time.perf_counter()
                if identities != [data_identity(path) for path in paths]: raise ValueError("Source/input changed before a fresh side")
                node, folder = args.output / f"t{trial}_{mode}_node", args.output / f"t{trial}_{mode}"
                command = [sys.executable, "-B", str(CODE / "streamed_ply_node_002.py"), "--kind", "camera_oracle",
                    "--mode", "diagnostic", "--node", str(node), "--output", str(folder), "--",
                    "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", LINUX, "-B",
                    mounted(CODE / "profile_streamed_gaussian_ply_002.py"), "--mode", mode,
                    "--runtime", mounted(RUNTIME), "--model", mounted(MODEL), "--output", mounted(folder)]
                with (args.output / f"t{trial}_{mode}_controller.log").open("x", encoding="utf-8") as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    code = child.wait()
                if code:
                    state["failed_side"] = {"trial": trial, "mode": mode, "returncode": code}
                    raise RuntimeError("A fresh complete protected save side failed")
                result, owned = read(folder / "RESULT.json"), read(node / "NODE.json")
                if identities != [data_identity(path) for path in paths]: raise ValueError("Source/input changed after complete owner close")
                checks = {"all_original_side_guards": all(result["checks"].values()), "node_completed": owned["status"] == "completed",
                    "all_owned_instances_closed": not owned["live_Windows_after_wait"] and not any(row["alive"] for row in owned["actual_Linux_after_wait"]),
                    "controller_source_unchanged": owned["final_source_unchanged"]}
                if not all(checks.values()): raise AssertionError("A frozen original complete node guard failed")
                sides[mode] = {"raw_whole_seconds": time.perf_counter() - begin,
                    "raw_complete_phase_seconds": result["complete_phase_seconds"], "checks": checks,
                    "WSL_peak_RSS_bytes": result["WSL_peak_RSS_bytes"], "CUDA_peak_reserved_bytes": result["CUDA_peak_reserved_bytes"]}
                folders[mode] = folder
                print(json.dumps({"trial": trial, "mode": mode, **sides[mode]}), flush=True)
            before = time.perf_counter()
            checks = {"all_complete_point_file_bytes": file_exact(folders["native"] / "point_cloud.ply", folders["stream"] / "point_cloud.ply"),
                "all_independent_reference_artifact_bytes": file_exact(folders["native"] / "native_oracle.ply", folders["stream"] / "native_oracle.ply"),
                "each_whole_at_most1_1": sides["stream"]["raw_whole_seconds"] <= 1.1 * sides["native"]["raw_whole_seconds"]}
            comparison = time.perf_counter() - before
            state["pairs"].append({"trial": trial, "order": order, "sides": sides, "checks": checks, "complete_comparison_seconds": comparison})
            atomic_json(args.output / "STATE.json", state)
            if not all(checks.values()): raise AssertionError("A full file or complete whole-cost guard failed")
        shared_fee = fee + sum(row["complete_comparison_seconds"] for row in state["pairs"])
        medians = {key: {mode: statistics.median(row["sides"][mode]["raw_" + key] + shared_fee / 6 for row in state["pairs"])
            for mode in ("native", "stream")} for key in ("whole_seconds", "complete_phase_seconds")}
        reductions = {key: 1 - row["stream"] / row["native"] for key, row in medians.items()}
        checks = {"all_three_original_complete_pairs": len(state["pairs"]) == 3 and all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_phase_at_least10percent": reductions["complete_phase_seconds"] >= .1,
            "complete_whole_median_not_regressed": reductions["whole_seconds"] >= 0,
            "complete_sources_inputs_unchanged": identities == [data_identity(path) for path in paths]}
        state.update({"status": "cost_passed_root_supplement_and_review_pending" if all(checks.values()) else "closed_negative_complete_cost",
            "checks": checks, "median_seconds": medians, "reductions": reductions, "complete_shared_first_fee_seconds": shared_fee})
    except BaseException as error:
        state.update(status="failed_actual_guard", error=repr(error))
        raise
    finally:
        state["whole_campaign_seconds"] = time.perf_counter() - started
        atomic_json(args.output / "RESULT.json", state)
        print(json.dumps({key: state[key] for key in ("status", "checks", "median_seconds", "reductions") if key in state}), flush=True)


if __name__ == "__main__":
    main()
