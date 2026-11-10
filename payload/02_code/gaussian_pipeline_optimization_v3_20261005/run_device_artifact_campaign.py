"""Measure all exact artifact outputs and three actual complete evaluation pairs."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "pair-source", "localization", "reference-run", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The native artifact oracle did not pass")
    paths = [Path(__file__), Path(__file__).with_name("evaluate_device_artifacts.py"),
        Path(__file__).with_name("device_artifact_picture.py"), Path(__file__).with_name("evaluate_dag_models.py"),
        V1/"evaluate_pair.py", V2/"reference_ssim.py", V2/"vectorized_gaussian_read.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1] or sources[4] != oracle["sources"][2]:
        raise ValueError("The complete evaluator does not use the same native artifact oracle source")
    inputs = []
    versions, iterations = set(), set()
    for label in ("continuous", "resumed"):
        state = read(args.pair_source/label/"STATE.json")
        manifest = read(args.pair_source/label/"INPUT_MANIFEST.json")
        if state["status"] != "completed":
            raise ValueError("Artifact comparison requires completed native model inputs")
        versions.add(manifest["input_version"])
        iterations.add(state["iteration"])
        inputs.extend(args.pair_source/label/name for name in ("STATE.json", "INPUT_MANIFEST.json",
            f"model/point_cloud/iteration_{state['iteration']}/point_cloud.ply"))
    if len(versions) != 1 or len(iterations) != 1:
        raise ValueError("Artifact comparison requires matching map version and training budget")
    version, iteration = versions.pop(), iterations.pop()
    inputs.append(args.localization)
    identities = [data_identity(path) for path in inputs]
    state = {"status": "running", "sources": sources, "inputs": identities,
        "oracle": data_identity(args.oracle/"RESULT.json"), "pairs": [], "runs": [],
        "minimum_actual_artifact_stage_reduction": .1, "minimum_actual_returned_tensor_byte_reduction": .7,
        "complete_entry_median_cannot_regress": True, "maximum_each_complete_entry_ratio": 1.1,
        "all24_original_metric_rows_and18_artifact_pixels_exact": True,
        "prepare_copy_source_identity_and_conversion_setup_in_complete_makespan": True,
        "actual_transfer_bytes_not_peak_RAM_CUDA_memory_or_FPS": True, "GPU_heavy_tasks": 1,
        "maximum_group_seconds": 600, "input_version": version, "iterations": iteration, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        for trial in range(3):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            durations, costs = {}, {}
            for mode in (("native", "device") if trial % 2 == 0 else ("device", "native")):
                before = time.perf_counter()
                if time.perf_counter()-started > 600 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The actual artifact comparison source or bounded schedule changed")
                pair = folder/mode/"pair"
                for source, label in (("continuous", "baseline"), ("resumed", "combined")):
                    for name in ("STATE.json", "INPUT_MANIFEST.json", f"model/point_cloud/iteration_{iteration}/point_cloud.ply"):
                        origin, destination = args.pair_source/source/name, pair/label/name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(origin, destination)
                        if data_identity(origin)["sha256"] != data_identity(destination)["sha256"]:
                            raise ValueError("An actual native artifact model copy changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "120", "--maximum-rss-gib", "16", "--",
                    sys.executable, str(Path(__file__).with_name("evaluate_device_artifacts.py")), "--mode", mode,
                    "--runtime", str(args.runtime), "--pair", str(pair), "--localization", str(args.localization),
                    "--reference-run", str(args.reference_run), "--input-version", version, "--iterations", str(iteration)]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                run.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("An actual complete original artifact evaluator failed")
                costs[mode] = read(pair/"evaluation/ARTIFACTS.json")
                durations[mode] = time.perf_counter()-before
            evaluations = {mode: read(folder/mode/"pair/evaluation/EVALUATION.json") for mode in ("native", "device")}
            native, device = costs["native"], costs["device"]
            picture_path = lambda mode: folder/mode/"pair/evaluation/SIX_VIEW_COMPARISON.jpg"
            checks = {"all24_two_model_metrics_exact": all(evaluations["native"]["results"][label]["rows"]
                    == evaluations["device"]["results"][label]["rows"] for label in ("baseline", "combined")),
                "full_original_denominators_and_quality_guards": all(value["all_frames_included_in_denominator"] == 24
                    and value["development_metric_guard_passed"] for value in evaluations.values()),
                "all18_full_unresized_artifact_pixels_exact": len(native["trace"]["calls"]) == len(device["trace"]["calls"]) == 18
                    and all(tuple(left[k] for k in ("shape", "pixel_sha256", "size", "mode"))
                        == tuple(right[k] for k in ("shape", "pixel_sha256", "size", "mode"))
                        for left, right in zip(native["trace"]["calls"], device["trace"]["calls"])),
                "every_actual_transfer_retains_all_pixels_as_uint8": all(row["route"] == "device_uint8"
                    and row["actual_returned_CPU_tensor_dtype"] == "torch.uint8"
                    and row["device_to_host_tensor_bytes"]*4 == row["original_float_tensor_bytes"]
                    for row in device["trace"]["calls"]),
                "actual_final_JPEG_bytes_and_decoded_pixels_exact": picture_path("native").read_bytes() == picture_path("device").read_bytes()
                    and np.array_equal(np.asarray(Image.open(picture_path("native"))), np.asarray(Image.open(picture_path("device")))),
                "complete_entry_pair_cost_protected": durations["device"] <= 1.1*durations["native"]}
            row = {"trial": trial, "checks": checks, "complete_makespan_seconds": durations,
                "complete_artifact_seconds": {mode: cost["complete_artifact_conversion_seconds_including_source_setup"] for mode, cost in costs.items()},
                "actual_device_to_host_tensor_bytes": {mode: cost["actual_device_to_host_tensor_bytes"] for mode, cost in costs.items()},
                "actual_GPU_resource_profile": {mode: read(folder/mode/"pair/evaluation/COMBINATION.json") for mode in ("native", "device")}}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print({key: row[key] for key in ("trial", "checks", "complete_makespan_seconds", "complete_artifact_seconds", "actual_device_to_host_tensor_bytes")}, flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete artifact output or entry cost protection failed")
        medians = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "device")}
            for key in ("complete_makespan_seconds", "complete_artifact_seconds", "actual_device_to_host_tensor_bytes")}
        reductions = {key: 1-values["device"]/values["native"] for key, values in medians.items()}
        checks = {"all_three_original_complete_pairs_protected": len(state["pairs"]) == 3
                and all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_entry_median_cannot_regress": reductions["complete_makespan_seconds"] >= 0,
            "at_least_one_preregistered_measured_stage_or_actual_transfer_benefit":
                reductions["complete_artifact_seconds"] >= .1 or reductions["actual_device_to_host_tensor_bytes"] >= .7,
            "all_sources_and_input_models_unchanged": sources == [data_identity(path) for path in paths]
                and identities == [data_identity(path) for path in inputs]}
        state.update(status="all_actual_artifact_protections_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
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
