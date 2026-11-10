"""Measure three full1016 Scene pairs with exact metadata and original initialization."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_prefetch_resources import pair_checks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wsl_path(path):
    value = str(Path(path).resolve()).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("An explicit Windows drive path is required for the WSL controller")
    return "/mnt/"+value[0].lower()+value[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "dataset", "runtime", "validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--linux-python", required=True)
    args = parser.parse_args()
    lifecycle = read(args.validation)
    adapter = Path(__file__).with_name("training_camera_metadata.py")
    if lifecycle["status"] != "lifecycle_passed" or lifecycle["sources"][1] != data_identity(adapter):
        raise ValueError("The exact training metadata lifecycle or source identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), adapter, Path(__file__).with_name("load_with_training_metadata.py"),
        Path(__file__).with_name("validate_camera_prefetch.py"), Path(__file__).with_name("prefetch_camera_images.py")]
    sources = [data_identity(p) for p in paths]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshot/path.name)
    state = {"status": "running", "sources": sources, "pairs": [], "runs": [], "counted": False,
        "all1016_camera_pixels_K_transforms_order_RNG_initial_Gaussians_exact": True,
        "common_Q31_prefetch_on_both_sides": True, "minimum_reader_stage_speed_or_Scene_speed_or_peak_RAM_reduction": .1,
        "absolute_reader_and_complete_Scene_costs_disclosed": True,
        "complete_Scene_time_ratio_max": 1.1, "maximum_seconds_each": 300,
        "maximum_RSS_GiB": 32, "original_GPU_headroom_required": True,
        "all_observation_sources_retained": True, "three_actual_pairs": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "metadata"] if trial % 2 == 0 else ["metadata", "baseline"]):
                if sources != [data_identity(p) for p in paths]:
                    raise ValueError("The metadata input implementation changed")
                command = ["wsl.exe", "-d", args.distribution, "--", args.linux_python,
                    wsl_path(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005/run_bounded_task.py"),
                    "--receipt", wsl_path(pair/(mode+"_process.json")), "--log", wsl_path(pair/(mode+"_entry.log")),
                    "--maximum-seconds", "300", "--maximum-rss-gib", "32", "--", args.linux_python,
                    wsl_path(Path(__file__).with_name("load_with_training_metadata.py")), "--metadata-mode", mode,
                    "--runtime", wsl_path(args.runtime), "--dataset", wsl_path(args.dataset),
                    "--output", wsl_path(pair/mode)]
                row = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(row)
                with (pair/(mode+".log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait(timeout=360)
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("An actual metadata Scene entry failed")
            results = {mode: read(pair/mode/"scene_entry/RESULT.json") for mode in ("baseline", "metadata")}
            guards = pair_checks(results["baseline"], results["metadata"])
            traces = {mode: read(pair/mode/"METADATA_TRACE.json") for mode in results}
            item = {"trial": trial, "all_scene_values_and_RNG_exact": all(guards.values()), "guards": guards,
                "Scene_seconds": {mode: r["Scene_load_seconds"] for mode, r in results.items()},
                "reader_seconds": {mode: r["records"][0]["seconds"] for mode, r in traces.items()},
                "peak_RSS_bytes": {mode: r["peak_RSS_after_load_bytes"] for mode, r in results.items()},
                "RSS_after_load_bytes": {mode: r["RSS_after_load_bytes"] for mode, r in results.items()},
                "same_peak_CUDA_reserved_bytes": guards["same_GPU_payload"],
                "metadata_view": traces["metadata"]["records"]}
            item["complete_time_protected"] = item["Scene_seconds"]["metadata"] <= 1.1*item["Scene_seconds"]["baseline"]
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not item["all_scene_values_and_RNG_exact"] or not item["complete_time_protected"]:
                raise ValueError("The full Scene metadata optimization failed an original input guard")
        medians = {kind: {mode: statistics.median(p[kind][mode] for p in state["pairs"]) for mode in ("baseline", "metadata")}
            for kind in ("reader_seconds", "Scene_seconds", "peak_RSS_bytes", "RSS_after_load_bytes")}
        speed = 1-medians["Scene_seconds"]["metadata"]/medians["Scene_seconds"]["baseline"]
        reader_speed = 1-medians["reader_seconds"]["metadata"]/medians["reader_seconds"]["baseline"]
        memory = 1-medians["peak_RSS_bytes"]["metadata"]/medians["peak_RSS_bytes"]["baseline"]
        result = {"status": "resource_guards_passed_training_combination_pending" if max(reader_speed, speed, memory) >= .1 else "rejected",
            "median": medians, "Scene_time_reduction": speed, "peak_RSS_reduction": memory,
            "reader_stage_reduction": reader_speed,
            "absolute_reader_seconds_saved": medians["reader_seconds"]["baseline"]-medians["reader_seconds"]["metadata"],
            "pairs": state["pairs"], "sources": sources, "sources_unchanged": sources == [data_identity(p) for p in paths],
            "counted": False, "all_original_observations_preserved_on_disk": True,
            "training_and_final_quality_pending": True, "cold_path_without_extra_disk_cache": True}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
