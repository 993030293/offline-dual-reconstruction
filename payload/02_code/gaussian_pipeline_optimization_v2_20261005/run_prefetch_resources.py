"""Complete three full-input prefetch pairs without repeating a valid first pair."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pair_checks(control, candidate):
    keys = ("rows", "initial_Gaussians", "before_RNG", "after_RNG", "camera_manifest", "camera_extent")
    return {**{key+"_exact": control[key] == candidate[key] for key in keys},
        "all_worker_guards": all(control["checks"].values()) and all(candidate["checks"].values()),
        "peak_RAM_ratio_at_most1_15": candidate["peak_RSS_after_load_bytes"] <= 1.15*control["peak_RSS_after_load_bytes"],
        "same_GPU_payload": control["peak_CUDA_reserved_bytes"] == candidate["peak_CUDA_reserved_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "baseline", "prefetch", "lifecycle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source_paths = [Path(__file__).with_name(name) for name in
                    ("prefetch_camera_images.py", "validate_camera_prefetch.py")]
    sources = [data_identity(path) for path in source_paths]
    lifecycle = read(args.lifecycle/"RESULT.json")
    if not all(lifecycle["checks"].values()) or lifecycle["adapter_source"] != sources[0]:
        raise ValueError("Prefetch lifecycle or source identity is not verified")
    first = {mode: read(path/"RESULT.json") for mode, path in (("baseline", args.baseline), ("prefetch", args.prefetch))}
    if not all(pair_checks(first["baseline"], first["prefetch"]).values()):
        raise ValueError("The actual first pair failed before repeat measurements")
    for path in (args.baseline, args.prefetch):
        protocol = read(path/"PROTOCOL.json")
        if protocol["adapter_source"] != sources[0] or protocol["validator_source"] != sources[1]:
            raise ValueError("The first full-input resource pair uses different sources")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "running", "controller_pid": os.getpid(), "sources": sources,
        "runs": [], "first_pair_sources": [data_identity(path/"RESULT.json") for path in (args.baseline, args.prefetch)],
        "lifecycle": data_identity(args.lifecycle/"RESULT.json"), "counted": False}
    atomic_json(args.output/"STATE.json", state)
    pairs = [first]
    for trial, order in ((1, ("prefetch", "baseline")), (2, ("baseline", "prefetch"))):
        pair = {}
        for mode in order:
            if sources != [data_identity(path) for path in source_paths]:
                raise ValueError("Prefetch sources changed between paired trials")
            name = f"trial{trial}_{mode}"
            output = args.output/name
            command = [sys.executable, str(V1/"run_bounded_task.py"),
                "--receipt", str(args.output/(name+"_process.json")), "--log", str(args.output/(name+".log")),
                "--maximum-seconds", "300", "--maximum-rss-gib", "32", "--", sys.executable,
                str(source_paths[1]), "--mode", mode, "--runtime", str(args.runtime),
                "--dataset", str(args.dataset), "--output", str(output)]
            with (args.output/(name+"_controller.log")).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                row = {"trial": trial, "mode": mode, "status": "running", "pid": process.pid, "command": command}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                code = process.wait()
            row.update(status="completed" if code == 0 else "failed", returncode=code)
            atomic_json(args.output/"STATE.json", state)
            if code:
                raise RuntimeError("A prefetch resource worker failed")
            pair[mode] = read(output/"RESULT.json")
            print(json.dumps({"trial": trial, "mode": mode, "Scene_load_seconds": pair[mode]["Scene_load_seconds"]}), flush=True)
        pairs.append(pair)
        if not all(pair_checks(pair["baseline"], pair["prefetch"]).values()):
            raise ValueError("A repeated prefetch pair violated the frozen input or memory guard")
    times = {mode: statistics.median(p[mode]["Scene_load_seconds"] for p in pairs) for mode in ("baseline", "prefetch")}
    gain = 1-times["prefetch"]/times["baseline"]
    checks = {"three_complete_pairs": len(pairs) == 3,
        "all1016_inputs_RNG_and_initialization_exact": all(all(pair_checks(p["baseline"], p["prefetch"]).values()) for p in pairs),
        "Scene_load_gain_at_least10percent": gain >= .1,
        "all_lifecycle_checks": all(lifecycle["checks"].values()),
        "sources_unchanged": sources == [data_identity(path) for path in source_paths]}
    result = {"status": "input_resource_guards_passed_training_pending" if all(checks.values()) else "failed",
        "checks": checks, "median_Scene_load_seconds": times, "stage_time_reduction": gain,
        "pair_checks": [pair_checks(p["baseline"], p["prefetch"]) for p in pairs],
        "pair_resource_metrics": [{mode: {k: p[mode][k] for k in ("Scene_load_seconds", "RSS_after_load_bytes", "peak_RSS_after_load_bytes", "peak_CUDA_reserved_bytes")} for mode in p} for p in pairs],
        "sources": sources, "elapsed_seconds": time.monotonic()-started, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    state.update(status=result["status"], elapsed_seconds=result["elapsed_seconds"])
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({k: result[k] for k in ("status", "median_Scene_load_seconds", "stage_time_reduction", "checks")}))
    if not all(checks.values()):
        raise ValueError("The original resource admission failed")


if __name__ == "__main__":
    main()
