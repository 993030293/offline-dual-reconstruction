"""Compare three isolated complete Scenes with exact normalization admission."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from run_prefetch_resources import pair_checks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runtime", "dataset", "oracle", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The native normalization oracle is not admitted")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("exact_image_normalization.py"),
             Path(__file__).with_name("load_with_exact_normalization.py"),
             Path(__file__).with_name("check_exact_image_normalization.py"),
             V2/"validate_camera_prefetch.py", V2/"camera_execution.py", V2/"prefetch_camera_images.py"]
    sources = [data_identity(path) for path in paths]
    if sources[1] != oracle["sources"][1]:
        raise ValueError("The oracle and actual image conversion source differ")
    (args.output/"sources").mkdir()
    for path in paths:
        shutil.copyfile(path, args.output/"sources"/path.name)
    protocol = {"sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"), "runs": [], "pairs": [],
        "three_full1016_isolated_pairs": True, "both_common_Q31_Q58_Q59_Q60": True,
        "original_float32_CPU_residency_copy_loss_optimizer_and_sampling_unchanged": True,
        "minimum_stage_call_plus_first_setup_reduction": .1, "maximum_complete_Scene_ratio": 1.1,
        "maximum_peak_RSS_ratio": 1.15, "maximum_group_seconds": 900,
        "accumulated_parallel_worker_call_seconds_are_not_real_Scene_wall_seconds": True,
        "full_three_seed12k_quality_and_recovery_required": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    state = {**protocol, "status": "running"}
    started = time.perf_counter()
    try:
        for trial in range(3):
            pair = {}
            for mode in (("native", "lookup") if trial % 2 == 0 else ("lookup", "native")):
                if sources != [data_identity(path) for path in paths]:
                    raise ValueError("A frozen normalization source changed")
                name = f"trial{trial}_{mode}"
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(name+"_process.json")),
                    "--log", str(args.output/(name+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "32", "--",
                    sys.executable, str(Path(__file__).with_name("load_with_exact_normalization.py")),
                    "--normalization-mode", mode, "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(args.output/name)]
                row = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                with (args.output/(name+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(returncode=code, status="completed" if code == 0 else "failed")
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("A complete Scene normalization worker failed")
                pair[mode] = read(args.output/name/"RESULT.json")
                print(json.dumps({"trial": trial, "mode": mode, "Scene_seconds": pair[mode]["Scene_load_seconds"],
                                  "normalization": pair[mode]["normalization_traces"][0]["normalization"]}), flush=True)
            checks = pair_checks(pair["native"], pair["lookup"])
            checks["real_Scene_cost_protected"] = pair["lookup"]["Scene_load_seconds"] <= 1.1*pair["native"]["Scene_load_seconds"]
            traces = {mode: pair[mode]["normalization_traces"][0]["normalization"] for mode in pair}
            checks["all1016_input_calls_preserved"] = all(t["calls"] == 1016 for t in traces.values())
            checks["one_only_conversion_mechanism"] = traces["lookup"]["lookup_calls"] == traces["native"]["native_calls"] == 1016
            state["pairs"].append({"trial": trial, "checks": checks,
                "Scene_seconds": {mode: pair[mode]["Scene_load_seconds"] for mode in pair},
                "phase_seconds": {mode: traces[mode]["call_seconds"]+traces[mode]["setup_seconds"] for mode in pair},
                "peak_RSS_bytes": {mode: pair[mode]["peak_RSS_after_load_bytes"] for mode in pair}})
            atomic_json(args.output/"STATE.json", state)
            if not all(checks.values()) or time.perf_counter()-started > 900:
                raise ValueError("A full Scene normalization invariant or schedule failed")
        median = {key: {mode: statistics.median(pair[key][mode] for pair in state["pairs"]) for mode in ("native", "lookup")}
                  for key in ("Scene_seconds", "phase_seconds", "peak_RSS_bytes")}
        reduction = 1-median["phase_seconds"]["lookup"]/median["phase_seconds"]["native"]
        state.update(status="stage_passed_training_quality_pending" if reduction >= .1 else "rejected_insufficient_stage_effect",
            median=median, stage_reduction=reduction, real_Scene_reduction=1-median["Scene_seconds"]["lookup"]/median["Scene_seconds"]["native"],
            elapsed_seconds=time.perf_counter()-started, sources_unchanged=sources == [data_identity(path) for path in paths])
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "median", "stage_reduction", "real_Scene_reduction")}), flush=True)


if __name__ == "__main__":
    main()
