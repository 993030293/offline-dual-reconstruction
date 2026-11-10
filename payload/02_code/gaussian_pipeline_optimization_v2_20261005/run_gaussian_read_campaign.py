"""Validate three complete Gaussian reads and identical actual24 render metrics."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "control", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("vectorized_gaussian_read.py"),
        Path(__file__).with_name("validate_gaussian_reader.py"), Path(__file__).with_name("evaluate_gaussian_read.py")]
    sources = [data_identity(p) for p in paths]
    model = args.control/"model/point_cloud/iteration_12000/point_cloud.ply"
    state = {"status": "running", "sources": sources, "pairs": [], "counted": False,
        "model": data_identity(model), "minimum_full_model_load_wall_reduction": .1,
        "all_native_tensors_values_strides_grad_flags_exact": True, "all24_full_and_target_metrics_exact": True,
        "maximum_RAM_GiB": 32, "maximum_CUDA_GiB": 12,
        "original_source_unchanged": True, "no_new_training_or_metric_quality_claim": True}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(command, name, seconds=180):
        if sources != [data_identity(p) for p in paths]:
            raise ValueError("An exact Gaussian read source changed")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(name+"_process.json")),
            "--log", str(args.output/(name+".log")), "--maximum-seconds", str(seconds), "--maximum-rss-gib", "32", "--", *command]
        with (args.output/(name+"_controller.log")).open("x", encoding="utf-8") as stream:
            code = subprocess.call(bounded, stdout=stream, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError("A full native Gaussian read or evaluation failed")

    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "vectorized"] if trial % 2 == 0 else ["vectorized", "baseline"]):
                run([sys.executable, str(Path(__file__).with_name("validate_gaussian_reader.py")), "--runtime", str(args.runtime),
                    "--model", str(model), "--output", str(pair/mode), "--mode", mode], f"trial{trial}_"+mode)
            rows = {mode: read(pair/mode/"RESULT.json") for mode in ("baseline", "vectorized")}
            exact = all(rows["baseline"][key] == rows["vectorized"][key] for key in ("parameters", "active_degree", "maximum_degree", "model"))
            item = {"trial": trial, "all_native_parameters_exact": exact,
                "load_seconds": {mode: value["load_seconds"] for mode, value in rows.items()},
                "peak_RSS_bytes": {mode: value["peak_RSS_bytes"] for mode, value in rows.items()},
                "peak_CUDA_reserved_bytes": {mode: value["peak_CUDA_reserved_bytes"] for mode, value in rows.items()}}
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not exact:
                raise ValueError("A native Gaussian parameter layout or value changed")
        pair = args.output/"actual24"
        for label in ("reference", "vectorized"):
            for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
                target = pair/label/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.control/name, target)
                if data_identity(target)["sha256"] != data_identity(args.control/name)["sha256"]:
                    raise ValueError("A native frozen evaluation model copy changed")
        run([sys.executable, str(Path(__file__).with_name("evaluate_gaussian_read.py")), "--pair", str(pair),
            "--runtime", str(args.runtime), "--reference-run", str(args.reference_run), "--localization", str(args.localization),
            "--input-version", "new_map_admission_v1"], "actual24_evaluation", 300)
        evaluation = read(pair/"evaluation/EVALUATION.json")
        left, right = [evaluation["results"][label] for label in ("reference", "vectorized")]
        if left["rows"] != right["rows"] or left["summary"] != right["summary"] or not evaluation["development_metric_guard_passed"]:
            raise ValueError("A complete actual24 render metric changed after Gaussian readback")
        median = {mode: statistics.median(p["load_seconds"][mode] for p in state["pairs"]) for mode in ("baseline", "vectorized")}
        reduction = 1-median["vectorized"]/median["baseline"]
        state.update(status="stage_optimization_passed_lifecycle_pending" if reduction >= .1 else "rejected_insufficient_benefit",
            median_load_seconds=median, load_reduction=reduction, all24_pixels_metrics_preserved=True,
            original_model_identity_unchanged=state["model"] == data_identity(model))
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
