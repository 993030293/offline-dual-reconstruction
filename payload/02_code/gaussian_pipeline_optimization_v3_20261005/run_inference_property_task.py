"""Launch bounded generic, paired, or failure diagnostics with complete cost windows."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/aw001"
MODELS = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0"
LOC = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def linux(path):
    return "/mnt/c/" + Path(path).as_posix()[3:]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(output, script, arguments, expect_error=False):
    from nvml_memory_monitor_002 import NativeMemory
    command = [sys.executable, str(CODE / "run_bounded_journal_task.py"), "--receipt", str(output / "WINDOWS_GUARD.json"),
        "--log", str(output / "RUN.log"), "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--",
        "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        linux(CODE / script), *arguments]
    memory = NativeMemory()
    samples, failures, stop = [], [], threading.Event()
    def sample():
        while not stop.wait(.1):
            try:
                samples.append(memory.read("v2"))
            except BaseException as error:
                failures.append(repr(error))
                return
    thread = threading.Thread(target=sample, name="owned-R139-GPU-budget", daemon=True)
    try:
        samples.append(memory.read("v2"))
        thread.start()
        code = subprocess.run(command).returncode
    finally:
        stop.set()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("The owned GPU memory sampler did not join")
        samples.append(memory.read("v2"))
        memory.close()
        atomic_json(output / "GPU_MEMORY.json", {"samples": samples, "minimum_free_bytes": min(row["free"] for row in samples),
            "errors": failures, "NVML_closed": memory.closed, "metadata": memory.metadata,
            "actual_sample_interval_seconds": .1, "minimum_free_required_bytes": 2 * 1024**3})
    if failures or min(row["free"] for row in samples) < 2 * 1024**3:
        raise RuntimeError("The observed original GPU free-memory budget failed")
    if (code != 0) != expect_error:
        raise RuntimeError("The owned property diagnostic returned an unexpected status")
    return code


def identities():
    paths = [LOC]
    for row in read(LOC)["rows"]:
        name = row["image_reference"].replace("\\", "/")
        paths.append(Path("C:/" + name[7:]) if name.startswith("/mnt/c/") else Path(name))
    for label in ("baseline", "combined"):
        state, manifest = read(MODELS / label / "STATE.json"), read(MODELS / label / "INPUT_MANIFEST.json")
        if state["status"] != "completed" or state["iteration"] != 12000 or manifest["input_version"] != "new_map_admission_v1" or manifest["image_count"] != 1016:
            raise ValueError("The original actual complete model contract changed")
        paths.extend(MODELS / label / name for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"))
    return [data_identity(path) for path in paths]


def copy_models(output):
    for label in ("baseline", "combined"):
        for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
            source, target = MODELS / label / name, output / "pair" / label / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if data_identity(source)["sha256"] != data_identity(target)["sha256"]:
                raise AssertionError("A complete native model copy differs")


def bytes_equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while block := a.read(4 * 1024**2):
            if block != b.read(len(block)):
                return False
        return not b.read(1)


def compare(control, cache):
    checks = {}
    checks["all48_full_float32_RGB_arrays_byte_exact"] = all(bytes_equal(control / "render_arrays" / f"{index:03d}.npy",
        cache / "render_arrays" / f"{index:03d}.npy") for index in range(48))
    left, right = read(control / "pair/evaluation/EVALUATION.json"), read(cache / "pair/evaluation/EVALUATION.json")
    checks["all_original24x2_full_target_metrics_metadata_exact"] = all(left["results"][label]["rows"] == right["results"][label]["rows"]
        and left["results"][label]["summary"] == right["results"][label]["summary"] for label in ("baseline", "combined"))
    checks["original_complete_JPEG_all_bytes_exact"] = bytes_equal(control / "pair/evaluation/SIX_VIEW_COMPARISON.jpg",
        cache / "pair/evaluation/SIX_VIEW_COMPARISON.jpg")
    checks["both_original_development_quality_guards"] = left["development_metric_guard_passed"] and right["development_metric_guard_passed"]
    lresult, rresult = read(control / "RESULT.json"), read(cache / "RESULT.json")
    checks["both_complete_property_evaluation_protections"] = all(lresult["checks"].values()) and all(rresult["checks"].values())
    lelements, relements = lresult["workspace"]["native_CUDA_elements"], rresult["workspace"]["native_CUDA_elements"]
    reductions = {name: 1. - relements[name] / lelements[name] for name in lelements}
    checks["each_actual_CUDA_property_work_reduced20percent"] = all(value >= .2 for value in reductions.values())
    if not all(checks.values()):
        raise AssertionError("A complete paired property validation failed: " + repr(checks))
    return checks, reductions


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("generic", "pair", "fault"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise ValueError("An owned direct child output name is required")
    output = OUT / args.name
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    source_paths = [Path(__file__), CODE / "inference_property_workspace.py", CODE / "evaluate_inference_properties.py",
        CODE / "check_inference_properties.py", CODE / "evaluate_view_major.py", CODE / "evaluate_interior_ssim.py",
        CODE / "view_major_evaluation.py", CODE / "evaluate_device_artifacts.py", CODE / "evaluate_dag_models.py",
        CODE / "device_artifact_picture.py", CODE / "run_bounded_journal_task.py", CODE / "mapped_image_linux_guard.py",
        CODE / "nvml_memory_monitor_002.py",
        V1 / "evaluate_pair.py", V1 / "runtime.py"]
    sources = [data_identity(path) for path in source_paths]
    copies = output / "source_copies"
    copies.mkdir()
    for index, path in enumerate(source_paths):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    frozen = identities() if args.task != "generic" else []
    atomic_json(output / "CONTRACT.json", {"task": args.task, "sources": sources, "inputs": frozen,
        "initial_setup_seconds": time.perf_counter() - started, "work_reduction_minimum": .2, "whole_median_ratio_maximum": 1.01,
        "maximum_pair_ratio": 1.1, "current24x2_complete_denominator": True, "new_optimization_count": 0})
    if args.task == "generic":
        run(output, "check_inference_properties.py", ["--output", linux(output / "linux")])
        result = read(output / "linux/RESULT.json")
        if not all(result["checks"].values()):
            raise AssertionError("An original native getter generic check failed")
        atomic_json(output / "RESULT.json", {"status": "generic_complete", "checks": result["checks"], "sources": sources,
            "whole_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        return
    if args.task == "fault":
        side = output / "cache"
        side.mkdir()
        copy_models(side)
        run(side, "evaluate_inference_properties.py", ["--mode", "cache", "--pair", linux(side / "pair"),
            "--localization", linux(LOC), "--fault-render", "17"], expect_error=True)
        failed = read(side / "FAILURE.json")
        checks = {"actual_registered_render_error_propagated": "REGISTERED_R139_RENDER_FAILURE" in failed["error"],
            "actual_original16_render_prefix_preserved": len(failed["render_rows"]) == 16,
            "all_private_module_bindings_restored": failed["bindings_restored"],
            "all_workspace_entries_drained_after_error": failed["workspace"]["closed"] and failed["workspace"]["resident_bytes_after_close"] == 0}
        if not all(checks.values()):
            raise AssertionError("A complete native failure did not preserve its original lifecycle")
        atomic_json(output / "RESULT.json", {"status": "registered_complete_error_closed", "checks": checks,
            "sources": sources, "inputs": frozen, "whole_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        return
    rows = {}
    for mode in (("cache", "control") if args.reverse else ("control", "cache")):
        before = time.perf_counter()
        side = output / mode
        side.mkdir()
        copy_models(side)
        run(side, "evaluate_inference_properties.py", ["--mode", mode, "--pair", linux(side / "pair"), "--localization", linux(LOC)])
        result = read(side / "RESULT.json")
        if result["status"] != "complete_evaluation_protected" or not all(result["checks"].values()):
            raise AssertionError("The actual complete property evaluator failed")
        rows[mode] = {"complete_whole_seconds": time.perf_counter() - before, "result": str(side / "RESULT.json"),
            "CUDA_peak_reserved_bytes": result["actual_CUDA_reserved_peak_bytes"]}
        atomic_json(output / "PROGRESS.json", {"completed_modes": rows, "status": "running", "new_optimization_count": 0})
    compared, reductions = compare(output / "control", output / "cache")
    if sources != [data_identity(path) for path in source_paths] or frozen != identities():
        raise AssertionError("An original complete input or executed source changed")
    ratio = rows["cache"]["complete_whole_seconds"] / rows["control"]["complete_whole_seconds"]
    result = {"status": "complete_pair_protected" if ratio <= 1.1 else "failed_whole_screen", "checks": compared,
        "actual_CUDA_property_element_work_reductions": reductions, "sides": rows, "raw_whole_ratio": ratio,
        "sources": sources, "inputs": frozen, "whole_seconds": time.perf_counter() - started,
        "parent_nonoverlap_seconds": time.perf_counter() - started - sum(row["complete_whole_seconds"] for row in rows.values()),
        "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "work": reductions, "whole_ratio": ratio}), flush=True)
    if ratio > 1.1:
        raise RuntimeError("The preregistered complete whole screen exceeded its maximum ratio")


if __name__ == "__main__":
    main()
