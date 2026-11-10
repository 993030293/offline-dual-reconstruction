"""Run the preregistered complete current SSIM consumer under bounded guards."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/iscw001"
MODELS = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0"
LOCALIZATION = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def linux(path):
    value = Path(path).resolve().as_posix()
    if not value.startswith("C:/"):
        raise ValueError("The current consumer supports only the held C workspace")
    return "/mnt/c/" + value[3:]


def windows(path):
    value = str(path)
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(value)


def prepare(folder):
    pair = folder / "pair"
    for label in ("baseline", "combined"):
        for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
            origin, target = MODELS / label / name, pair / label / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            if data_identity(origin)["sha256"] != data_identity(target)["sha256"]:
                raise ValueError("An original complete model physical copy changed")
    return pair


def invoke(folder, mode, script="evaluate_interior_conv_work.py", extra=(), expected_code=0):
    pair = folder / "pair"
    command = [sys.executable, str(CODE / "run_bounded_journal_task.py"), "--receipt", str(folder / "WINDOWS_GUARD.json"),
               "--log", str(folder / "EVALUATE.log"), "--maximum-seconds", "360", "--maximum-rss-gib", "8", "--",
               "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
               linux(CODE / script), "--mode", mode, "--pair", linux(pair), "--localization", linux(LOCALIZATION), *extra]
    before = time.perf_counter()
    code = subprocess.run(command).returncode
    atomic_json(folder / "ACTUAL_GUARD_EXIT.json", {"returncode": code, "seconds": time.perf_counter() - before,
                                                  "expected_child_returncode": expected_code})
    guard = read(folder / "WINDOWS_GUARD.json")
    if code != (0 if expected_code == 0 else 1) or guard["returncode"] != expected_code:
        raise RuntimeError("The current native evaluator or guard returned an unexpected exit code")
    return guard


def compare(native, interior):
    evaluations = {mode: read(folder / "pair/evaluation/EVALUATION.json") for mode, folder in
                   (("native", native), ("interior", interior))}
    exact, max_ssim = True, 0.0
    for label in ("baseline", "combined"):
        left = evaluations["native"]["results"][label]["rows"]
        right = evaluations["interior"]["results"][label]["rows"]
        if len(left) != 24 or len(right) != 24:
            raise ValueError("The original complete model view denominator changed")
        for first, second in zip(left, right):
            one, two = dict(first), dict(second)
            for region in ("full_image", "target"):
                expected, actual = dict(one.pop(region)), dict(two.pop(region))
                max_ssim = max(max_ssim, abs(expected.pop("SSIM") - actual.pop("SSIM")))
                exact = exact and expected == actual
            exact = exact and one == two
    contracts = {mode: read(folder / "pair/evaluation/EVALUATION_CONTRACT.json") for mode, folder in
                 (("native", native), ("interior", interior))}
    checks = {"all48_original_metadata_PSNR_MSE_exact": exact,
              "all96_original_SSIM_tolerance": max_ssim <= 1e-7,
              "all_original_K24_photo_pose_contract_exact": contracts["native"] == contracts["interior"],
              "complete_JPEG_bytes_exact": (native / "pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes()
                  == (interior / "pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes()}
    for mode, folder in (("native", native), ("interior", interior)):
        measurement, work = read(folder / "measurement/RESULT.json"), read(folder / "CONV_WORK.json")
        checks[mode + "_original_inputs_parameters_RNG_and_bindings"] = all(measurement["checks"].values())
        checks[mode + "_census_binding_and_cache_close"] = all(work["checks"].values())
        checks[mode + "_all24_original_development_guard"] = bool(evaluations[mode]["development_metric_guard_passed"])
    return {"checks": checks, "maximum_absolute_SSIM_error": max_ssim}


def identity_set():
    paths = [Path(__file__), CODE / "evaluate_interior_conv_work.py", CODE / "ssim_conv_work.py",
             CODE / "interior_ssim_reuse.py", CODE / "view_major_evaluation.py", CODE / "evaluate_interior_ssim.py",
             CODE / "evaluate_dag_models.py", CODE / "evaluate_device_artifacts.py", CODE / "device_artifact_picture.py",
             CODE / "run_bounded_journal_task.py", CODE / "mapped_image_linux_guard.py",
             ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/evaluate_pair.py",
             ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/full_intrinsics.py",
             ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/reference_ssim.py",
             ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/vectorized_gaussian_read.py"]
    localization = read(LOCALIZATION)
    if localization["input_version"] != "new_map_admission_v1" or len(localization["rows"]) != 24:
        raise ValueError("The original current 24-view map version changed")
    inputs = [LOCALIZATION, *(windows(row["image_reference"]) for row in localization["rows"])]
    for label in ("baseline", "combined"):
        state, manifest = read(MODELS / label / "STATE.json"), read(MODELS / label / "INPUT_MANIFEST.json")
        if state["status"] != "completed" or state["iteration"] != 12000 or manifest["image_count"] != 1016:
            raise ValueError("The original complete current model or training denominator changed")
        inputs.extend(MODELS / label / name for name in ("STATE.json", "INPUT_MANIFEST.json",
                      "model/point_cloud/iteration_12000/point_cloud.ply"))
    return paths, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("first", "rest", "all"), required=True)
    parser.add_argument("--campaign", choices=("c0", "c1"), default="c0")
    args = parser.parse_args()
    output = OWN / args.campaign
    if args.part in ("first", "all"):
        output.mkdir(parents=True, exist_ok=False)
    elif not read(output / "STATE.json")["first_pair_passed"]:
        raise ValueError("The first complete current pair did not pass")
    process = psutil.Process()
    atomic_json(output / (args.part + "_PROCESS.json"), {"pid": process.pid, "creation_time": process.create_time()})
    generic = read(OWN / "g0/RESULT.json")
    if generic["status"] != "generic_bridge_and_census_passed" or not all(generic["checks"].values()):
        raise ValueError("The actual view-finisher and native census generic failed")
    for row in generic["sources"]:
        if data_identity(windows(row["path"]))["sha256"] != row["sha256"]:
            raise ValueError("A frozen generic-tested implementation changed")
    paths, inputs = identity_set()
    sources, identities = [data_identity(p) for p in paths], [data_identity(p) for p in inputs]
    if args.part in ("first", "all"):
        copies = output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": identities,
                    "common_Q30_Q61_R100_R135": True, "original_1016_24_48_96": True,
                    "native_conv_output_work_minimum_net_reduction": .1, "whole_median_ratio_limit": 1.01,
                    "each_whole_ratio_limit": 1.1, "new_optimization_count": 0})
    else:
        contract = read(output / "CONTRACT.json")
        if sources != contract["sources"] or identities != contract["inputs"]:
            raise ValueError("The formal current sources or complete input changed between pairs")
    prior = read(output / "STATE.json") if args.part == "rest" else {"rows": [], "comparisons": []}
    rows, comparisons = prior["rows"], prior["comparisons"]
    try:
        for trial in ((0,) if args.part == "first" else ((1, 2) if args.part == "rest" else (0, 1, 2))):
            local = {}
            for mode in (("native", "interior") if trial % 2 == 0 else ("interior", "native")):
                before = time.perf_counter()
                if shutil.disk_usage(OWN).free < 20 * 1024**3 or time.perf_counter() - STARTED > 900:
                    raise RuntimeError("The original disk or campaign time reserve is unavailable")
                if sources != [data_identity(p) for p in paths] or identities != [data_identity(p) for p in inputs]:
                    raise ValueError("An original current input or source changed before consumption")
                folder = output / f"p{trial}" / mode
                prepare(folder)
                guard = invoke(folder, mode)
                measurement, work = read(folder / "measurement/RESULT.json"), read(folder / "CONV_WORK.json")
                resource = read(folder / "measurement/WSL_RESOURCE.json")
                if guard["status"] != "completed" or resource["status"] != "completed" or not all(measurement["checks"].values()):
                    raise AssertionError("An original complete consumer resource or scientific protection failed")
                if sources != [data_identity(p) for p in paths] or identities != [data_identity(p) for p in inputs]:
                    raise ValueError("The actual full evaluation mutated an original input or source")
                row = {"pair": trial, "mode": mode, "complete_whole_seconds": time.perf_counter() - before,
                       "CUDA_conv_output_elements": work["trace"]["successful_CUDA_output_elements"],
                       "CUDA_conv_output_bytes": work["trace"]["successful_CUDA_output_bytes"],
                       "peak_CUDA_reserved_bytes": measurement["peak_CUDA_reserved_bytes"],
                       "folder": str(folder)}
                rows.append(row)
                local[mode] = row
                atomic_json(output / "STATE.json", {"status": "running", "rows": rows, "comparisons": comparisons,
                                                    "first_pair_passed": args.part == "rest"})
            before = time.perf_counter()
            comparison = compare(output / f"p{trial}/native", output / f"p{trial}/interior")
            comparison.update(pair=trial, comparison_seconds=time.perf_counter() - before,
                raw_work_reduction=1 - local["interior"]["CUDA_conv_output_elements"] / local["native"]["CUDA_conv_output_elements"],
                whole_ratio=local["interior"]["complete_whole_seconds"] / local["native"]["complete_whole_seconds"])
            comparison["checks"]["each_whole_ratio_at_most_1_1"] = comparison["whole_ratio"] <= 1.1
            comparison["checks"]["raw_native_work_reduction_at_least10percent"] = comparison["raw_work_reduction"] >= .1
            comparisons.append(comparison)
            atomic_json(output / "STATE.json", {"status": "first_pair_passed" if args.part == "first" else "running",
                        "rows": rows, "comparisons": comparisons, "first_pair_passed": all(comparison["checks"].values())})
            print(json.dumps(comparison), flush=True)
            if not all(comparison["checks"].values()):
                raise AssertionError("An original complete science, native work or whole protection failed")
        if args.part in ("rest", "all"):
            medians = {mode: {key: statistics.median(r[key] for r in rows if r["mode"] == mode)
                       for key in ("complete_whole_seconds", "CUDA_conv_output_elements")} for mode in ("native", "interior")}
            if medians["interior"]["complete_whole_seconds"] / medians["native"]["complete_whole_seconds"] > 1.01:
                raise AssertionError("The preregistered original whole median guard failed")
            atomic_json(output / "RESULT.json", {"status": "three_fresh_current_pairs_pending_cost_and_recovery",
                        "rows": rows, "comparisons": comparisons, "raw_medians": medians, "new_optimization_count": 0})
        atomic_json(output / (args.part + "_COMPLETE.json"), {"status": "completed", "seconds": time.perf_counter() - STARTED,
                                                              "new_optimization_count": 0})
    except BaseException as error:
        atomic_json(output / "RESULT.json", {"status": "failed_original_complete_protection", "error": repr(error),
                    "rows": rows, "comparisons": comparisons, "actual_part": args.part,
                    "seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()

