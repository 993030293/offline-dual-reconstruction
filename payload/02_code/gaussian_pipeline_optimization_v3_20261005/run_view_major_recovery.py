"""Run owned failures and complete fresh evaluator recovery under one resource lease."""

import time

STARTED = time.perf_counter()

import json
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001"
MODELS = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0"
LOCALIZATION = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def linux(path):
    return "/mnt/c/" + Path(path).as_posix()[3:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", choices=("r0", "r1"), default="r0")
    args = parser.parse_args()
    output = OUT / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    sources = [Path(__file__), CODE / "fault_view_major_evaluation.py", CODE / "evaluate_view_major.py",
               CODE / "view_major_evaluation.py", CODE / "evaluate_interior_ssim.py"]
    source_ids = [data_identity(path) for path in sources]
    input_paths = [MODELS / label / name for label in ("baseline", "combined")
                   for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply")]
    input_ids = [data_identity(path) for path in input_paths] + [data_identity(LOCALIZATION)]
    copies = output / "source_copies"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": input_ids,
        "before_results_cases": [["native", "error"], ["view", "error"], ["view", "cancel"]],
        "fifth_SSIM_call_failure_four_real_completed_calls": True, "cancel_real_SIGINT_to_owned_child_only": True,
        "fresh_native_and_view_complete24_recovery": True})
    runs = []

    def prepare(folder):
        pair = folder / "pair"
        for label in ("baseline", "combined"):
            for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
                origin, destination = MODELS / label / name, pair / label / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
                if data_identity(origin)["sha256"] != data_identity(destination)["sha256"]:
                    raise ValueError("An original full recovery model copy changed")
        return pair

    def invoke(folder, script, args, expected_code):
        command = [sys.executable, str(CODE / "run_bounded_journal_task.py"), "--receipt", str(folder / "WINDOWS_GUARD.json"),
                   "--log", str(folder / "RUN.log"), "--maximum-seconds", "360", "--maximum-rss-gib", "8", "--",
                   "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                   linux(script), *args]
        code = subprocess.run(command).returncode
        journal = read(folder / "WINDOWS_GUARD.json")
        if journal["returncode"] != expected_code or code != (0 if expected_code == 0 else 1):
            raise RuntimeError("The actual recovery child return code differs from its registered expectation")
        runs.append({"folder": str(folder), "actual_child_returncode": journal["returncode"],
                     "guard_wrapper_returncode": code, "expected_child_code": expected_code})

    for mode, fault in (("native", "error"), ("view", "error"), ("view", "cancel")):
        if args.output_name == "r1" and mode == "native" and fault == "error":
            prior = OUT / "r0/native_error"
            receipt = read(prior / "FAULT_RESULT.json")
            journal = read(prior / "WINDOWS_GUARD.json")
            if receipt["status"] != "expected_failure_cleanup_verified" or not all(receipt["checks"].values()) or journal["returncode"] != 86:
                raise AssertionError("The preserved actual successful first failure probe is not reusable")
            saved = OUT / "r0/source_copies/01_fault_view_major_evaluation.py"
            if saved.read_bytes() != (CODE / "fault_view_major_evaluation.py").read_bytes():
                raise ValueError("The owned failure probe implementation changed")
            runs.append({"folder": str(prior), "actual_child_returncode": 86, "reused_actual_preserved_first_probe": True})
            continue
        folder = output / (mode + "_" + fault)
        pair = prepare(folder)
        invoke(folder, CODE / "fault_view_major_evaluation.py",
               ["--mode", mode, "--fault", fault, "--pair", linux(pair), "--localization", linux(LOCALIZATION)], 86)
        receipt = read(folder / "FAULT_RESULT.json")
        if receipt["status"] != "expected_failure_cleanup_verified" or not all(receipt["checks"].values()):
            raise AssertionError("An original real error, cancellation or same-directory rejection failed")
    for mode in ("native", "view"):
        folder = output / (mode + "_recovered")
        pair = prepare(folder)
        invoke(folder, CODE / "evaluate_view_major.py", ["--mode", mode, "--pair", linux(pair), "--localization", linux(LOCALIZATION)], 0)
        if not all(read(folder / "measurement/RESULT.json")["checks"].values()):
            raise AssertionError("The fresh complete recovery failed an original input/model/RNG protection")
    evaluations = {mode: read(output / (mode + "_recovered/pair/evaluation/EVALUATION.json")) for mode in ("native", "view")}
    checks = {"all_three_real_owned_failures_closed_and_original_retry_refused": len(runs) == 5,
              "both_complete24_development_quality_denominators": all(value["development_metric_guard_passed"] and value["all_frames_included_in_denominator"] == 24 for value in evaluations.values()),
              "all48_original_recovered_model_view_metric_rows_exact": all(evaluations["native"]["results"][label]["rows"] == evaluations["view"]["results"][label]["rows"] for label in ("baseline", "combined")),
              "all_recovered_rows_exact_current_complete_pair": all(evaluations[mode]["results"][label]["rows"] == read(OUT / f"c2/p0/{mode}/pair/evaluation/EVALUATION.json")["results"][label]["rows"] for mode in ("native", "view") for label in ("baseline", "combined")),
              "recovered_full_JPEG_bytes_exact": (output / "native_recovered/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes() == (output / "view_recovered/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes(),
              "all_original_inputs_and_sources_unchanged": input_ids == [data_identity(path) for path in input_paths] + [data_identity(LOCALIZATION)]
                  and source_ids == [data_identity(path) for path in sources]}
    result = {"status": "recovery_passed" if all(checks.values()) else "failed_recovery", "checks": checks,
              "runs": runs, "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0,
              "evaluation_recovery_only_no_training_checkpoint_claim": True}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "complete_child_seconds": result["complete_child_seconds"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original full fresh evaluator recovery protection failed")


if __name__ == "__main__":
    main()
