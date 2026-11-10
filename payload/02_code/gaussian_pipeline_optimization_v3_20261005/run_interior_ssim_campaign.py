"""Measure fresh complete evaluators with current models and full input protection."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/is001"
MODELS = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0"
LOCALIZATION = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows(path):
    text = str(path)
    if text.startswith("/mnt/c/"):
        return Path("C:/" + text[7:])
    return Path(path)


def linux(path):
    return "/mnt/c/" + Path(path).as_posix()[3:]


def main():
    output = OUT / "c0"
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    generic, extra = read(OUT / "g0/RESULT.json"), read(OUT / "g1/RESULT.json")
    if generic["status"] != "generic_passed" or extra["status"] != "generic_extra_passed" or not all(generic["checks"].values()) or not all(extra["checks"].values()):
        raise ValueError("The complete original generic SSIM protection did not pass")
    paths = [Path(__file__), CODE / "evaluate_interior_ssim.py", CODE / "interior_ssim_reuse.py",
             CODE / "evaluate_dag_models.py", CODE / "evaluate_device_artifacts.py", CODE / "device_artifact_picture.py",
             V1 / "evaluate_pair.py", V1 / "full_intrinsics.py", V1 / "runtime.py", V2 / "reference_ssim.py",
             V2 / "vectorized_gaussian_read.py", CODE / "mapped_image_linux_guard.py"]
    sources = [data_identity(path) for path in paths]
    frozen_candidate = read(OUT / "g0/CONTRACT.json")["sources"][1]
    if sources[2]["sha256"] != frozen_candidate["sha256"] or sources[2]["bytes"] != frozen_candidate["bytes"]:
        raise ValueError("The candidate changed after its first protected generic test")
    localization = read(LOCALIZATION)
    if localization["input_version"] != "new_map_admission_v1" or len(localization["rows"]) != 24:
        raise ValueError("The current full localization denominator or map version changed")
    inputs = [LOCALIZATION] + [windows(row["image_reference"]) for row in localization["rows"]]
    for label in ("baseline", "combined"):
        state, manifest = read(MODELS / label / "STATE.json"), read(MODELS / label / "INPUT_MANIFEST.json")
        if state["status"] != "completed" or state["iteration"] != 12000 or manifest["input_version"] != "new_map_admission_v1" or manifest["image_count"] != 1016:
            raise ValueError("The current actual complete model identity or training denominator changed")
        inputs.extend(MODELS / label / name for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"))
    identities = [data_identity(path) for path in inputs]
    copies = output / "source_copies"
    copies.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": identities,
        "current_input_version": "new_map_admission_v1", "training_images": 1016, "dev24": 24,
        "iterations_both": 12000, "orders": [["native", "interior"], ["interior", "native"], ["native", "interior"]],
        "V2_complete_phase_reduction_minimum": .05, "whole_median_cannot_increase": True,
        "maximum_each_whole_ratio": 1.1, "absolute_SSIM_tolerance": 1e-7,
        "first_pair_stop_on_no_raw_phase_benefit_or_whole_ratio_violation": True,
        "generic_failed_extra_review_first_fees_and_campaign_extra_charged_six_sides": True,
        "common_current_Q30_Q61_R100": True, "strict_blind_test": False, "new_optimization_count": 0})
    rows, comparisons = [], []
    try:
        for trial in range(3):
            trial_rows = {}
            for mode in (("native", "interior") if trial % 2 == 0 else ("interior", "native")):
                before = time.perf_counter()
                folder = output / f"p{trial}" / mode
                pair = folder / "pair"
                if time.perf_counter() - STARTED > 1600 or shutil.disk_usage(output).free < 20 * 1024**3:
                    raise RuntimeError("The original full campaign time or disk reserve is unavailable")
                if sources != [data_identity(path) for path in paths] or identities != [data_identity(path) for path in inputs]:
                    raise ValueError("The frozen current complete evaluation input changed")
                for label in ("baseline", "combined"):
                    for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
                        origin, destination = MODELS / label / name, pair / label / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(origin, destination)
                        if data_identity(origin)["sha256"] != data_identity(destination)["sha256"]:
                            raise ValueError("A complete physical model copy changed")
                setup_seconds = time.perf_counter() - before
                command = [sys.executable, str(CODE / "run_bounded_journal_task.py"), "--receipt", str(folder / "WINDOWS_GUARD.json"),
                    "--log", str(folder / "EVALUATE.log"), "--maximum-seconds", "360", "--maximum-rss-gib", "8", "--",
                    "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                    linux(CODE / "evaluate_interior_ssim.py"), "--mode", mode, "--pair", linux(pair), "--localization", linux(LOCALIZATION)]
                subprocess.run(command, check=True)
                post = time.perf_counter()
                measurement = read(folder / "measurement/RESULT.json")
                guard = read(folder / "WINDOWS_GUARD.json")
                resource = read(folder / "measurement/WSL_RESOURCE.json")
                evaluation = read(pair / "evaluation/EVALUATION.json")
                if not all(measurement["checks"].values()) or guard["status"] != "completed" or resource["status"] != "completed":
                    raise ValueError("A full evaluator protection or real resource guard failed")
                if not evaluation["development_metric_guard_passed"] or evaluation["all_frames_included_in_denominator"] != 24:
                    raise ValueError("The original full development quality or denominator guard failed")
                if identities != [data_identity(path) for path in inputs] or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The complete evaluation changed original input or source identities")
                post_seconds = time.perf_counter() - post
                phase = setup_seconds + measurement["composition_source_copy_identity_seconds"] + measurement["complete_SSIM_seconds_including_input_protection"]
                phase += measurement["complete_model_state_validation_seconds"] + post_seconds
                row = {"pair": trial, "mode": mode, "complete_phase_seconds": phase,
                       "complete_whole_seconds": time.perf_counter() - before, "setup_seconds": setup_seconds,
                       "post_validation_seconds": post_seconds, "measurement": str(folder / "measurement/RESULT.json"),
                       "all96_SSIM_inputs_model_state_RNG_source_and_bindings_protected": True}
                rows.append(row)
                trial_rows[mode] = row
                atomic_json(output / "STATE.json", {"status": "running", "rows": rows, "comparisons": comparisons})
            before_compare = time.perf_counter()
            native_folder, candidate_folder = output / f"p{trial}/native/pair/evaluation", output / f"p{trial}/interior/pair/evaluation"
            evaluations = {mode: read(output / f"p{trial}/{mode}/pair/evaluation/EVALUATION.json") for mode in ("native", "interior")}
            maximum_ssim = 0.0
            exact_other = True
            for label in ("baseline", "combined"):
                left, right = evaluations["native"]["results"][label]["rows"], evaluations["interior"]["results"][label]["rows"]
                if len(left) != len(right) or len(left) != 24:
                    raise ValueError("The original complete two-model metric rows changed")
                for first, second in zip(left, right):
                    one, two = dict(first), dict(second)
                    for region in ("full_image", "target"):
                        expected, actual = dict(one.pop(region)), dict(two.pop(region))
                        maximum_ssim = max(maximum_ssim, abs(expected.pop("SSIM") - actual.pop("SSIM")))
                        exact_other = exact_other and expected == actual
                    exact_other = exact_other and one == two
            contracts = {mode: read(output / f"p{trial}/{mode}/pair/evaluation/EVALUATION_CONTRACT.json") for mode in ("native", "interior")}
            checks = {"all48_original_rows_metadata_PSNR_MSE_exact": exact_other,
                      "all96_SSIM_original_absolute_scalar_gate": maximum_ssim <= 1e-7,
                      "all_original_K24_photo_and_pose_contract_exact": contracts["native"] == contracts["interior"],
                      "six_view_complete_JPEG_bytes_exact": (native_folder / "SIX_VIEW_COMPARISON.jpg").read_bytes() == (candidate_folder / "SIX_VIEW_COMPARISON.jpg").read_bytes(),
                      "each_complete_whole_ratio_at_most_1_1": trial_rows["interior"]["complete_whole_seconds"] <= 1.1 * trial_rows["native"]["complete_whole_seconds"]}
            comparisons.append({"pair": trial, "checks": checks, "maximum_absolute_SSIM_error": maximum_ssim,
                                "independent_compare_seconds": time.perf_counter() - before_compare})
            atomic_json(output / "STATE.json", {"status": "running", "rows": rows, "comparisons": comparisons})
            print(json.dumps({"pair": trial, "checks": checks, "maximum_SSIM_error": maximum_ssim,
                              "complete_phase_seconds": {mode: row["complete_phase_seconds"] for mode, row in trial_rows.items()},
                              "complete_whole_seconds": {mode: row["complete_whole_seconds"] for mode, row in trial_rows.items()}}), flush=True)
            if not all(checks.values()):
                raise ValueError("An original actual complete evaluation numeric, contract, artifact or whole protection failed")
            if trial == 0 and trial_rows["interior"]["complete_phase_seconds"] >= trial_rows["native"]["complete_phase_seconds"]:
                raise RuntimeError("The preregistered first complete pair has no raw phase benefit")
        medians = {mode: {key: statistics.median(row[key] for row in rows if row["mode"] == mode)
                   for key in ("complete_phase_seconds", "complete_whole_seconds")} for mode in ("native", "interior")}
        result = {"status": "three_complete_pairs_pending_fees_and_review", "rows": rows, "comparisons": comparisons,
                  "raw_medians": medians, "complete_campaign_seconds": time.perf_counter() - STARTED,
                  "new_optimization_count": 0}
        atomic_json(output / "RESULT.json", result)
    except BaseException as error:
        result = {"status": "closed_negative_first_protection_or_raw_cost", "error": repr(error), "rows": rows,
                  "comparisons": comparisons, "complete_campaign_seconds": time.perf_counter() - STARTED,
                  "formal_three_pairs_external_recovery_final_cost": "not_run_after_preregistered_failure", "new_optimization_count": 0}
        atomic_json(output / "RESULT.json", result)
        raise


if __name__ == "__main__":
    main()
