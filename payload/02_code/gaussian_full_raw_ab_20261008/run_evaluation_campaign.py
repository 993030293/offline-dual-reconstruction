"""Evaluate each fresh raw side once its independently generated inputs are ready."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity
from atomic_json_windows import atomic_json, read_json
from run_side import windows_to_wsl


def read(path):
    return read_json(Path(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=21600)
    args = parser.parse_args()
    state_path = args.campaign / "EVALUATION_CAMPAIGN_STATE.json"
    if state_path.exists():
        old = read(state_path)
        try:
            if psutil.Process(old["pid"]).create_time() == old["process_birth"]:
                raise RuntimeError("An evaluation controller with this identity is already alive")
        except psutil.NoSuchProcess:
            pass
    state = {"status": "running", "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
        "source": data_identity(Path(__file__)), "results": {}, "errors": {},
        "failure_rows_not_removed": True, "primary_region": "whole image",
        "map_coupled_localization_is_not_independent_geometry_ground_truth": True}
    started = time.monotonic()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cases = read(args.campaign / "CASES.json")["cases"]
    sides = [(row["case"], mode) for row in cases for mode in ("baseline", "optimized")]

    def publish():
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(state_path, state)

    def execute(side, label, command, result, budget, rss=12):
        receipt, log = side / (label + "_process.json"), side / (label + ".log")
        if receipt.exists():
            measured = read(receipt)
            if measured["status"] != "completed" or measured.get("returncode") != 0 or not result.exists():
                raise ValueError("An existing failed or live evaluation stage cannot be overwritten: " + label)
            return read(result)
        if result.exists():
            raise ValueError("Evaluation result has no matching completion receipt")
        guarded = [sys.executable, str(CODE / "run_owned_task.py"), "--receipt", str(receipt), "--log", str(log),
                   "--maximum-seconds", str(budget), "--maximum-rss-gib", str(rss), "--", *map(str, command)]
        completed = subprocess.run(guarded, env=env)
        if completed.returncode or not result.exists():
            raise RuntimeError("The native evaluation stage failed: " + label)
        return read(result)

    try:
        while True:
            work = False
            for case, mode in sides:
                key = case + "/" + mode
                if key in state["errors"] or state["results"].get(key, {}).get("render"):
                    continue
                side = args.campaign / case / mode
                admission = side / "export/TRAINING_ADMISSION.json"
                export_receipt = side / "export_process.json"
                if not admission.exists() or not export_receipt.exists() or read(export_receipt)["status"] != "completed":
                    continue
                try:
                    state["current_side"], state["current_stage"] = key, "localization"
                    publish()
                    location = execute(side, "localization", [sys.executable, CODE / "localize_evaluation.py", "--side", side,
                        "--mode", mode, "--output", side / "localization"], side / "localization/LOCALIZATION.json", 1800)
                    if len(location["rows"]) != 24:
                        raise ValueError("The complete frozen raw evaluation denominator changed")
                    state["results"].setdefault(key, {})["localization"] = {"status": location["status"], **location["summary"]}
                    publish()
                    training = side / "training/STATE.json"
                    train_receipt = side / "training_process.json"
                    if not training.exists() or not train_receipt.exists() or read(train_receipt)["status"] != "completed":
                        continue
                    trained = read(training)
                    if trained["status"] != "completed" or trained["iteration"] != trained["target_iterations"]:
                        raise ValueError("A incomplete training model is not an evaluation input")
                    state["current_stage"] = "render_waiting_for_GPU_lease"
                    publish()
                    command = ["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--", "env", "PYTHONUTF8=1",
                        "PYTHONDONTWRITEBYTECODE=1", "/opt/e26_thin_surface_v1/venv/bin/python",
                        windows_to_wsl(CODE / "render_evaluation.py"), "--side", windows_to_wsl(side), "--runtime",
                        windows_to_wsl(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"), "--output", windows_to_wsl(side / "evaluation")]
                    command = [sys.executable, CODE / "gpu_lease.py", "--receipt", side / "evaluation_GPU_LEASE.json",
                               "--label", key + "_evaluation", "--", *command]
                    rendered = execute(side, "evaluation", command, side / "evaluation/RESULT.json", 3300, rss=8)
                    if len(rendered["rows"]) != 24:
                        raise ValueError("Rendering omitted a raw evaluation row")
                    state["results"][key]["render"] = {"status": rendered["status"], "summary": rendered["summary"],
                        "gaussian_count": rendered["gaussian_count"], "elapsed_seconds": rendered["elapsed_seconds"]}
                    work = True
                    publish()
                except BaseException as error:
                    state["errors"][key] = {"error": repr(error), "failed_stage": state["current_stage"], "outputs_preserved": True}
                    publish()
            campaign = read(args.campaign / "CAMPAIGN_STATE.json")
            unfinished = []
            for case, mode in sides:
                key = case + "/" + mode
                if key not in state["errors"] and not state["results"].get(key, {}).get("render"):
                    unfinished.append(key)
            if campaign["status"] != "running":
                owned_alive = False
                for observer_path in args.campaign.glob("*/*_projection_observer_*/STATE.json"):
                    observer = read(observer_path)
                    if observer["status"] in ("waiting_for_existing_side_outcome", "running_declared_renderer_interface_recovery"):
                        try:
                            if psutil.Process(observer["pid"]).create_time() == observer["process_birth"]:
                                owned_alive = True
                        except psutil.NoSuchProcess:
                            pass
                for key in unfinished:
                    side_state = args.campaign / key / "PIPELINE_STATE.json"
                    if not side_state.exists():
                        continue
                    item = read(side_state)
                    try:
                        if item["status"] == "running" and psutil.Process(item["pid"]).create_time() == item["process_birth"]:
                            owned_alive = True
                    except psutil.NoSuchProcess:
                        pass
                if not owned_alive:
                    for key in unfinished:
                        state["errors"][key] = {"error": "fresh_training_unavailable_after_campaign_resolution", "outputs_preserved": True}
                    state["status"] = "all_sides_evaluated_or_failures_recorded_module_and_visual_review_pending"
                    publish()
                    break
            if time.monotonic()-started > args.maximum_seconds:
                raise TimeoutError("Evaluation campaign wait budget reached; healthy pipeline jobs remain untouched")
            if not work:
                time.sleep(5)
    except BaseException as error:
        state.update(status="failed_evaluation_controller", error=repr(error))
        publish()
        raise


if __name__ == "__main__":
    main()
