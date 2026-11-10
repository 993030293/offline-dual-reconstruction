"""Run the registered resident-input and fused-Adam combination with frozen sources."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output", "build", "cache-validation", "adam-validation", "smoke",
                 "reference-profile", "candidate-profile"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--seed-zero-evaluation", type=Path)
    args = parser.parse_args()
    source_root = args.training_source_dir or Path(__file__).parent
    numeric = json.loads(args.cache_validation.read_text())
    adam = json.loads((args.adam_validation/"RESULT.json").read_text())
    adam_contract = json.loads((args.adam_validation/"CONTRACT.json").read_text())
    if numeric["status"] != "passed" or not all(numeric["checks"].values()):
        raise ValueError("Resident input numeric and actual camera validation failed")
    if (adam["status"] != "numerical_passed_training_pending" or not all(
            row["parameter_passed"] and row["exp_avg_passed"] and row["exp_avg_sq_passed"] for row in adam["rows"])):
        raise ValueError("Fused Adam growth, pruning and opacity-state checks failed")
    cache_source = next(row for row in numeric["sources"] if Path(row["path"]).name == "resident_image_cache.py")
    for name, digest in (("resident_image_cache.py", cache_source["sha256"]),
                         ("optimizer_adapter.py", adam_contract["source_adapter"]["sha256"])):
        if data_identity(source_root/name)["sha256"] != digest:
            raise ValueError("A validated component implementation changed")
    profiles = [json.loads(path.read_text()) for path in (args.reference_profile, args.candidate_profile)]
    if (any(row["status"] != "diagnostic_completed" or row["optional_ssim"] is not None for row in profiles)
            or profiles[0]["fused_adam"] or profiles[0]["resident_image_cache"] is not None
            or not profiles[1]["fused_adam"] or profiles[1]["resident_image_cache"] is None
            or profiles[0]["model"]["sha256"] != profiles[1]["model"]["sha256"]
            or profiles[1]["wall_ms_p50_p95"][0]/profiles[0]["wall_ms_p50_p95"][0] > .9):
        raise ValueError("The same-model diagnostic does not support the registered combined speed target")
    smoke = json.loads((args.smoke/"STATE.json").read_text())
    smoke_config = json.loads((args.smoke/"CONFIG.json").read_text())
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["image_count"] != 212
            or smoke["gaussian_count"] <= 35633
            or not smoke_config["intervention"]["resident_fused_adam"]
            or not smoke["resident_image_cache"]["original_references_restored"]
            or smoke["resident_image_cache"]["samples"] != 1100):
        raise ValueError("Actual combined growth and cache restoration smoke failed")
    for row in smoke_config["implementation"]:
        if data_identity(source_root/Path(row["path"]).name)["sha256"] != row["sha256"]:
            raise ValueError("Training sources differ from the actual combination smoke")
    if args.seed != 0:
        if args.seed_zero_evaluation is None or args.training_source_dir is None:
            raise ValueError("Later seeds require first-pair frozen source and evaluation")
        evaluation = json.loads(args.seed_zero_evaluation.read_text())
        first_pair = json.loads((args.seed_zero_evaluation.parent.parent/"PAIR_STATE.json").read_text())
        if (not evaluation["development_metric_guard_passed"] or evaluation["control_label"] != "reference"
                or evaluation["candidate_label"] != "resident_adam" or first_pair["time_ratio"] > .9):
            raise ValueError("The first pair did not pass quality and preliminary speed admission")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py", "optimizer_adapter.py", "resident_image_cache.py"]
    for name in names:
        shutil.copyfile(source_root/name, snapshot/name)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "candidate": "P19_RESIDENT_FUSED_ADAM", "iterations": 12000, "dataset": str(args.dataset),
        "gpu_concurrency": 1, "maximum_seconds_each": 1800, "projection": "full",
        "ssim": "reference", "cache_validation": data_identity(args.cache_validation),
        "adam_validation": data_identity(args.adam_validation/"RESULT.json"),
        "smoke": data_identity(args.smoke/"STATE.json"), "build": data_identity(args.build/"RESULT.json"),
        "profile": [data_identity(path) for path in (args.reference_profile, args.candidate_profile)],
        "source": data_identity(Path(__file__)), "sources": [data_identity(snapshot/name) for name in names],
        "counted": False, "maximum_optimization_count_if_accepted": 1}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    for label in (["resident_adam", "reference"] if args.seed == 1 else ["reference", "resident_adam"]):
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full",
            "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build)]
        if label == "resident_adam":
            command.append("--resident-fused-adam")
        row = {"label": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        with (args.output/(label+".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(pid=process.pid, status="running")
            atomic_json(args.output/"PAIR_STATE.json", state)
            try:
                code = process.wait(timeout=1900)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                row["status"] = "watchdog_timeout"
                state["status"] = "failed"
                atomic_json(args.output/"PAIR_STATE.json", state)
                raise
        row.update(returncode=code, status="completed" if code == 0 else "failed")
        atomic_json(args.output/"PAIR_STATE.json", state)
        if code:
            state["status"] = "failed"
            atomic_json(args.output/"PAIR_STATE.json", state)
            raise RuntimeError("Resident Adam training failed")
    control, candidate = [json.loads((args.output/label/"STATE.json").read_text()) for label in ("reference", "resident_adam")]
    inputs = [json.loads((args.output/label/"INPUT_MANIFEST.json").read_text())["files"] for label in ("reference", "resident_adam")]
    if ([row["sha256"] for row in inputs[0]] != [row["sha256"] for row in inputs[1]]
            or control["sample_sequence_sha256"] != candidate["sample_sequence_sha256"]):
        state["status"] = "failed_input_or_sampling_sequence"
        atomic_json(args.output/"PAIR_STATE.json", state)
        raise ValueError("Frozen input or camera sample sequence changed")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started,
        reference_seconds=control["training_call_seconds"], candidate_seconds=candidate["training_call_seconds"],
        identical_input_and_sampling=True, time_ratio=candidate["training_call_seconds"]/control["training_call_seconds"])
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "reference_seconds", "candidate_seconds", "time_ratio")}))


if __name__ == "__main__":
    main()
