"""Run the preregistered single-GPU full versus progressive resolution pair."""

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
    for name in ("runtime", "dataset", "output", "build", "validation", "smoke"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--seed-zero-evaluation", type=Path)
    args = parser.parse_args()
    source_root = args.training_source_dir or Path(__file__).parent
    validation = json.loads(args.validation.read_text())
    smoke = json.loads((args.smoke/"STATE.json").read_text())
    smoke_config = json.loads((args.smoke/"CONFIG.json").read_text())
    if validation["status"] != "passed" or not all(validation["checks"].values()):
        raise ValueError("The real-camera and gradient prerequisite failed")
    if (smoke["status"] != "completed" or smoke["iteration"] != 1200 or smoke["gaussian_count"] <= 35633
            or smoke["image_count"] != 212 or smoke["resolution_schedule"]["samples"] != {"coarse": 600, "full": 600}
            or not smoke["resolution_schedule"]["original_tensor_references_and_calibration_restored"]):
        raise ValueError("The actual growth and resolution-switch smoke failed")
    for name in ("resolution_adapter.py", "gsplat_adapter.py", "full_intrinsics.py"):
        if data_identity(source_root/name)["sha256"] != validation["source_sha256"][name]:
            raise ValueError("The source differs from the validated adapter")
    smoke_hashes = {Path(row["path"]).name: row["sha256"] for row in smoke_config["implementation"]}
    for name, digest in smoke_hashes.items():
        if data_identity(source_root/name)["sha256"] != digest:
            raise ValueError("The training source differs from the actual smoke")
    if args.seed != 0:
        if args.seed_zero_evaluation is None or args.training_source_dir is None:
            raise ValueError("Subsequent seeds require the first pair's frozen sources and evaluation")
        evaluation = json.loads(args.seed_zero_evaluation.read_text())
        if (not evaluation["development_metric_guard_passed"] or evaluation["control_label"] != "full"
                or evaluation["candidate_label"] != "progressive" or evaluation["iterations"] != 12000):
            raise ValueError("The first pair failed its fixed quality gates")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py", "resolution_adapter.py"]
    for name in names:
        shutil.copyfile(source_root/name, snapshot/name)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "candidate": "P12_PROGRESSIVE_RESOLUTION", "dataset": str(args.dataset), "iterations": 12000,
        "control": "full812x620_all12000", "candidate_schedule": "406x310_first5000_then_original812x620_7000",
        "radius_unit": "original_full_resolution_pixels", "gpu_concurrency": 1, "maximum_seconds_each": 1800,
        "validation": data_identity(args.validation), "smoke": data_identity(args.smoke/"STATE.json"),
        "build": data_identity(args.build/"RESULT.json"), "source": data_identity(Path(__file__)),
        "training_sources": [data_identity(snapshot/name) for name in names], "counted": False,
        "acceptance": "3 seeds median training call >=10% faster; all full/target PSNR>=-0.1dB and SSIM>=-0.002; protected views"}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    for label in (["progressive", "full"] if args.seed == 1 else ["full", "progressive"]):
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full",
            "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build),
            "--pyramid-until", "5000" if label == "progressive" else "0"]
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
            raise RuntimeError("Resolution training failed")
    control = json.loads((args.output/"full/STATE.json").read_text())
    candidate = json.loads((args.output/"progressive/STATE.json").read_text())
    if control["sample_sequence_sha256"] != candidate["sample_sequence_sha256"]:
        state["status"] = "failed_sample_sequence"
        atomic_json(args.output/"PAIR_STATE.json", state)
        raise ValueError("The fixed camera sampling order changed")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started,
        identical_sampling_sequence=True, reference_seconds=control["training_call_seconds"],
        candidate_seconds=candidate["training_call_seconds"])
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "reference_seconds", "candidate_seconds")}), flush=True)


if __name__ == "__main__":
    main()
