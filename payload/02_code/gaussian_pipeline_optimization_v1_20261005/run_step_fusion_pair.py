"""Compare one frozen gsplat loss/Adam fusion package under identical budgets."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output", "build", "ssim-source", "ssim-library", "ssim-validation", "adam-validation", "smoke"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--seed-zero-evaluation", type=Path)
    args = parser.parse_args()
    source_root = args.training_source_dir or Path(__file__).parent
    ssim = read(args.ssim_validation/"RESULT.json")
    ssim_contract = read(args.ssim_validation/"CONTRACT.json")
    adam = read(args.adam_validation/"RESULT.json")
    adam_contract = read(args.adam_validation/"CONTRACT.json")
    if not ssim["rows"] or not all(row["passed"] for row in ssim["rows"]):
        raise ValueError("The frozen SSIM numeric gates did not pass")
    if adam["status"] != "numerical_passed_training_pending":
        raise ValueError("The actual density/opacity optimizer validation failed")
    if data_identity(source_root/"optimizer_adapter.py")["sha256"] != adam_contract["source_adapter"]["sha256"]:
        raise ValueError("The optimizer adapter changed after numeric validation")
    extensions = list(args.ssim_library.glob("fused_ssim_cuda*.so"))
    if len(extensions) != 1 or data_identity(extensions[0])["sha256"] != ssim_contract["extension"][0]["sha256"]:
        raise ValueError("The fused SSIM extension differs from its validation")
    smoke = read(args.smoke/"STATE.json")
    smoke_config = read(args.smoke/"CONFIG.json")
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["gaussian_count"] <= 35633
            or smoke["image_count"] != 212 or smoke["resolution_schedule"] is not None
            or smoke_config["intervention"]["dense_adam_backend"] != "fused"
            or smoke_config["intervention"]["ssim_backend"] != "fused"):
        raise ValueError("The real combined growth smoke did not pass its exact scope")
    for item in smoke_config["implementation"]:
        if data_identity(source_root/Path(item["path"]).name)["sha256"] != item["sha256"]:
            raise ValueError("Training source changed after the real smoke")
    if smoke_config["optional_gsplat_identity"]["build"]["sha256"] != data_identity(args.build/"RESULT.json")["sha256"]:
        raise ValueError("The gsplat build changed after smoke")
    if data_identity(args.ssim_source/"fused_ssim/__init__.py")["sha256"] != smoke_config["optional_ssim_identity"]["python"]["sha256"]:
        raise ValueError("The SSIM Python adapter changed after smoke")
    if args.seed != 0:
        if args.training_source_dir is None or args.seed_zero_evaluation is None:
            raise ValueError("Follow-up seeds require the first pair's frozen scripts and evaluation")
        evaluation = read(args.seed_zero_evaluation)
        if (not evaluation["development_metric_guard_passed"] or evaluation["control_label"] != "reference"
                or evaluation["candidate_label"] != "fused_step" or evaluation["iterations"] != 12000):
            raise ValueError("The first fixed-budget pair did not pass quality")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py", "optimizer_adapter.py"]
    for name in names:
        shutil.copyfile(source_root/name, snapshot/name)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "candidate": "P13_GSPLAT_STAGE_FUSION", "iterations_both": 12000, "images_both": 212,
        "projection": "full", "resolution": "812x620_all_steps", "GPU_concurrency": 1,
        "ssim_validation": data_identity(args.ssim_validation/"RESULT.json"),
        "adam_validation": data_identity(args.adam_validation/"RESULT.json"),
        "smoke": data_identity(args.smoke/"STATE.json"), "controller_source": data_identity(Path(__file__)),
        "frozen_sources": [data_identity(snapshot/name) for name in names], "counted": False,
        "mechanism_count_if_accepted": 1,
        "acceptance": "3 fixed paired seeds median>=10% wall reduction; every full/target PSNR>=-0.1 and SSIM>=-0.002; protected views and freeflight"}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    for label in (["fused_step", "reference"] if args.seed == 1 else ["reference", "fused_step"]):
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full",
            "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build)]
        if label == "fused_step":
            command += ["--ssim-backend", "fused", "--ssim-source", str(args.ssim_source),
                        "--ssim-library", str(args.ssim_library), "--dense-adam-backend", "fused"]
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
            raise RuntimeError("Combined step-fusion training failed")
    old = read(args.output/"reference/STATE.json")
    new = read(args.output/"fused_step/STATE.json")
    if old["sample_sequence_sha256"] != new["sample_sequence_sha256"]:
        state["status"] = "failed_sampling_sequence"
        atomic_json(args.output/"PAIR_STATE.json", state)
        raise ValueError("The fixed sampling sequence changed")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started,
        identical_camera_sampling_sequence=True, reference_seconds=old["training_call_seconds"],
        candidate_seconds=new["training_call_seconds"])
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "reference_seconds", "candidate_seconds")}), flush=True)


if __name__ == "__main__":
    main()
