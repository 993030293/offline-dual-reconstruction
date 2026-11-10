"""Compare a single backend intervention using frozen scripts and one GPU queue."""

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
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=["accelerated", "fused_adam", "sparse_adam", "gsplat"], required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--rasterizer-library", type=Path)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--gsplat-build", type=Path)
    parser.add_argument("--density-validation", type=Path)
    parser.add_argument("--smoke-validation", type=Path)
    parser.add_argument("--candidate-iterations", type=int, choices=[10000, 12000], default=10000)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    expected = "numerical_passed_training_pending" if args.candidate == "fused_adam" else "passed"
    if validation["status"] != expected:
        raise ValueError("Candidate did not pass its registered numerical or projection gate")
    if args.candidate in ("accelerated", "sparse_adam") and args.rasterizer_library is None:
        raise ValueError("Accelerated and sparse candidates require an explicit compiled rasterizer")
    if args.candidate == "gsplat" and args.gsplat_build is None:
        raise ValueError("gsplat requires its explicit isolated build directory")
    if args.candidate_iterations != 10000 and args.candidate != "gsplat":
        raise ValueError("The separately registered extended budget applies only to gsplat")
    source_root = args.training_source_dir or Path(__file__).parent
    if args.candidate == "gsplat":
        if args.density_validation is None or args.smoke_validation is None:
            raise ValueError("gsplat also requires real density and training smoke receipts")
        density = json.loads(args.density_validation.read_text())
        smoke = json.loads(args.smoke_validation.read_text())
        if density["status"] != "passed" or smoke["status"] != "completed" or smoke["iteration"] < 1100:
            raise ValueError("gsplat did not pass the density and real-growth training prerequisites")
        expected_adapter = validation["runtime"]["gsplat"]["adapter"]["sha256"]
        if density.get("candidate_adapter_sha256") != expected_adapter:
            raise ValueError("Projection and density receipts refer to different adapters")
        if data_identity(source_root / "gsplat_adapter.py")["sha256"] != expected_adapter:
            raise ValueError("The active gsplat adapter differs from the projection-validated version")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    files = ["train_case.py", "runtime.py", "full_intrinsics.py", "checkpoint_adapter.py", "optimizer_adapter.py"]
    if args.candidate == "gsplat":
        files.append("gsplat_adapter.py")
    for name in files:
        shutil.copyfile(source_root/name, snapshot/name)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "candidate": args.candidate,
        "runs": [], "same_fixed_212_inputs": str(args.dataset), "projection": "full", "ssim": "reference",
        "checkpointing_during_timed_pair": False, "validation": data_identity(args.validation),
        "control_iterations": 10000, "candidate_iterations": args.candidate_iterations,
        "comparison": "equal_iterations" if args.candidate_iterations == 10000 else "fixed_10k_control_12k_candidate_budget",
        "density_validation": data_identity(args.density_validation) if args.density_validation else None,
        "smoke_validation": data_identity(args.smoke_validation) if args.smoke_validation else None,
        "source": data_identity(Path(__file__)), "frozen_training_sources": [data_identity(snapshot/name) for name in files],
        "acceptance": "3 paired seeds; median training wall reduction >=10%; each seed full/target PSNR delta >=-0.1 dB and SSIM >=-0.002; protected views",
        "counted": False}
    started = time.monotonic()
    atomic_json(args.output/"PAIR_STATE.json", state)
    for label in ([args.candidate, "reference"] if args.seed == 1 else ["reference", args.candidate]):
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full", "--seed", str(args.seed)]
        if args.candidate == "sparse_adam" or (args.candidate == "accelerated" and label != "reference"):
            command += ["--rasterizer-library", str(args.rasterizer_library)]
        if label == "sparse_adam":
            command += ["--optimizer-type", "sparse_adam"]
        if label == "fused_adam":
            command += ["--dense-adam-backend", "fused"]
        if label == "gsplat":
            command += ["--gsplat-build", str(args.gsplat_build), "--iterations", str(args.candidate_iterations)]
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
            raise RuntimeError("Paired backend run failed")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({"status": state["status"], "candidate": args.candidate, "seed": args.seed}), flush=True)


if __name__ == "__main__":
    main()
