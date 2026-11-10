"""Run one bounded mask-free PGSR training job with machine receipts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import psutil


MINIMUM_FREE_GPU_MIB = 6144
MINIMUM_HEADROOM_BELOW_MONITOR_MIB = 5120
MAXIMUM_TOTAL_GPU_USED_MIB = 14336


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_status() -> dict[str, int | str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    uuid, used, free = [value.strip() for value in result.stdout.strip().split(",")]
    return {"uuid": uuid, "used_mib": int(used), "free_mib": int(free)}


def assess_gpu_budget(gpu: dict[str, int | str]) -> dict[str, int | str]:
    free_mib = int(gpu["free_mib"])
    used_mib = int(gpu["used_mib"])
    headroom_mib = MAXIMUM_TOTAL_GPU_USED_MIB - used_mib
    if free_mib < MINIMUM_FREE_GPU_MIB or headroom_mib < MINIMUM_HEADROOM_BELOW_MONITOR_MIB:
        raise RuntimeError(
            "Insufficient PGSR GPU budget: "
            f"free={free_mib} MiB (minimum={MINIMUM_FREE_GPU_MIB}), "
            f"headroom_to_monitor={headroom_mib} MiB "
            f"(minimum={MINIMUM_HEADROOM_BELOW_MONITOR_MIB})"
        )
    return {
        "status": "admitted",
        "free_mib": free_mib,
        "used_mib": used_mib,
        "headroom_to_monitor_mib": headroom_mib,
        "minimum_free_gpu_mib": MINIMUM_FREE_GPU_MIB,
        "minimum_headroom_below_monitor_mib": MINIMUM_HEADROOM_BELOW_MONITOR_MIB,
        "maximum_total_gpu_used_mib": MAXIMUM_TOTAL_GPU_USED_MIB,
    }


def verify_input(stage: Path, receipt: dict) -> None:
    input_root = stage / "input"
    for row in receipt["images"]:
        path = input_root / "images" / row["name"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Image input identity changed: {path}")
    for row in receipt["sparse_files"]:
        path = input_root / "sparse" / row["name"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Sparse input identity changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stage-name", default="06_pgsr")
    args = parser.parse_args()

    run = args.run_root.resolve(strict=True)
    stage = run / args.stage_name
    source = args.source.resolve(strict=True)
    model = stage / "model_pgsr_15000"
    logs = stage / "logs"
    logs.mkdir(exist_ok=True)
    if model.exists() or (stage / "launch.json").exists():
        raise FileExistsError("PGSR launch or output already exists; automatic retry is forbidden")
    input_receipt = read_json(stage / "metric_input_receipt.json")
    if input_receipt.get("schema_version") != "e26_mask_free_metric_pgsr_inlier_input_v3":
        raise RuntimeError("Metric PGSR input lacks repaired lineage and geometry admission")
    if input_receipt.get("status") != "admitted_for_mask_free_pgsr":
        raise RuntimeError("Metric PGSR input is not admitted")
    if input_receipt.get("independent_geometry_admitted_camera_count") != input_receipt.get(
        "registered_camera_count"
    ):
        raise RuntimeError("Independent geometry admission count is inconsistent")
    if input_receipt.get("language_model_used") is not False or input_receipt.get("object_mask_used") is not False:
        raise RuntimeError("No-LLM or no-mask contract is not satisfied")
    verify_input(stage, input_receipt)
    gpu = gpu_status()
    gpu_budget = assess_gpu_budget(gpu)
    if shutil.disk_usage(stage).free < 64 * 2**30:
        raise RuntimeError("Insufficient free disk space")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import torch

    command = [
        str(source / "train.py"),
        "-s", str(stage / "input"),
        "-m", str(model),
        "-r", "4",
        "--iterations", "15000",
        "--single_view_weight_from_iter", "7000",
        "--multi_view_weight_from_iter", "7000",
        "--multi_view_sample_num", "16384",
        "--max_abs_split_points", "20000",
        "--max_all_points", "2500000",
        "--save_iterations", "7000", "15000",
        "--test_iterations", "7000", "15000",
        "--checkpoint_iterations", "7000", "15000",
    ]
    process = psutil.Process()
    launch = {
        "schema_version": "e26_c1_mask_free_pgsr_launch_v1",
        "status": "launched",
        "command": command,
        "pid": os.getpid(),
        "process_create_time": process.create_time(),
        "gpu_before_launch": gpu,
        "gpu_budget": gpu_budget,
        "camera_count": input_receipt["registered_camera_count"],
        "language_model_used": False,
        "object_mask_used": False,
        "iterations": 15000,
        "minimum_free_gpu_mib": MINIMUM_FREE_GPU_MIB,
        "maximum_total_gpu_used_mib": MAXIMUM_TOTAL_GPU_USED_MIB,
        "maximum_wall_seconds": 7200,
        "source": str(source),
        "source_train_sha256": sha256_file(source / "train.py"),
    }
    write_json_new(stage / "launch.json", launch)

    started = time.monotonic()
    stop = threading.Event()
    backward_count = [0]
    last_loss = [None]
    peaks = {"rss_bytes": 0, "cuda_reserved_bytes": 0, "total_gpu_used_mib": 0}
    native_backward = torch.autograd.backward

    def checked_backward(tensors, *positional, **keywords):
        values = (tensors,) if isinstance(tensors, torch.Tensor) else tensors
        for value in values:
            if not bool(torch.isfinite(value.detach()).all()):
                raise FloatingPointError("Nonfinite PGSR training loss")
        if isinstance(tensors, torch.Tensor) and tensors.numel() == 1:
            last_loss[0] = float(tensors.detach())
        result = native_backward(tensors, *positional, **keywords)
        backward_count[0] += 1
        return result

    def monitor() -> None:
        while not stop.wait(5):
            rss = process.memory_info().rss + sum(
                child.memory_info().rss
                for child in process.children(recursive=True)
                if child.is_running()
            )
            reserved = torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0
            current_gpu = gpu_status()
            peaks["rss_bytes"] = max(peaks["rss_bytes"], rss)
            peaks["cuda_reserved_bytes"] = max(peaks["cuda_reserved_bytes"], reserved)
            peaks["total_gpu_used_mib"] = max(peaks["total_gpu_used_mib"], current_gpu["used_mib"])
            heartbeat = {
                "status": "running",
                "elapsed_seconds": time.monotonic() - started,
                "completed_backward_calls": backward_count[0],
                "last_loss": last_loss[0],
                "rss_bytes": rss,
                "cuda_reserved_bytes": reserved,
                "gpu": current_gpu,
            }
            atomic_json(stage / "heartbeat.json", heartbeat)
            if heartbeat["elapsed_seconds"] > 7200 or current_gpu["used_mib"] > MAXIMUM_TOTAL_GPU_USED_MIB:
                atomic_json(stage / "resource_failure.json", heartbeat)
                os._exit(124)

    torch.autograd.backward = checked_backward
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        os.chdir(source)
        sys.path.insert(0, str(source))
        sys.argv = command
        with (logs / "train.log").open("x", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                runpy.run_path(str(source / "train.py"), run_name="__main__")
        if backward_count[0] != 15000:
            raise RuntimeError(f"Expected 15000 backward calls, observed {backward_count[0]}")
        verify_input(stage, input_receipt)
        from plyfile import PlyData

        ply = model / "point_cloud" / "iteration_15000" / "point_cloud.ply"
        checkpoint = model / "chkpnt15000.pth"
        vertices = PlyData.read(ply)["vertex"].data
        if len(vertices) == 0 or any(not np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
            raise RuntimeError("Final Gaussian PLY is empty or nonfinite")
        state, iteration = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if iteration != 15000 or len(state[1]) != len(vertices):
            raise RuntimeError("Checkpoint and Gaussian PLY disagree")
        cameras = read_json(model / "cameras.json")
        expected_cameras = int(input_receipt["registered_camera_count"])
        if len(cameras) != expected_cameras:
            raise RuntimeError(f"Expected {expected_cameras} trained cameras, observed {len(cameras)}")
        receipt = {
            "schema_version": "e26_c1_mask_free_pgsr_training_receipt_v1",
            "status": "trained_15000_checkpoint_verified",
            "iteration": iteration,
            "registered_camera_count": len(cameras),
            "gaussian_count": len(vertices),
            "completed_backward_calls": backward_count[0],
            "last_loss": last_loss[0],
            "runtime_seconds": time.monotonic() - started,
            "maximum_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peaks": peaks,
            "input_identities_preserved": True,
            "language_model_used": False,
            "object_mask_used": False,
            "checkpoint_sha256": sha256_file(checkpoint),
            "ply_sha256": sha256_file(ply),
        }
        write_json_new(stage / "training_receipt.json", receipt)
        atomic_json(stage / "heartbeat.json", {"status": "complete", **receipt})
        print(json.dumps(receipt, indent=2))
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "completed_backward_calls": backward_count[0],
            "elapsed_seconds": time.monotonic() - started,
            "peaks": peaks,
        }
        atomic_json(stage / "training_failure.json", failure)
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
        torch.autograd.backward = native_backward
        sys.argv = previous_argv
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
