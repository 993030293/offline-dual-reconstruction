"""Capture actual complete inverse API probe costs and owned terminal state."""

import subprocess
import sys
import time

import psutil

from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"


def main():
    leaf, node = OUT / "k0", OUT / "k0_node"
    command = [sys.executable, "-X", "utf8", str(CODE / "inverse_overlap_probe_node.py"),
        "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
        "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/probe_camera_inverse_overlap.py",
        "--output", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001/k0"]
    started = time.perf_counter()
    with (OUT / "k0.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
        returncode = child.wait()
    elapsed = time.perf_counter() - started
    node_result = read(node / "NODE.json")
    atomic_json(OUT / "K0_EXECUTION.json", {"owner": owner, "returncode": returncode,
        "actual_spawn_to_exit_seconds": elapsed, "controller": str(node / "NODE.json"), "new_optimization_count": 0})
    if returncode or node_result["status"] != "completed" or node_result["live_Windows_after_wait"] or any(r["alive"] for r in node_result["actual_Linux_after_wait"]):
        raise RuntimeError("The actual inverse API probe failed; preserve its original receipt")
    result = read(leaf / "RESULT.json")
    if not all(result["checks"].values()):
        raise AssertionError("A protected full inverse API probe check failed")
    print({"actual_spawn_to_exit_seconds": elapsed, "medians": result["component_medians"],
        "raw_component_reduction": result["raw_component_reduction"], "new": 0})


if __name__ == "__main__":
    main()
