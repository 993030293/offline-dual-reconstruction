"""Run one fixed-configuration INRIA training case in the existing WSL environment."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case-dir", type=Path, required=True)
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--phase", choices=["preflight", "main"], required=True)
parser.add_argument("--attempt", type=int, default=1)
cli = parser.parse_args()
case = cli.case_dir.resolve()
runtime = cli.runtime.resolve()
repo = runtime / "source/gaussian-splatting"
libraries = sorted((runtime / "runtime/rasterizer-build").glob("lib.*"))
if len(libraries) != 1:
    raise RuntimeError("The preserved rasterizer runtime is ambiguous")
sys.path[:0] = [str(repo), str(libraries[0])]

import torch
import numpy as np
import train as reference
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.dataset_readers import fetchPly, storePly
from scene.colmap_loader import read_points3D_binary
from plyfile import PlyData
from utils.general_utils import safe_state

commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if commit != "54c035f7834b564019656c3e3fcc3646292f727d":
    raise RuntimeError("The frozen INRIA implementation commit changed")
main = cli.phase == "main"
phase_label = cli.phase if cli.attempt == 1 else f"{cli.phase}_attempt{cli.attempt}"
ply_path = case / "dataset/sparse/0/points3D.ply"
properties = set(PlyData.read(ply_path)["vertex"].data.dtype.names)
if not {"nx", "ny", "nz"}.issubset(properties):
    xyz, rgb, _ = read_points3D_binary(case / "dataset/sparse/0/points3D.bin")
    repaired = ply_path.with_name("points3D_compatible.tmp.ply")
    storePly(repaired, xyz, rgb)
    checked = fetchPly(repaired)
    if not np.allclose(checked.points, xyz, atol=1e-6) or not np.allclose(checked.colors * 255, rgb, atol=1e-6):
        raise RuntimeError("INRIA-compatible PLY changed the optimized points or colors")
    original = ply_path.with_name("points3D_colmap_xyzrgb.ply")
    if original.exists():
        raise FileExistsError("Original PLY preservation path already exists")
    ply_path.rename(original)
    repaired.rename(ply_path)
    save_json(case / "PLY_COMPATIBILITY_RECEIPT.json", {"status": "passed",
        "original_PLY": str(original), "compatible_PLY": str(ply_path), "point_count": len(xyz),
        "repair": "INRIA_storePly_adds_zero_normals_required_by_fetchPly",
        "points_and_colors_verified": True})
native_parser = argparse.ArgumentParser()
model_group = ModelParams(native_parser)
optimization_group = OptimizationParams(native_parser)
pipeline_group = PipelineParams(native_parser)
native = native_parser.parse_args([])
native.source_path = str(case / "dataset")
native.model_path = str(case / ("model" if main else f"{phase_label}_model"))
native.resolution = 2 if main else 4
native.data_device = "cpu"
native.iterations = 10000 if main else 20
native.densify_until_iter = 7000 if main else 0
native.depth_l1_weight_init = native.depth_l1_weight_final = 0.0
if Path(native.model_path).exists():
    raise FileExistsError("Training outputs already exist; preserve them")
config = json.loads((runtime / "MAIN_CONFIG.json").read_text())
for key, value in config["parameters"].items():
    if main and key not in ("source_path", "model_path") and getattr(native, key) != value:
        raise RuntimeError(f"Training configuration drift: {key}")
model = model_group.extract(native)
optimization = optimization_group.extract(native)
pipeline = pipeline_group.extract(native)
save_json(case / f"{phase_label.upper()}_CONFIG.json", {
    "reference_repository": "https://github.com/graphdeco-inria/gaussian-splatting",
    "reference_commit": commit, "parameters": vars(native), "random_seed": 0,
    "camera_pose_optimization": False, "baseline_training_configuration_matched": main,
    "previous_pgsr_checkpoint_used": False, "maximum_wall_seconds": 1800,
    "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0)})
state_path = case / f"{phase_label.upper()}_STATE.json"
start = time.monotonic()
original_report = reference.training_report
last_state = {"status": "starting", "phase": cli.phase, "pid": os.getpid(), "iteration": 0}
save_json(state_path, last_state)


def monitored_report(writer, iteration, l1, loss, loss_function, elapsed,
                     test_iterations, scene, render, render_args, train_test_exp):
    global last_state
    original_report(writer, iteration, l1, loss, loss_function, elapsed,
                    test_iterations, scene, render, render_args, train_test_exp)
    if time.monotonic() - start > 1800:
        raise TimeoutError("The registered training wall-clock cap was reached")
    if iteration == 1 or iteration % 100 == 0 or iteration == optimization.iterations:
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training produced a nonfinite loss")
        last_state = {"status": "running", "phase": cli.phase, "pid": os.getpid(),
            "iteration": iteration, "target_iterations": optimization.iterations,
            "completed_backward_calls": iteration, "loss": float(loss.item()), "l1": float(l1.item()),
            "gaussian_count": int(scene.gaussians.get_xyz.shape[0]),
            "elapsed_seconds": time.monotonic() - start,
            "maximum_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        save_json(state_path, last_state)
        with (case / f"{phase_label}_progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(last_state) + "\n")


reference.training_report = monitored_report
safe_state(False)
try:
    reference.training(model, optimization, pipeline, [10000] if main else [20],
                       [7000, 10000] if main else [20], [], None, -1)
    final_ply = Path(native.model_path) / f"point_cloud/iteration_{native.iterations}/point_cloud.ply"
    if not final_ply.is_file():
        raise RuntimeError("Final Gaussian checkpoint is missing")
    image_count = len(list((case / "dataset/images").glob("*.jpg")))
    last_state.update(status="completed", elapsed_seconds=time.monotonic() - start,
                      final_checkpoint=str(final_ply), image_count=image_count,
                      mean_training_exposures=native.iterations / image_count)
    save_json(state_path, last_state)
    print(json.dumps(last_state))
except Exception:
    last_state.update(status="failed", error=traceback.format_exc())
    save_json(state_path, last_state)
    raise
