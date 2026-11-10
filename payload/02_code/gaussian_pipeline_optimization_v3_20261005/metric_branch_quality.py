"""Apply original full-view metrics using each R119 model's own frozen poses."""

import argparse
import inspect
import json
from pathlib import Path
import resource
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/metric_branch_dag_001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import evaluate_pair
from evaluate_pair import mounted_path
import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def current_path(value):
    path = mounted_path(value)
    return path if path.is_absolute() else ROOT/path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.pair.resolve().relative_to(OWNED.resolve())
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    process = psutil.Process()
    atomic_json(args.output/"PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time(), "counted": False})
    paths = [Path(__file__), V1/"evaluate_pair.py", V1/"full_intrinsics.py", V1/"runtime.py"]
    sources = [data_identity(p) for p in paths]
    localizations = {mode: current_path(read(OWNED/folder/"RESULT.json")["localization"])
        for mode, folder in (("serial", "h0"), ("dag", "h1"))}
    loaded = {mode: read(path) for mode, path in localizations.items()}
    by_name = {mode: {r["image_name"]: r for r in value["rows"]} for mode, value in loaded.items()}
    if any(len(v) != 24 for v in by_name.values()) or set(by_name["serial"]) != set(by_name["dag"]):
        raise ValueError("The complete24 frozen names differ between the own maps")
    if loaded["serial"]["PINHOLE_camera"] != loaded["dag"]["PINHOLE_camera"]:
        raise ValueError("The common reference calibration changed")
    for name in by_name["serial"]:
        a, b = by_name["serial"][name], by_name["dag"][name]
        if (a["raw_image"] != b["raw_image"] or a["target_window_PINHOLE"] != b["target_window_PINHOLE"]
                or a["status"] != "localized" or b["status"] != "localized"):
            raise ValueError("A frozen raw reference, rectangle or PnP admission changed")
        with Image.open(current_path(a["image_reference"])) as first, Image.open(current_path(b["image_reference"])) as second:
            if not np.array_equal(np.asarray(first), np.asarray(second)):
                raise ValueError("Own calibrated reference pixels differ")
    for mode in ("serial", "dag"):
        result = read(args.pair/mode/"RESULT.json")
        state = read(args.pair/mode/"training/STATE.json")
        if (not all(result["checks"].values()) or state["status"] != "completed"
                or state["iteration"] != 12000 or state["seed"] != args.seed):
            raise ValueError("Both own complete12000 runs must finish this seed")
    source = inspect.getsource(evaluate_pair.main)
    anchor = '    for label in [args.control_label, args.candidate_label]:\n'
    run_path = 'args.pair/label/'
    if source.count(anchor) != 1 or source.count(run_path) != 3:
        raise ValueError("Original evaluator model/camera boundaries changed")
    own_camera = '''        localization = _own_localizations[label]
        cameras = {}
        for own_row in localization["rows"]:
            with Image.open(mounted_path(own_row["image_reference"])) as own_photo:
                cameras[own_row["image_name"]] = camera_for(own_row["R_CW"], own_row["t_CW"], own_photo.convert("RGB"), own_row["image_name"])
'''
    source = source.replace(anchor, anchor+own_camera).replace(run_path, 'args.pair/label/"training"/')
    namespace = dict(evaluate_pair.main.__globals__)
    namespace["_own_localizations"] = loaded
    exec(compile(source, "<original_metrics_each_R119_map_PnP_camera>", "exec"), namespace)
    atomic_json(args.output/"CONTRACT.json", {"sources": sources, "seed": args.seed,
        "localizations": {mode: data_identity(p) for mode, p in localizations.items()},
        "original_renderer_metrics_and_full24_references": True, "each_model_own_map_poses": True,
        "original_PSNR_min_delta_dB": -.1, "original_SSIM_min_delta": -.002,
        "cross_map_pose_equality_or_GT_claim": False, "counted": False})
    previous = sys.argv
    try:
        sys.argv = [str(V1/"evaluate_pair.py"), "--runtime", str(ROOT/"04_outputs/box2_3dgs_display_v1_20260930"),
            "--reference-run", str(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"), "--pair", str(args.pair),
            "--control-label", "serial", "--candidate-label", "dag", "--iterations", "12000",
            "--evaluation-folder", "evaluation_seed"+str(args.seed), "--localization", str(localizations["serial"]),
            "--input-version", "new_map_admission_v1"]
        namespace["main"]()
    finally:
        sys.argv = previous
    native = read(args.pair/("evaluation_seed"+str(args.seed))/"EVALUATION.json")
    results = {mode: {r["image_name"]: r for r in native["results"][mode]["rows"]} for mode in loaded}
    deltas = []
    for name in sorted(by_name["serial"]):
        for region in ("full_image", "target"):
            a, b = results["serial"][name].get(region), results["dag"][name].get(region)
            if a is None or b is None:
                raise ValueError("A complete24 full/target metric is missing")
            row = {"image_name": name, "region": region, "PSNR_delta_dB": b["PSNR_dB"]-a["PSNR_dB"],
                   "SSIM_delta": b["SSIM"]-a["SSIM"]}
            row["passed"] = row["PSNR_delta_dB"] >= -.1 and row["SSIM_delta"] >= -.002
            deltas.append(row)
    import torch
    actual_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    actual_cuda = torch.cuda.max_memory_reserved()
    checks = {"all24_full_target_metrics": len(deltas) == 48,
        "all_original_per_view_quality": all(r["passed"] for r in deltas),
        "original_mean_guard": native["development_metric_guard_passed"],
        "WSL_CUDA_bounds": actual_rss <= 32*1024**3 and actual_cuda <= 12*1024**3,
        "original_time_bound": time.perf_counter()-STARTED <= 1750,
        "source_unchanged": sources == [data_identity(p) for p in paths]}
    atomic_json(args.output/"RESULT.json", {"status": "own_seed_quality_passed" if all(checks.values()) else "required_own_full24_quality_failed",
        "seed": args.seed, "checks": checks, "per_view_deltas": deltas,
        "native_evaluation": str(args.pair/("evaluation_seed"+str(args.seed))/"EVALUATION.json"),
        "actual_WSL_peak_RSS_bytes": actual_rss, "actual_CUDA_peak_reserved_bytes": actual_cuda,
        "complete_seconds": time.perf_counter()-STARTED, "counted": False})
    if not all(checks.values()):
        raise AssertionError("Original complete24 full/target quality protection failed")


if __name__ == "__main__":
    main()
