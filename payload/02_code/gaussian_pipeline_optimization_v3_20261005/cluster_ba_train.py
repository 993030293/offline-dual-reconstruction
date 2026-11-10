"""Train each admitted R129 map with the unchanged complete reference camera route."""

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/ba001"
sys.path[:0] = [str(V2), str(V1)]
from runtime import atomic_json, data_identity
import train_camera_execution
import train_case
import checkpoint_adapter


def mounted(value):
    value = str(value).replace("\\", "/")
    return Path("/mnt/"+value[0].lower()+value[2:]) if len(value) > 2 and value[1] == ":" else Path(value)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "cluster"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-file", type=Path)
    parser.add_argument("--simulate-crash-after", type=int, choices=[8000])
    args = parser.parse_args()
    for folder in ("a0", "a1", "h0", "h1"):
        result = read(OWNED/folder/"RESULT.json")
        if not all(result["checks"].values()):
            raise ValueError("Both own admissions and all24 localizations must precede training")
    admission = OWNED/("a0" if args.mode == "auto" else "a1")/"export/TRAINING_ADMISSION.json"
    admitted = read(admission)
    dataset = mounted(admitted["dataset"])
    count = admitted["training_count"]
    qualified = {r["image_name"] for r in admitted["rows"] if r["training_role"] == "admitted"}
    images = {p.name for p in (dataset/"images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if type(count) is not int or count != 1016 or images != qualified or len(images) != count:
        raise ValueError("The complete own admitted training image membership changed")
    if admitted["status"] != "export_and_admission_verified_training_pending":
        raise ValueError("An unadmitted map cannot be trained")
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    process = psutil.Process()
    atomic_json(args.output/"PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time(), "counted": False})
    config = ROOT/"02_code/gaussian_pipeline_optimization_v3_20261005/parallel_a_003/camera_config.json"
    paths = [Path(__file__), config, V2/"train_camera_execution.py", V2/"camera_execution.py",
        V2/"prefetch_camera_images.py", V1/"train_case.py", V1/"checkpoint_adapter.py", V1/"full_intrinsics.py",
        V1/"gsplat_adapter.py", V1/"ply_serialization.py"]
    sources = [data_identity(p) for p in paths]
    external = {"schema_version": 1, "own_mode": args.mode, "training_count": count,
        "own_admission": data_identity(admission), "source_sha256": [r["sha256"] for r in sources],
        "same_original_full_K_camera_flags_and_prefetch": True, "complete_final_state_common_measurement": True}
    original_recovery, original_json, previous = checkpoint_adapter.RecoveryState, train_case.atomic_json, sys.argv

    class OwnRecovery(original_recovery):
        def __init__(self, output, identity, interval, resume=None, crash_after=None):
            if "own_map_adapter" in identity:
                raise ValueError("Own-map recovery identity already bound")
            super().__init__(output, {**identity, "own_map_adapter": external}, interval, resume, crash_after)

        def step(self, model, iteration, scene, stack, indices, ema_loss, ema_depth, options):
            saving_options = options
            if iteration == options.iterations and iteration % self.interval == 0:
                saving_options = copy.copy(options)
                saving_options.iterations += 1
            return super().step(model, iteration, scene, stack, indices, ema_loss, ema_depth, saving_options)

    output = args.output/"training"

    def publish(path, value):
        if Path(path) == output/"CONFIG.json":
            value = {**value, "own_map_adapter": external}
        original_json(path, value)

    atomic_json(args.output/"CONTRACT.json", {"sources": sources, "external": external, "seed": args.seed,
        "registered_denominator": admitted["full_registered_denominator"], "initial_denominator": 1002,
        "expected_training_images": count, "iterations": 12000, "checkpoint_every": 4000, "counted": False})
    try:
        checkpoint_adapter.RecoveryState, train_case.atomic_json = OwnRecovery, publish
        sys.argv = [str(V2/"train_camera_execution.py"), "--camera-config", str(config),
            "--runtime", str(ROOT/"04_outputs/box2_3dgs_display_v1_20260930"), "--dataset", str(dataset),
            "--output", str(output), "--projection", "full", "--iterations", "12000", "--seed", str(args.seed),
            "--maximum-seconds", "1750", "--input-version", "new_map_admission_v1", "--expected-image-count", str(count),
            "--gsplat-build", str(ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001"),
            "--checkpoint-every", "4000", "--vectorized-ply", "--verify-native-ply"]
        if args.resume_file:
            sys.argv += ["--resume-file", str(args.resume_file)]
        if args.simulate_crash_after:
            sys.argv += ["--simulate-crash-after", str(args.simulate_crash_after)]
        train_camera_execution.main()
        state = read(output/"STATE.json")
        checks = {"full12000_completed": state["status"] == "completed" and state["iteration"] == 12000,
            "all1016_qualified_images_kept": state["image_count"] == count,
            "sources_unchanged": sources == [data_identity(p) for p in paths],
            "complete_final_checkpoint": (output/"checkpoint_012000.pth").is_file()}
        atomic_json(args.output/"RESULT.json", {"status": "full_training_complete_quality_pending" if all(checks.values()) else "required_training_guard_failed",
            "mode": args.mode, "seed": args.seed, "checks": checks, "training": str(output),
            "complete_seconds": time.perf_counter()-STARTED, "counted": False})
        if not all(checks.values()):
            raise AssertionError("An unchanged complete own-map training guard failed")
    finally:
        checkpoint_adapter.RecoveryState, train_case.atomic_json, sys.argv = original_recovery, original_json, previous
        atomic_json(args.output/"BINDING_RESTORATION.json", {"recovery_restored": checkpoint_adapter.RecoveryState is original_recovery,
            "publisher_restored": train_case.atomic_json is original_json, "argv_restored": sys.argv is previous, "counted": False})


if __name__ == "__main__":
    main()


