"""Time native complete recovery without executing any training iteration."""

import time

STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from private_checkpoint_load import private_checkpoint_load

DATASET = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
CHECKPOINT = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_execution_training_001/recovery/continuous/checkpoint_001000.pth"
RUNTIME = ROOT / "04_outputs/box2_3dgs_display_v1_20260930"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def timed_recovery(adapter, gaussians, torch, device_layout):
    originals = {name: getattr(adapter, name) for name in ("exact_equal", "data_identity", "atomic_json")}
    original_restore = gaussians.restore
    original_load = torch.load
    rows, depth = [], 0
    load_layouts, load_storages = [], []

    def storages(value, destination):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if storage.nbytes():
                destination.add((str(value.device), storage.data_ptr(), storage.nbytes()))
        elif isinstance(value, dict):
            for child in value.values():
                storages(child, destination)
        elif isinstance(value, (list, tuple)):
            for child in value:
                storages(child, destination)

    def observed_load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        load_layouts.append(device_layout(value))
        tokens = set()
        storages(value, tokens)
        load_storages.append(tokens)
        return value

    def measured(name, function, *args, **kwargs):
        before = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            rows.append({"operation": name, "API_wall_seconds": time.perf_counter() - before})

    def equal(left, right):
        nonlocal depth
        top = depth == 0
        depth += 1
        before = time.perf_counter()
        try:
            return originals["exact_equal"](left, right)
        finally:
            depth -= 1
            if top:
                rows.append({"operation": "original_exact_equal_top_level", "API_wall_seconds": time.perf_counter() - before})

    adapter.exact_equal = equal
    adapter.data_identity = lambda *args, **kwargs: measured("checkpoint_identity", originals["data_identity"], *args, **kwargs)
    adapter.atomic_json = lambda *args, **kwargs: measured("original_recovery_receipt_write", originals["atomic_json"], *args, **kwargs)
    gaussians.restore = lambda *args, **kwargs: measured("original_model_and_Adam_restore", original_restore, *args, **kwargs)
    torch.load = observed_load
    trace = {"rows": rows, "API_times_are_not_native_kernel_times": True}
    try:
        yield trace
    finally:
        for name, original in originals.items():
            setattr(adapter, name, original)
        gaussians.restore = original_restore
        torch.load = original_load
        trace["all_bindings_restored"] = all(getattr(adapter, name) is original for name, original in originals.items())
        trace["all_bindings_restored"] = trace["all_bindings_restored"] and gaussians.restore == original_restore
        trace["all_bindings_restored"] = trace["all_bindings_restored"] and torch.load is original_load
        trace["recursive_depth_closed"] = depth == 0
        trace["two_complete_load_layouts_devices_and_alias_groups_exact"] = len(load_layouts) == 2 and load_layouts[0] == load_layouts[1]
        trace["two_loaded_states_own_independent_nonempty_storages"] = len(load_storages) == 2 and not (load_storages[0] & load_storages[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT.resolve())
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": psutil.Process().pid,
        "creation_time": psutil.Process().create_time()})
    sources = [Path(__file__), CODE / "private_checkpoint_load.py", CODE / "mapped_image_linux_guard.py",
        V1 / "runtime.py", V1 / "checkpoint_adapter.py", V1 / "full_intrinsics.py",
        V2 / "camera_execution.py", V2 / "prefetch_camera_images.py",
        RUNTIME / "source/gaussian-splatting/scene/gaussian_model.py",
        RUNTIME / "source/gaussian-splatting/scene/__init__.py"]
    inputs = [CHECKPOINT, CHECKPOINT.with_suffix(".json"), *sorted(path for path in DATASET.rglob("*") if path.is_file())]
    source_ids = [data_identity(path) for path in sources]
    identities = [data_identity(path) for path in inputs]
    archive = output / "source_copies"
    archive.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, archive / (f"{index:02d}_" + path.name))
    original_receipt = read(CHECKPOINT.with_suffix(".json"))
    if (len(inputs) != 1038 or original_receipt["status"] != "complete"
            or identities[0]["sha256"] != original_receipt["checkpoint"]["sha256"]
            or identities[0]["bytes"] != 97079305):
        raise AssertionError("The original complete checkpoint or dataset contract changed")
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise MemoryError("The original disk reserve is unavailable")
    atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": identities,
        "original_checkpoint_identity": original_receipt["identity"], "no_training_iterations": True,
        "original_full_K1016_and_two_independent_native_loads": True, "new_optimization_count": 0})
    runtime = activate_reference(RUNTIME)
    import numpy as np
    import torch
    import checkpoint_adapter
    from arguments import ModelParams, OptimizationParams
    from camera_execution import camera_execution
    from full_intrinsics import camera_loader_adapter
    from prefetch_camera_images import prefetch_camera_images
    from profile_checkpoint_mmap import device_layout
    from scene import GaussianModel, Scene

    if torch.__version__ != "2.7.1+cu128":
        raise ValueError("The installed native Torch contract changed")
    phases = {}
    try:
        with LinuxRSSGuard(output, atomic_json, maximum_seconds=540, maximum_bytes=16 * 1024**3):
            free, total = torch.cuda.mem_get_info()
            budget = min(12 * 1024**3, free - 2 * 1024**3)
            if budget <= 0:
                raise MemoryError("The original CUDA headroom is unavailable")
            torch.cuda.reset_peak_memory_stats()
            native_parser = argparse.ArgumentParser()
            model_group = ModelParams(native_parser)
            opt_group = OptimizationParams(native_parser)
            arguments = native_parser.parse_args([])
            for key, value in original_receipt["identity"]["parameters"].items():
                setattr(arguments, key, value)
            arguments.source_path = str(DATASET)
            arguments.model_path = str(output / "scene")
            Path(arguments.model_path).mkdir()
            before = time.perf_counter()
            owned = output / "checkpoint_001000.pth"
            shutil.copyfile(CHECKPOINT, owned)
            owned_identity = data_identity(owned)
            if any(owned_identity[key] != identities[0][key] for key in ("sha256", "bytes")):
                raise AssertionError("The private native checkpoint copy differs")
            atomic_json(owned.with_suffix(".json"), {**original_receipt, "checkpoint": owned_identity})
            phases["physical_checkpoint_copy_and_complete_verify_seconds"] = time.perf_counter() - before
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.synchronize()
            before = time.perf_counter()
            flags = original_receipt["identity"]["external_adapters"]["camera_execution"]
            with (camera_loader_adapter(DATASET), camera_execution(**{key: flags[key]
                    for key in ("opaque_alpha", "clampfree", "calibrated_projection")}),
                  prefetch_camera_images(4, 8) as prefetch):
                scene = Scene(model_group.extract(arguments), GaussianModel(3), shuffle=True)
            torch.cuda.synchronize()
            phases["complete_original_Scene_seconds"] = time.perf_counter() - before
            if len(scene.getTrainCameras()) != 1016 or prefetch["prepared"] != 1016 or prefetch["consumed"] != 1016:
                raise AssertionError("The complete original camera denominator changed")
            restore = checkpoint_adapter.RecoveryState(output, original_receipt["identity"], 500, resume=owned)
            before = time.perf_counter()
            with private_checkpoint_load(owned) as loads, timed_recovery(checkpoint_adapter, scene.gaussians, torch, device_layout) as trace:
                iteration = restore.prepare(scene.gaussians, opt_group.extract(arguments), scene)
                torch.cuda.synchronize()
                phases["original_complete_prepare_seconds"] = time.perf_counter() - before
                before = time.perf_counter()
                stack, indices, ema, depth = restore.restore_loop(scene, [], [], 0., 0.)
                torch.cuda.synchronize()
                phases["original_complete_loop_RNG_restore_seconds"] = time.perf_counter() - before
            restored = read(output / "RESTORE_CHECK.json")
            layout = device_layout(restore.restored)
            original_rng = random.Random()
            original_rng.setstate(restore.restored["rng_python"])
            actual_rng = random.Random()
            actual_rng.setstate(random.getstate())
            original_queue = list(restore.restored["remaining_cameras"])
            actual_queue = [camera.image_name for camera in stack]
            original_names, actual_names = [], []
            for _ in range(32):
                if not original_queue:
                    original_queue = list(restore.restored["camera_order"])
                if not actual_queue:
                    actual_queue = [camera.image_name for camera in scene.getTrainCameras()]
                original_names.append(original_queue.pop(original_rng.randint(0, len(original_queue) - 1)))
                actual_names.append(actual_queue.pop(actual_rng.randint(0, len(actual_queue) - 1)))
            peak = torch.cuda.max_memory_reserved()
            checks = {
                "all_original_nine_restore_fields": len(restored["checks"]) == 9 and all(restored["checks"].values()),
                "actual_iteration1000_no_step_and_1016_original_cameras": iteration == 1000 and len(scene.getTrainCameras()) == 1016,
                "two_native_private_saved_device_loads": len(loads["calls"]) == 2
                    and all(row["mmap"] and row["map_location_argument_unchanged"] == "None" for row in loads["calls"]),
                "all_original_local_bindings_closed": loads["original_global_loader_restored"]
                    and trace["all_bindings_restored"] and trace["recursive_depth_closed"],
                "complete_load_layout_devices_aliases_and_independent_storages": trace["two_complete_load_layouts_devices_and_alias_groups_exact"]
                    and trace["two_loaded_states_own_independent_nonempty_storages"],
                "next32_names_exact": original_names == actual_names,
                "complete_original_camera_identity_and_RNG": len(restored["checks"]) == 9 and all(restored["checks"].values()),
                "original_CUDA_and_disk_guards": peak <= budget and shutil.disk_usage(output).free >= 20 * 1024**3,
                "sources_original_inputs_and_owned_checkpoint_unchanged": source_ids == [data_identity(path) for path in sources]
                    and identities == [data_identity(path) for path in inputs] and owned_identity == data_identity(owned),
            }
            before = time.perf_counter()
            del restore, scene, stack, indices
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            phases["actual_owned_scene_and_CUDA_close_seconds"] = time.perf_counter() - before
            if not all(checks.values()):
                raise AssertionError({key: value for key, value in checks.items() if not value})
            result = {"status": "complete_native_restore_cost_diagnostic_passed", "checks": checks,
                "phases": phases, "private_two_load_trace": loads, "native_recovery_API_scopes": trace,
                "complete_original_restore_receipt": restored, "complete_saved_device_alias_layout": layout,
                "actual_next32_names": actual_names, "peak_CUDA_reserved_bytes": peak,
                "CUDA_budget_bytes": budget, "runtime": runtime,
                "elapsed_before_publication_seconds": time.perf_counter() - STARTED,
                "no_training_iteration_executed": True, "not_performance_comparison_or_accepted_algorithm": True,
                "new_optimization_count": 0}
            atomic_json(output / "RESULT.json", result)
            print(json.dumps({"status": result["status"], "checks": checks, "phases": phases,
                "native_load_seconds": [row["seconds"] for row in loads["calls"]], "API_scopes": trace}), flush=True)
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"type": type(error).__name__, "error": str(error),
            "phases": phases, "actual_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
