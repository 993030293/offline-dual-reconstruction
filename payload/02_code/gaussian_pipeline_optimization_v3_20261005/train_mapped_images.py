"""Bind exact mapped CPU pixels and clean-page release to the native training entry."""

import argparse
import copy
from contextlib import contextmanager
import inspect
import json
import linecache
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import train_camera_execution as native_entry
import checkpoint_adapter
from mapped_training_images import mapped_training_images
from mapped_image_linux_guard import LinuxRSSGuard


@contextmanager
def mapped_training_loop(store):
    import train as reference
    if store is None:
        yield {"samples": 0, "installed": False, "restored": True}
        return
    original = reference.training
    name = "_r131_release_private_image"
    if hasattr(reference, name):
        raise ValueError("The private image-release binding already exists")
    trace = {"samples": 0, "installed": True, "restored": False,
        "release_after_original_blocking_cuda_copy": True}

    def release(value):
        store.release_clean_pages(value)
        trace["samples"] += 1

    source = inspect.getsource(original)
    anchor = "        gt_image = viewpoint_cam.original_image.cuda()\n"
    if source.count(anchor) != 1:
        raise ValueError("The native blocking ground-truth copy boundary changed")
    source = source.replace(anchor, anchor+"        _r131_release_private_image(viewpoint_cam.original_image)\n")
    filename = "<r131_native_training_private_image_release>"
    prior_cache = linecache.cache.get(filename)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    setattr(reference, name, release)
    namespace = dict(original.__globals__)
    namespace[name] = release
    try:
        exec(compile(source, filename, "exec"), namespace)
        reference.training = namespace["training"]
        yield trace
    finally:
        reference.training = original
        delattr(reference, name)
        if prior_cache is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = prior_cache
        trace["restored"] = reference.training is original and not hasattr(reference, name)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-mode", choices=("baseline", "mapped"), required=True)
    parser.add_argument("--mapping-cache", type=Path)
    args, remaining = parser.parse_known_args()
    if remaining.count("--output") != 1 or remaining.count("--iterations") != 1:
        raise ValueError("The mapped training entry requires an explicit native output and budget")
    output = Path(remaining[remaining.index("--output")+1])
    iterations = int(remaining[remaining.index("--iterations")+1])
    output.resolve().relative_to(OUT.resolve())
    if iterations not in (1100, 12000):
        raise ValueError("The mapped campaign keeps its original bounded smoke or full training budget")
    if iterations == 12000 and (remaining.count("--checkpoint-every") != 1 or remaining[remaining.index("--checkpoint-every")+1] != "4000"):
        raise ValueError("Full mapped training retains complete state at the original 4000-step boundaries")
    enabled = args.mapping_mode == "mapped"
    if enabled:
        if args.mapping_cache is None:
            raise ValueError("The candidate requires its explicit private verified cache")
        args.mapping_cache.resolve().relative_to(OUT.resolve())
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard_output = output.parent/(output.name+"_mapping_guard")
    guard_output.mkdir(exist_ok=False)
    with LinuxRSSGuard(guard_output, atomic_json) as guard:
        paths = [Path(__file__), Path(__file__).with_name("mapped_training_images.py"),
            Path(__file__).with_name("mapped_image_linux_guard.py"), V2/"train_camera_execution.py",
            V2/"camera_execution.py", V2/"prefetch_camera_images.py", V1/"checkpoint_adapter.py"]
        sources = [data_identity(path) for path in paths]
        snapshots = output.parent/(output.name+"_mapping_sources")
        snapshots.mkdir(exist_ok=False)
        for index, row in enumerate(sources):
            copied = snapshots/(f"{index:03d}_"+Path(row["path"]).name)
            shutil.copyfile(row["path"], copied)
            if data_identity(copied)["sha256"] != row["sha256"]:
                raise ValueError("A complete mapped training source copy changed")
        binding = {"schema_version": 1, "enabled": enabled, "source_sha256": [row["sha256"] for row in sources],
            "native_CPU_HWC_float32_bytes_and_CHW_view": True, "full_admitted_image_count": 1016,
            "kernel_private_page_check": True, "maximum_cache_bytes": 8*1024**3,
            "loss_sampling_and_resolution_unchanged": True}
        original_camera, original_recovery, previous = native_entry.camera_execution, checkpoint_adapter.RecoveryState, sys.argv
        receipts = []

        class BoundMappedRecovery(original_recovery):
            def __init__(self, output, identity, interval, resume=None, crash_after=None):
                identity = dict(identity)
                adapters = dict(identity.get("external_adapters", {}))
                if "mapped_training_images" in adapters:
                    raise ValueError("The native checkpoint already has a mapped-image identity")
                adapters["mapped_training_images"] = binding
                identity["external_adapters"] = adapters
                super().__init__(output, identity, interval, resume, crash_after)

            def step(self, model, iteration, scene, stack, indices, ema_loss, ema_depth, options):
                saving = options
                if iteration == options.iterations and iteration % self.interval == 0:
                    saving = copy.copy(options)
                    saving.iterations += 1
                return super().step(model, iteration, scene, stack, indices, ema_loss, ema_depth, saving)

        @contextmanager
        def camera_context(**flags):
            with original_camera(**flags) as execution, mapped_training_images(args.mapping_cache, data_identity, enabled) as store:
                with mapped_training_loop(store) as loop:
                    yield execution
                receipts.append({"mapped": None if store is None else store.receipt(), "loop": dict(loop)})

        try:
            native_entry.camera_execution, checkpoint_adapter.RecoveryState = camera_context, BoundMappedRecovery
            sys.argv = [str(Path(native_entry.__file__)), *remaining]
            native_entry.main()
        finally:
            native_entry.camera_execution, checkpoint_adapter.RecoveryState, sys.argv = original_camera, original_recovery, previous
        state = json.loads((output/"STATE.json").read_text())
        if len(receipts) != 1:
            raise ValueError("The original complete camera entry did not close exactly once")
        record = receipts[0]
        count = state["executed_iterations"]
        checks = {
            "native_training_completed_full_declared_budget": state["status"] == "completed" and state["iteration"] == iterations,
            "all1016_native_inputs": state["image_count"] == 1016,
            "complete_final_native_checkpoint": iterations != 12000 or (output/"checkpoint_012000.pth").is_file(),
            "release_hook_closed": record["loop"]["restored"],
            "one_complete_original_copy_per_executed_iteration": record["loop"]["samples"] == (count if enabled else 0),
            "every_original_native_mapping_preserved": not enabled or record["mapped"]["prepared"] == record["mapped"]["mapping_count"] == record["mapped"]["unique_payloads"] == 1016,
            "no_unregistered_training_image_modification": not enabled or record["mapped"]["dirty_release_skipped"] == 0,
            "complete_source_identity_preserved": sources == [data_identity(path) for path in paths],
            "native_camera_and_recovery_hooks_restored": native_entry.camera_execution is original_camera and checkpoint_adapter.RecoveryState is original_recovery and sys.argv is previous,
        }
        guard.sample()
        result = {"status": "native_complete_training_pending_quality_and_cost_acceptance" if all(checks.values()) else "failed_verification",
            "checks": checks, "mode": args.mapping_mode, "state": state, "mapping_binding": binding,
            "sources": sources, "receipts": receipts, "Linux_resource": dict(guard.row),
            "complete_adapter_seconds": time.perf_counter()-started, "new_optimization_count": 0,
            "three_seed_quality_and_complete_recovery": "pending"}
        atomic_json(output/"MAPPED_TRAINING_RESULT.json", result)
        print(json.dumps({"status": result["status"], "mode": args.mapping_mode, "complete_adapter_seconds": result["complete_adapter_seconds"], "checks": checks}))
        if not all(checks.values()):
            raise AssertionError("A native mapped-training contract failed")


if __name__ == "__main__":
    main()
