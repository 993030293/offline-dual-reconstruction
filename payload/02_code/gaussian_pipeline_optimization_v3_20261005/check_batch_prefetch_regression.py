"""Exercise the actual calibrated and prefetched camera chain after late binding."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import threading

import numpy as np

import profile_scene_transform_cost as common
from batched_camera_execution import batched_camera_execution

sys.path.insert(0, str(common.ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"))
from camera_execution import camera_execution
from prefetch_camera_images import prefetch_camera_images
from full_intrinsics import camera_loader_adapter
from mapped_image_linux_guard import LinuxRSSGuard
from runtime import activate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(common.BASE.resolve())
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), common.CODE / "batched_camera_execution.py", common.CODE / "batched_camera_transforms.py",
               common.CODE / "profile_batched_camera_scene.py", common.CODE / "batch_prefetch_regression_node.py"]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    source_ids = [common.data_identity(p) for p in sources]
    with LinuxRSSGuard(output, common.atomic_json, maximum_seconds=60, maximum_bytes=2 * 1024**3):
        runtime = activate_reference(common.ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        import scene
        import scene.cameras as camera_module
        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
        from scene.dataset_readers import readColmapCameras
        from arguments import ModelParams
        dataset = common.ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
        extrinsics = read_extrinsics_binary(dataset / "sparse/0/images.bin")
        subset = {key: extrinsics[key] for key in list(extrinsics)[:3]}
        infos = readColmapCameras(subset, read_intrinsics_binary(dataset / "sparse/0/cameras.bin"), None, str(dataset / "images"), "", [])
        native_parser = argparse.ArgumentParser()
        group = ModelParams(native_parser)
        native = native_parser.parse_args([])
        config = json.loads((common.ROOT / "04_outputs/box2_3dgs_display_v1_20260930/MAIN_CONFIG.json").read_text())
        for key, value in config["parameters"].items():
            setattr(native, key, value)
        native.source_path = str(dataset)
        parameters = group.extract(native)
        torch.cuda.set_device(0)
        saved_init, saved_list, saved_pil = camera_module.Camera.__init__, scene.cameraList_from_camInfos, camera_module.PILtoTorch

        def tensor(value):
            if value is None:
                return None
            return {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset(),
                    "dtype": str(value.dtype), "device": str(value.device), "requires_grad": value.requires_grad,
                    "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}

        def rng():
            return {"numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    "torch_CPU": tensor(torch.get_rng_state()), "torch_CUDA": [tensor(v) for v in torch.cuda.get_rng_state_all()]}

        results, traces = {}, {}
        for mode in ("native", "batch"):
            private = output / mode
            private.mkdir()
            np.random.seed(0)
            torch.manual_seed(0)
            before_rng = rng()
            trace = {}
            preparation = batched_camera_execution(private, trace) if mode == "batch" else nullcontext(None)
            with preparation as binder:
                with (camera_loader_adapter(dataset) as manifest,
                      camera_execution(opaque_alpha=True, clampfree=True, calibrated_projection=True),
                      prefetch_camera_images(4, 8) as prefetched):
                    binding = binder() if mode == "batch" else nullcontext()
                    with binding:
                        cameras = scene.cameraList_from_camInfos(infos, 1.0, parameters, False, False)
                    torch.cuda.synchronize()
                    groups, aliases, rows = {}, [], []
                    fields = ("original_image", "alpha_mask", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")
                    for camera in cameras:
                        row_aliases = {}
                        for name in fields:
                            value = getattr(camera, name)
                            key = None if value is None else (str(value.device), value.untyped_storage().data_ptr())
                            if key is not None and key not in groups:
                                groups[key] = len(groups)
                            row_aliases[name] = None if key is None else groups[key]
                        aliases.append(row_aliases)
                        rows.append({"uid": camera.uid, "name": camera.image_name, "colmap_id": camera.colmap_id,
                                     "intrinsics": camera.calibrated_intrinsics, **{name: tensor(getattr(camera, name)) for name in fields}})
                    results[mode] = {"rows": rows, "aliases": aliases, "manifest": manifest,
                                     "before_rng": before_rng, "after_rng": rng()}
                    del camera, cameras
            traces[mode] = {"batch": trace, "prefetch": prefetched}
            gc.collect()
            torch.cuda.synchronize()
        checks = {"three_actual_full_pixel_matrix_intrinsic_alias_RNG_states_exact": results["native"] == results["batch"] and len(results["batch"]["rows"]) == 3,
                  "actual_prefetch_late_binding_prepared_consumed3_not_native_fallback": traces["batch"]["batch"]["list_bound_after_prefetch"]
                    and traces["batch"]["batch"]["batch_prepared"] == traces["batch"]["batch"]["batch_consumed"] == 3
                    and traces["batch"]["batch"]["native_calls"] == 0,
                  "both_real_prefetch3_consumed_and_bounded": all(row["prefetch"]["prepared"] == row["prefetch"]["consumed"] == 3
                    and row["prefetch"]["maximum_pending"] <= 8 for row in traces.values()),
                  "all_native_callbacks_linecache_and_threads_closed": camera_module.Camera.__init__ is saved_init
                    and scene.cameraList_from_camInfos is saved_list and camera_module.PILtoTorch is saved_pil
                    and traces["batch"]["batch"]["original_functions_and_linecache_restored"]
                    and not any(t.name.startswith("camera-decode") for t in threading.enumerate()),
                  "all_current_sources_unchanged": source_ids == [common.data_identity(p) for p in sources]}
        common.atomic_json(output / "RESULT.json", {"status": "actual_prefetch_three_camera_regression_passed0" if all(checks.values()) else "failed_actual_prefetch_regression0",
            "checks": checks, "results": results, "traces": traces, "runtime": runtime, "sources": source_ids,
            "full1016_three_pairs_normal_errors_recovery_quality_or_acceptance": "not_run", "new_optimization_count": 0})
        print(json.dumps({"checks": checks, "actual_batch_prepared": traces["batch"]["batch"]["batch_prepared"], "new_optimization_count": 0}))
        if not all(checks.values()):
            raise AssertionError("The actual prefetched camera binding regression failed")


if __name__ == "__main__":
    main()
