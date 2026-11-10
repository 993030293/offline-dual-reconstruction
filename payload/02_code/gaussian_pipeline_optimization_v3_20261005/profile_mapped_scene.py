"""Load and fingerprint every admitted camera with the existing complete RGB path."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import resource
import shutil
import sys
import threading
import time
import weakref

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import camera_loader_adapter
from camera_execution import camera_execution
from prefetch_camera_images import prefetch_camera_images
from mapped_training_images import mapped_training_images
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--mode", choices=("baseline", "mapped"), required=True)
    args = parser.parse_args()
    out_root = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"
    args.output.resolve().relative_to(out_root.resolve())
    if args.mode == "mapped":
        if args.cache is None:
            raise ValueError("The mapped profile requires an explicit private cache")
        args.cache.resolve().relative_to(out_root.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = ROOT/"04_outputs/box2_3dgs_display_v1_20260930"
    dataset = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
    with LinuxRSSGuard(args.output, atomic_json) as guard:
        metadata = activate_reference(runtime)
        import torch
        from arguments import ModelParams
        import scene as scene_module
        import scene.cameras as camera_module
        from scene import Scene, GaussianModel
        from scene.colmap_loader import read_extrinsics_binary
        reference = Path(metadata["reference_source"])
        source_paths = [Path(__file__), Path(__file__).with_name("mapped_training_images.py"),
            Path(__file__).with_name("mapped_image_linux_guard.py"),
            Path(__file__).with_name("mapped_image_node.py"),
            Path(__file__).with_name("streamed_ply_node_002.py"),
            V1/"runtime.py", V1/"full_intrinsics.py"]
        source_paths += [V2/name for name in ("camera_execution.py", "prefetch_camera_images.py")]
        source_paths += [reference/name for name in ("scene/__init__.py", "scene/cameras.py",
            "scene/gaussian_model.py", "scene/dataset_readers.py", "scene/colmap_loader.py",
            "utils/camera_utils.py", "utils/general_utils.py", "arguments/__init__.py")]
        sources = [data_identity(path) for path in source_paths]
        snapshots = args.output/"source_snapshot"
        snapshots.mkdir()
        for index, row in enumerate(sources):
            copied = snapshots/(f"{index:03d}_"+Path(row["path"]).name)
            shutil.copyfile(row["path"], copied)
            if data_identity(copied)["sha256"] != row["sha256"]:
                raise ValueError("An exact Scene source copy changed")
        inputs = [data_identity(path) for path in sorted(dataset.rglob("*")) if path.is_file()]
        bindings = read_extrinsics_binary(dataset/"sparse/0/images.bin")
        names = {row.name for row in bindings.values()}
        actual_names = {str(path.relative_to(dataset/"images")) for path in (dataset/"images").rglob("*") if path.is_file()}
        if len(bindings) != 1016 or len(names) != 1016 or names != actual_names:
            raise ValueError("The registered Scene dataset lost its complete admitted image set")
        native_parser = argparse.ArgumentParser()
        group = ModelParams(native_parser)
        native = native_parser.parse_args([])
        config = json.loads((runtime/"MAIN_CONFIG.json").read_text())
        for key, value in config["parameters"].items():
            setattr(native, key, value)
        native.source_path = str(dataset.resolve())
        native.model_path = str((args.output/"scene").resolve())
        Path(native.model_path).mkdir()
        if native.data_device != "cpu" or native.resolution != 2:
            raise ValueError("The frozen Scene CPU camera or full-image resolution changed")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        if free < 2*1024**3:
            raise MemoryError("The complete Scene has insufficient CUDA headroom")
        budget = min(12*1024**3, free-2*1024**3)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        def tensor(value):
            if value is None:
                return None
            return {"shape": list(value.shape), "stride": list(value.stride()),
                "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                "dtype": str(value.dtype), "device": str(value.device),
                "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}

        def rng():
            return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                "torch_CPU": tensor(torch.get_rng_state()),
                "torch_CUDA": [tensor(value) for value in torch.cuda.get_rng_state_all()]}

        before_rng = rng()
        saved_init, saved_list, saved_prepare = camera_module.Camera.__init__, scene_module.cameraList_from_camInfos, camera_module.PILtoTorch
        mapping = mapped_training_images(args.cache, data_identity) if args.mode == "mapped" else nullcontext(None)
        load_started = time.perf_counter()
        with (camera_loader_adapter(dataset) as manifest,
                camera_execution(opaque_alpha=True, clampfree=True, calibrated_projection=True) as execution,
                mapping as store, prefetch_camera_images(4, 8) as prefetch):
            scene = Scene(group.extract(native), GaussianModel(3), shuffle=True)
            torch.cuda.synchronize()
            load_seconds = time.perf_counter()-load_started
            after_rng = rng()
            guard.sample()
            after_load_rss = psutil.Process().memory_info().rss
            cameras = scene.getTrainCameras()
            if len(cameras) != 1016 or len(manifest) != 1016:
                raise ValueError("The complete Scene changed its admitted camera count")
            aliases, groups, rows = [], {}, []

            def alias(value):
                if value is None:
                    return None
                key = str(value.device), value.untyped_storage().data_ptr()
                if key not in groups:
                    groups[key] = len(groups)
                return groups[key]

            field_names = ("original_image", "alpha_mask", "world_view_transform",
                "projection_matrix", "full_proj_transform", "camera_center")
            pixel_started = time.perf_counter()
            for camera in cameras:
                rows.append({"name": camera.image_name, "uid": camera.uid, "colmap_id": camera.colmap_id,
                    "intrinsics": camera.calibrated_intrinsics,
                    **{name: tensor(getattr(camera, name)) for name in field_names}})
                aliases.append({name: alias(getattr(camera, name)) for name in field_names})
                if store is not None:
                    store.release_clean_pages(camera.original_image)
                guard.sample()
            pixel_seconds = time.perf_counter()-pixel_started
            initial = {name: tensor(getattr(scene.gaussians, name)) for name in (
                "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation",
                "_exposure", "max_radii2D", "xyz_gradient_accum", "denom")}
            auxiliary = {name: getattr(scene.gaussians, name) for name in (
                "active_sh_degree", "max_sh_degree", "optimizer_type", "percent_dense",
                "spatial_lr_scale", "exposure_mapping", "pretrained_exposures")}
            scientific = {"rows": rows, "initial_gaussians": initial, "auxiliary": auxiliary,
                "camera_alias_groups": aliases, "camera_manifest": manifest, "camera_extent": scene.cameras_extent,
                "before_rng": before_rng, "after_rng": after_rng, "after_pixel_rng": rng(),
                "native_parameters": {key: value for key, value in vars(native).items() if key != "model_path"},
                "train_scale_keys": list(scene.train_cameras), "test_scale_keys": list(scene.test_cameras),
                "test_names": {str(key): [view.image_name for view in views] for key, views in scene.test_cameras.items()},
                "loaded_iteration": scene.loaded_iter, "optimizer_is_none": scene.gaussians.optimizer is None}
            mapped_receipt = None if store is None else store.receipt()
            after_pixels_rss = psutil.Process().memory_info().rss
            scene_ref = weakref.ref(scene)
            del camera, cameras, scene
        gc.collect()
        torch.cuda.synchronize()
        cuda_peak = torch.cuda.max_memory_reserved()
        torch.cuda.empty_cache()
        checks = {
            "all1016_full_pixel_and_state_rows": len(rows) == 1016,
            "original_cpu_float32_and_nontransparent_scope": all(row["original_image"]["device"] == "cpu" and row["original_image"]["dtype"] == "torch.float32" and row["alpha_mask"] is None for row in rows),
            "all_original_prefetch_inputs_and_bound": prefetch["prepared"] == prefetch["consumed"] == 1016 and prefetch["maximum_pending"] <= 8,
            "all_native_camera_calls": execution["native_camera_calls"] == 1016,
            "original_native_functions_restored": camera_module.Camera.__init__ is saved_init and scene_module.cameraList_from_camInfos is saved_list and camera_module.PILtoTorch is saved_prepare,
            "no_native_scene_retained": scene_ref() is None,
            "owned_prefetch_threads_closed": not any(t.name.startswith("camera-decode") for t in threading.enumerate()),
            "full_input_and_source_identity_preserved": inputs == [data_identity(row["path"]) for row in inputs] and sources == [data_identity(path) for path in source_paths],
            "original_CUDA_allocation_bound": cuda_peak <= budget,
            "exact_complete_mapped_preparation_and_clean_release": store is None or mapped_receipt["prepared"] == mapped_receipt["mapping_count"] == mapped_receipt["unique_payloads"] == mapped_receipt["clean_page_release_calls"] == 1016,
            "no_unregistered_dirty_mapping": store is None or mapped_receipt["dirty_release_skipped"] == 0,
        }
        guard.sample()
        result = {"status": "complete_scene_pending_pair_cost_quality_acceptance" if all(checks.values()) else "failed_verification",
            "mode": args.mode, "checks": checks, "runtime": metadata, "runtime_config": config,
            "scientific": scientific, "sources": sources, "inputs": inputs,
            "generated_assets": {name: data_identity(args.output/"scene"/name) for name in ("input.ply", "cameras.json")},
            "Scene_load_seconds": load_seconds, "all_pixel_and_state_check_seconds": pixel_seconds,
            "RSS_after_load_bytes": after_load_rss, "RSS_after_all_pixels_bytes": after_pixels_rss,
            "full_process_tree_peak_RSS_bytes": guard.row["peak_tree_RSS_bytes"],
            "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "peak_CUDA_reserved_bytes": cuda_peak, "CUDA_budget_bytes": budget,
            "CUDA_start_total": total, "mapped": mapped_receipt, "prefetch": prefetch,
            "camera_execution": execution, "cache_or_system_memory_claim": "file page cache not a whole-host memory saving",
            "complete_profile_seconds": time.perf_counter()-started,
            "three_formal_pairs_training_quality_and_recovery": "not_run", "new_optimization_count": 0}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "mode", "Scene_load_seconds", "full_process_tree_peak_RSS_bytes", "complete_profile_seconds", "checks")}))
        if not all(checks.values()):
            raise ValueError("A complete native Scene contract failed")


if __name__ == "__main__":
    main()
