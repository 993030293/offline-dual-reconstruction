"""Exercise the native meter, original errors, fallbacks and public component."""

import ctypes
import gc
import hashlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from native_feature_zero_v2 import native_feature_zero
from native_transfer_meter import NativeTransferMeter
from check_native_feature_zero import state

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/g3"


def prefix(model):
    result = {}
    for key, value in vars(model).items():
        if hasattr(value, "untyped_storage"):
            result[key] = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                           "device": str(value.device), "grad": value.requires_grad,
                           "bytes": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        elif value is None or type(value) in (str, bool, int, float):
            result[key] = value
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=180, maximum_bytes=8 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        import scene.gaussian_model as gaussian
        from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary
        from utils.graphics_utils import BasicPointCloud
        torch.cuda.set_device(0)
        torch.manual_seed(157003)
        meter = NativeTransferMeter()
        checks, receipts, budget = {}, [], 0
        for number, (count, dtype) in enumerate(((17, torch.uint8), (21, torch.float32), (11, torch.float64))):
            value = torch.arange(count, dtype=dtype)
            meter.begin()
            actual = value.cuda()
            torch.cuda.synchronize()
            events = meter.end()
            h2d = sum(row["bytes"] for row in events if row["kind"] == 1 and row["result"] == 0)
            checks["actual_native_upload_dtype_" + str(number)] = (len(events) == 1 and h2d == value.numel() * value.element_size()
                                                                    and torch.equal(actual.cpu(), value))
            budget += h2d
            receipts.append({"test": "native_dtype", "events": events})
        meter.begin()
        direct = torch.zeros(17, device="cuda")
        unchanged = direct.cuda()
        torch.cuda.synchronize()
        events = meter.end()
        checks["GPU_zero_and_identity_cuda_no_HtoD"] = not events and unchanged is direct
        device_source = torch.arange(17, dtype=torch.uint8, device="cuda")
        device_destination = torch.empty_like(device_source)
        meter.begin()
        device_destination.copy_(device_source)
        host_destination = device_destination.cpu()
        torch.cuda.synchronize()
        events = meter.end()
        checks["device_and_return_copies_excluded_from_HtoD"] = (torch.equal(host_destination, torch.arange(17, dtype=torch.uint8))
            and any(row["kind"] == 2 for row in events) and all(row["kind"] in (2, 3) and row["result"] == 0 for row in events))
        receipts.append({"test": "device_and_return", "events": events})
        meter.begin()
        try:
            meter.begin()
        except RuntimeError:
            nested_rejected = True
        else:
            nested_rejected = False
        events = meter.end()
        checks["nested_scope_rejected_and_closed"] = nested_rejected and not events and not meter.active
        library = meter.library
        native = ctypes.CDLL(meter.backend)
        for handle in (library, native):
            handle.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            handle.cudaMemcpy.restype = ctypes.c_int
        expected_error = native.cudaMemcpy(None, None, 1, 1)
        meter.begin()
        actual_error = library.cudaMemcpy(None, None, 1, 1)
        events = meter.end()
        checks["native_error_return_preserved_and_recorded"] = expected_error == actual_error != 0 and len(events) == 1 and events[0]["result"] == actual_error
        native.cudaGetLastError()
        receipts.append({"test": "native_error", "events": events})
        meter.begin()
        for _ in range(65):
            if library.cudaMemcpy(None, None, 0, 1) != 0:
                raise RuntimeError("The original zero-byte native copy failed")
        try:
            meter.end()
        except RuntimeError as error:
            checks["native_capacity64_rejects65_zero_byte_calls"] = "overflowed" in str(error) and not meter.active
        else:
            checks["native_capacity64_rejects65_zero_byte_calls"] = False
        meter.begin()
        events = meter.end()
        checks["scope_reusable_after_rejected_capacity"] = not events

        generator = np.random.default_rng(157004)
        points = generator.normal(size=(31, 3)).astype(np.float32)
        colors = generator.uniform(size=(31, 3)).astype(np.float32)
        cloud = BasicPointCloud(points, colors, np.zeros_like(points))
        before_cloud = [hashlib.sha256(value.tobytes()).hexdigest() for value in cloud]
        cases = []
        for kind in ("default_float64", "bad_colors", "native_dist_failure"):
            sides = {}
            saved_dtype, saved_dist = torch.get_default_dtype(), gaussian.distCUDA2
            if kind == "default_float64":
                torch.set_default_dtype(torch.float64)
            if kind == "native_dist_failure":
                def fail_dist(value):
                    raise OSError("R157 registered original distCUDA2 failure")
                gaussian.distCUDA2 = fail_dist
            try:
                for mode in ("native", "direct"):
                    leaf = OUT / (kind + "_" + mode)
                    leaf.mkdir()
                    value = BasicPointCloud(points, colors[:, :2], np.zeros_like(points)) if kind == "bad_colors" else cloud
                    model = gaussian.GaussianModel(3)
                    error = None
                    with native_feature_zero(leaf, mode == "direct") as trace:
                        try:
                            model.create_from_pcd(value, [SimpleNamespace(image_name="generic")], 1.25)
                        except (OSError, RuntimeError) as actual:
                            error = {"type": type(actual).__name__, "message": str(actual)}
                        scientific = state(model, torch) if error is None else prefix(model)
                    budget += trace["actual_native_HtoD_bytes"]
                    sides[mode] = {"state": scientific, "error": error, "trace": trace}
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                torch.set_default_dtype(saved_dtype)
                gaussian.distCUDA2 = saved_dist
            valid = sides["native"]["state"] == sides["direct"]["state"] and sides["native"]["error"] == sides["direct"]["error"]
            if kind == "default_float64":
                valid = valid and sides["direct"]["trace"]["calls"][0]["route"] == "native" and sides["direct"]["error"] is None
            else:
                valid = valid and sides["direct"]["error"] is not None
            checks[kind + "_original_state_error_and_restoration"] = valid and all(side["trace"]["native_functions_restored"] for side in sides.values())
            cases.append({"kind": kind, "sides": sides})
            atomic_json(OUT / "PARTIAL.json", {"checks": checks, "cases": cases, "extra_actual_native_HtoD_bytes": budget})

        public = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
        public_paths = [public / name for name in ("points3D.bin", "images.bin", "cameras.bin")]
        public_ids = [data_identity(path) for path in public_paths]
        xyz, rgb, errors = read_points3D_binary(public / "points3D.bin")
        cameras = read_extrinsics_binary(public / "images.bin")
        if xyz.shape != (133020, 3) or len(cameras) != 360:
            raise ValueError("The actual public component denominator changed")
        public_cloud = BasicPointCloud(xyz, rgb / 255., np.zeros_like(xyz))
        infos = [SimpleNamespace(image_name=value.name) for _, value in sorted(cameras.items())]
        public_sides = {}
        for mode in ("native", "direct"):
            leaf = OUT / ("public_" + mode)
            leaf.mkdir()
            with native_feature_zero(leaf, mode == "direct") as trace:
                model = gaussian.GaussianModel(3)
                model.create_from_pcd(public_cloud, infos, 1.25)
                value = state(model, torch)
                del model
            budget += trace["actual_native_HtoD_bytes"]
            public_sides[mode] = {"state": value, "trace": trace}
            gc.collect()
            torch.cuda.empty_cache()
        checks["all_public133020_points360_names_component_exact"] = (public_sides["native"]["state"] == public_sides["direct"]["state"]
            and public_sides["native"]["trace"]["calls"][0]["parameter_storage_alias_groups"] == public_sides["direct"]["trace"]["calls"][0]["parameter_storage_alias_groups"])
        checks["all_public_native_API_payloads_verified"] = all(row["trace"]["calls"][0]["native_API_matches_all_tensor_uploads"] and row["trace"]["native_functions_restored"] for row in public_sides.values())
        checks["public_source_files_unchanged"] = public_ids == [data_identity(path) for path in public_paths]
        checks["original_generic_input_array_bytes_unchanged"] = before_cloud == [hashlib.sha256(value.tobytes()).hexdigest() for value in cloud]
        paths = [Path(__file__), CODE / "native_feature_zero.py", CODE / "native_feature_zero_v2.py",
                 CODE / "native_transfer_meter.py", CODE / "native_cuda_transfer_counter.c"]
        sources = [data_identity(path) for path in paths]
        copies = OUT / "source_snapshot"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
        atomic_json(OUT / "RESULT.json", {"status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "meter_receipts": receipts, "cases": cases, "public": public_sides,
            "public_inputs": public_ids, "sources": sources, "runtime": metadata, "native_backend": meter.backend,
            "extra_actual_native_HtoD_bytes": budget, "new_optimization_count": 0,
            "public_is_development_component_not_full_Scene_GT_or_unknown_object_training": True,
            "empty_nonfinite_or_default_GPU_native_errors_not_validated": True})
        if not all(checks.values()):
            raise AssertionError("A native meter or original model contract failed")


if __name__ == "__main__":
    main()
