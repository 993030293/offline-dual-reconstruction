"""Verify real native distance inputs, complete state, and safe fallback routes."""

import argparse
import gc
import hashlib
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from check_native_feature_zero import state
from check_zero_transfer_contracts import prefix
from native_coordinate_reuse import native_coordinate_reuse
from profile_coordinate_scene import CoordinateSceneGuard, OUT


def array_id(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape), "strides": list(value.strides),
            "bytes": hashlib.sha256(value.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with CoordinateSceneGuard(args.output, atomic_json):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        import scene.gaussian_model as gaussian
        from utils.graphics_utils import BasicPointCloud
        from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary
        torch.cuda.set_device(0)
        sources = [Path(__file__), *[CODE / name for name in (
            "native_coordinate_reuse.py", "profile_coordinate_scene.py", "native_transfer_meter.py")],
            Path(metadata["reference_source"]) / "scene/gaussian_model.py"]
        identities = [data_identity(path) for path in sources]
        archive = args.output / "source_snapshot"
        archive.mkdir()
        for number, path in enumerate(sources):
            shutil.copyfile(path, archive / (f"{number:03d}_" + path.name))
        rng = np.random.default_rng(169001)
        rows = []

        def pair(name, cloud, degree, infos, expected, fault=False, default64=False):
            before_cloud = [array_id(value) for value in cloud]
            sides = {}
            saved_dist, saved_dtype = gaussian.distCUDA2, torch.get_default_dtype()
            try:
                if default64:
                    torch.set_default_dtype(torch.float64)
                for enabled in (False, True):
                    torch.manual_seed(169001)
                    np.random.seed(169001)
                    random.seed(169001)
                    observations = []
                    def distance(value):
                        before = state_tensor(value)
                        if fault:
                            observations.append({"before": before, "after": before, "native_called": False})
                            raise OSError("R169 registered native distance failure")
                        result = saved_dist(value)
                        observations.append({"before": before, "after": state_tensor(value), "native_called": True})
                        return result
                    gaussian.distCUDA2 = distance
                    leaf = args.output / (name + "_" + str(int(enabled)))
                    leaf.mkdir()
                    model = gaussian.GaussianModel(degree)
                    error = None
                    with native_coordinate_reuse(leaf, enabled) as trace:
                        try:
                            model.create_from_pcd(cloud, infos, 1.25)
                        except (OSError, RuntimeError, ValueError, TypeError) as actual:
                            error = {"type": type(actual).__name__, "message": str(actual)}
                        value = prefix(model) if error else state(model, torch)
                    sides[str(int(enabled))] = {"state": value, "error": error, "trace": trace,
                                                "distance": observations}
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                gaussian.distCUDA2 = saved_dist
                torch.set_default_dtype(saved_dtype)
            a, b = sides["0"], sides["1"]
            checks = {"full_state_grad_RNG_and_errors_exact": a["state"] == b["state"] and a["error"] == b["error"],
                      "original_inputs_unchanged": before_cloud == [array_id(value) for value in cloud],
                      "native_distance_input_full_bits_stride_and_post_state_exact": a["distance"] == b["distance"]
                        and all(row["before"] == row["after"] for row in a["distance"] + b["distance"]),
                      "native_transfer_meter_and_all_bindings_closed": all(row["trace"]["native_functions_restored"]
                        and row["trace"]["calls"][0]["native_API_matches_all_tensor_uploads"] for row in sides.values()),
                      "preregistered_route_exact": b["trace"]["calls"][0]["route"] == expected}
            if expected == "coordinate_reuse" and not a["error"]:
                checks["only_duplicate_nonzero_point_upload_removed"] = (len(a["trace"]["calls"][0]["uploads"]) == 3
                    and len(b["trace"]["calls"][0]["uploads"]) == 2
                    and a["trace"]["calls"][0]["uploads"][:2] == b["trace"]["calls"][0]["uploads"]
                    and a["trace"]["actual_native_HtoD_bytes"] - b["trace"]["actual_native_HtoD_bytes"] == cloud.points.size * 4
                    and a["trace"]["calls"][0]["parameter_storage_alias_groups"] == b["trace"]["calls"][0]["parameter_storage_alias_groups"])
            checks["registered_error_or_normal_success"] = bool(a["error"]) if fault or name == "bad_colors" else a["error"] is None
            row = {"name": name, "checks": checks, "sides": sides}
            rows.append(row)
            atomic_json(args.output / "PARTIAL.json", {"rows": rows, "new_optimization_count": 0})
            if not all(checks.values()):
                raise AssertionError("The real native operand contract changed: " + name + repr(checks))

        def state_tensor(value):
            return {"dtype": str(value.dtype), "device": str(value.device), "shape": list(value.shape),
                    "stride": list(value.stride()), "grad": value.requires_grad,
                    "bytes": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}

        infos = [SimpleNamespace(image_name="irregular_view")]
        for number, (count, degree, layout) in enumerate(((31, 0, "C"), (129, 1, "F"), (257, 3, "C"), (65, 3, "strided"))):
            points = rng.normal(size=(count, 3)).astype(np.float32)
            colors = rng.uniform(size=(count, 3)).astype(np.float64)
            if layout == "F":
                points = np.asfortranarray(points)
            if layout == "strided":
                points = np.column_stack((points, points))[:, ::2]
            cloud = BasicPointCloud(points, colors, np.zeros_like(points))
            pair("generic" + str(number), cloud, degree, infos, "direct_zero" if layout == "strided" else "coordinate_reuse")
        points = rng.normal(size=(31, 3)).astype(np.float32)
        colors = rng.uniform(size=(31, 3)).astype(np.float64)
        cloud = BasicPointCloud(points, colors, np.zeros_like(points))
        pair("float64_points", BasicPointCloud(points.astype(np.float64), colors, np.zeros_like(points)), 3, infos, "direct_zero")
        pair("bad_colors", BasicPointCloud(points, colors[:, :2], np.zeros_like(points)), 3, infos, "coordinate_reuse")
        pair("native_dist_failure", cloud, 3, infos, "coordinate_reuse", fault=True)
        pair("default_float64", cloud, 3, infos, "native", default64=True)
        pair("single_point_fallback_at_error", BasicPointCloud(points[:1], colors[:1], np.zeros_like(points[:1])),
             3, infos, "direct_zero", fault=True)
        nonfinite = points.copy()
        nonfinite[0, 0] = np.nan
        pair("nonfinite_fallback_at_error", BasicPointCloud(nonfinite, colors, np.zeros_like(points)),
             3, infos, "direct_zero", fault=True)
        public = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
        public_paths = [public / name for name in ("points3D.bin", "images.bin", "cameras.bin")]
        public_ids = [data_identity(path) for path in public_paths]
        xyz, rgb, _ = read_points3D_binary(public / "points3D.bin")
        cameras = read_extrinsics_binary(public / "images.bin")
        if xyz.shape != (133020, 3) or len(cameras) != 360:
            raise ValueError("The existing public component population changed")
        public_cloud = BasicPointCloud(xyz.astype(np.float32), rgb / 255., np.zeros_like(xyz))
        pair("public_float32_adapter", public_cloud, 3,
             [SimpleNamespace(image_name=value.name) for _, value in sorted(cameras.items())], "coordinate_reuse")
        pair("public_original_float64", BasicPointCloud(xyz, rgb / 255., np.zeros_like(xyz)), 3,
             [SimpleNamespace(image_name=value.name) for _, value in sorted(cameras.items())], "direct_zero")
        checks = {"all_twelve_native_cases": len(rows) == 12 and all(all(row["checks"].values()) for row in rows),
                  "actual_sources_unchanged": identities == [data_identity(path) for path in sources],
                  "public_inputs_unchanged": public_ids == [data_identity(path) for path in public_paths]}
        result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                  "rows": rows, "sources": identities, "public_inputs": public_ids, "runtime": metadata,
                  "extra_actual_native_HtoD_bytes": sum(side["trace"]["actual_native_HtoD_bytes"]
                    for row in rows for side in row["sides"].values()), "new_optimization_count": 0,
                  "public_is_float32_adapted_development_component_not_original_float64_or_GT": True}
        atomic_json(args.output / "RESULT.json", result)
        if not all(checks.values()):
            raise AssertionError("A native operand source check failed")


if __name__ == "__main__":
    main()
