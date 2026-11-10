"""Probe exact inverse error collection on every original camera transform."""

import argparse
import gc
import hashlib
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from profile_scene_transform_cost import BoundedSceneGuard

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with BoundedSceneGuard(args.output, atomic_json) as guard:
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
        from utils.graphics_utils import getWorld2View2
        dataset = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
        inputs = [data_identity(dataset / "sparse/0/images.bin")]
        poses = read_extrinsics_binary(Path(inputs[0]["path"]))
        if len(poses) != 1016 or torch.__version__ != "2.7.1+cu128":
            raise ValueError("The current native full-pose or experimental API version changed")
        sources = [Path(__file__), CODE / "inverse_overlap_probe_node.py", CODE / "run_inverse_overlap_probe.py",
            Path(metadata["reference_source"]) / "scene/colmap_loader.py",
            Path(metadata["reference_source"]) / "utils/graphics_utils.py", Path(torch.linalg.__file__)]
        identities = [data_identity(path) for path in sources]
        archive = args.output / "source_snapshot"
        archive.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
        frozen_poses = [(row.id, row.name, np.transpose(qvec2rotmat(row.qvec)), row.tvec.copy()) for row in poses.values()]
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise MemoryError("The original GPU headroom is unavailable")
        budget = min(12 * 1024**3, free - 2 * 1024**3)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        before_rng = [torch.get_rng_state().clone(), *[t.clone() for t in torch.cuda.get_rng_state_all()]]
        for _ in range(2):
            warm = torch.eye(4, device="cuda")
            warm.inverse()
            torch.linalg.inv_ex(warm, check_errors=False)
        del warm
        torch.cuda.synchronize()

        def tensor(value):
            return {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset(),
                "dtype": str(value.dtype), "device": str(value.device), "requires_grad": value.requires_grad,
                "storage_bytes": value.untyped_storage().nbytes(),
                "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}

        def run(mode):
            torch.cuda.synchronize()
            started = time.perf_counter()
            values, infos = [], []
            for _, _, rotation, translation in frozen_poses:
                view = torch.tensor(getWorld2View2(rotation, translation)).transpose(0, 1).cuda()
                if mode == "native":
                    inverse = view.inverse()
                else:
                    inverse, info = torch.linalg.inv_ex(view, check_errors=False)
                    infos.append(info)
                values.append((view, inverse[3, :3]))
            info_values = torch.stack(infos).cpu().tolist() if infos else []
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if mode == "collected" and info_values != [0] * 1016:
                raise RuntimeError("An original valid-pose inverse failed; this probe has no error adapter")
            groups, rows = {}, []
            for (image_id, name, _, _), (view, center) in zip(frozen_poses, values):
                aliases = []
                for value in (view, center):
                    pointer = value.untyped_storage().data_ptr()
                    if pointer not in groups:
                        groups[pointer] = len(groups)
                    aliases.append(groups[pointer])
                rows.append({"id": image_id, "name": name, "view": tensor(view), "center": tensor(center), "aliases": aliases})
            values.clear()
            infos.clear()
            gc.collect()
            torch.cuda.synchronize()
            guard.sample()
            return {"component_seconds": elapsed, "rows": rows, "info_values": info_values,
                "GPU_error_checks_collected_after_valid_full_component": mode == "collected"}

        pairs = []
        orders = [("native", "collected"), ("collected", "native"), ("native", "collected")]
        for index, order in enumerate(orders):
            rows = {mode: run(mode) for mode in order}
            if rows["native"]["rows"] != rows["collected"]["rows"]:
                raise AssertionError("A native full transform bit, layout, ownership or order changed")
            pairs.append({"pair": index, "order": list(order), "rows": rows})
            atomic_json(args.output / "STATE.json", {"status": "running_registered_full_transform_API_probe0", "pairs": pairs, "new_optimization_count": 0})
        after_rng = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
        medians = {mode: statistics.median(pair["rows"][mode]["component_seconds"] for pair in pairs) for mode in orders[0]}
        checks = {
            "all_original1016_poses_three_fixed_order_pairs_exact": len(pairs) == 3 and all(len(p["rows"]["native"]["rows"]) == 1016 and p["rows"]["native"]["rows"] == p["rows"]["collected"]["rows"] for p in pairs),
            "all_actual3048_inverse_infos_checked_zero": all(p["rows"]["collected"]["info_values"] == [0] * 1016 for p in pairs),
            "original_RNG_unchanged": all(torch.equal(a, b) for a, b in zip(before_rng, after_rng)),
            "original_input_and_actual_sources_unchanged": inputs == [data_identity(row["path"]) for row in inputs] and identities == [data_identity(path) for path in sources],
            "original_GPU_and_Linux_resource_bounds": torch.cuda.max_memory_reserved() <= budget and guard.row["peak_tree_RSS_bytes"] <= 16 * 1024**3,
        }
        result = {"status": "complete_native_inverse_API_probe0" if all(checks.values()) else "failed_verification", "checks": checks,
            "runtime": metadata, "torch": torch.__version__, "native_inv_ex_doc": torch.linalg.inv_ex.__doc__,
            "inputs": inputs, "sources": identities, "orders": [list(p) for p in orders], "pairs": pairs,
            "component_medians": medians, "raw_component_reduction": 1 - medians["collected"] / medians["native"],
            "native_host_world_build_and_original_cuda_copy_included": True,
            "end_CUDA_completion_and_all_info_readback_included": True,
            "full_scene_pixels_Gaussians_failure_prefix_recovery_generalization_not_verified": True,
            "experimental_API_requires_version_and_error_semantics_adapter_before_any_acceptance": True,
            "new_optimization_count": 0, "parent_count": 29, "remaining": 21}
        atomic_json(args.output / "RESULT.json", result)
        atomic_json(args.output / "STATE.json", result)
        if not all(checks.values()):
            raise AssertionError("The native full inverse probe lost an original transform invariant")
        print({"checks": checks, "medians": medians, "raw_component_reduction": result["raw_component_reduction"], "new": 0})


if __name__ == "__main__":
    main()
