"""Compare actual and generic camera matrices before any full consumer mutation."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from batched_camera_transforms import prepare_camera_transforms

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001"


def structure(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype), "device": str(tensor.device), "requires_grad": tensor.requires_grad}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE.resolve())
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CODE / "batched_camera_transforms.py", CODE / "batched_camera_transform_probe_node.py"]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshots / path.name)
    source_ids = [data_identity(path) for path in sources]
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=60, maximum_bytes=2 * 1024**3):
        runtime = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
        from utils.graphics_utils import getWorld2View2
        dataset = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
        input_id = data_identity(dataset / "sparse/0/images.bin")
        actual = read_extrinsics_binary(input_id["path"])
        if len(actual) != 1016:
            raise ValueError("The exact current camera input count changed")
        torch.cuda.set_device(0)
        torch.cuda.synchronize()

        def bits(tensor):
            return tensor.detach().contiguous().view(torch.int32)

        def rng():
            return {"numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    "torch_CPU": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    "torch_CUDA": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in torch.cuda.get_rng_state_all()]}

        before_rng = rng()
        matrices = [getWorld2View2(np.transpose(qvec2rotmat(row.qvec)), np.array(row.tvec)) for row in actual.values()]
        generic = []
        generator = np.random.RandomState(20261007)
        for index in range(33):
            rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            translation = generator.normal(size=3) * (1., .01, 100.)[index % 3]
            offset = np.array([.2, -.1, .3]) if index % 2 else np.zeros(3)
            generic.append(getWorld2View2(rotation, translation, offset, (.5, 1., 2.)[index % 3]))
        groups = [("actual1016", matrices), *[(f"generic{count}", generic[:count]) for count in (1, 2, 7, 33)]]
        rows, differences = [], []
        for label, hosts in groups:
            native = []
            for host in hosts:
                view = torch.tensor(host).transpose(0, 1).cuda()
                native.append((view, view.inverse()[3, :3]))
            batch = prepare_camera_transforms(hosts)
            view_pointers, center_pointers = [], []
            for index, ((old_view, old_center), (new_view, new_center)) in enumerate(zip(native, batch)):
                checks = {"world_view_bits": torch.equal(bits(old_view), bits(new_view)),
                          "camera_center_bits": torch.equal(bits(old_center), bits(new_center)),
                          "world_view_structure": structure(old_view) == structure(new_view),
                          "camera_center_structure": structure(old_center) == structure(new_center),
                          "center_base_layout": structure(old_center._base) == structure(new_center._base),
                          "view_has_no_base": old_view._base is None and new_view._base is None}
                view_pointers.append(new_view.untyped_storage().data_ptr())
                center_pointers.append(new_center.untyped_storage().data_ptr())
                if not all(checks.values()):
                    differences.append({"group": label, "index": index, "checks": checks,
                        "native_center": old_center.cpu().tolist(), "batch_center": new_center.cpu().tolist(),
                        "native_center_bits": bits(old_center).cpu().tolist(), "batch_center_bits": bits(new_center).cpu().tolist()})
            rows.append({"group": label, "count": len(hosts), "different_rows": sum(row["group"] == label for row in differences),
                         "all_private_storage_groups": len(set(view_pointers + center_pointers)) == 2 * len(hosts)})
            del native, batch
        torch.cuda.synchronize()
        checks = {"all_actual1016_and_fixed_generic_counts": [row["count"] for row in rows] == [1016, 1, 2, 7, 33],
                  "all_consumed_float_bits_and_native_layout_exact": not differences,
                  "all_private_camera_storage_groups": all(row["all_private_storage_groups"] for row in rows),
                  "all_original_global_RNG_states_unchanged": rng() == before_rng,
                  "input_and_all_executed_sources_preserved": input_id == data_identity(input_id["path"])
                    and source_ids == [data_identity(path) for path in sources]}
        atomic_json(output / "RESULT.json", {"status": "passed_batch_semantics_full_consumer_not_run0" if all(checks.values()) else "failed_exact_batch_camera_semantics0",
            "checks": checks, "rows": rows, "differences": differences, "runtime": runtime,
            "input": input_id, "source_identities": source_ids, "all1016_row_order": [row.name for row in actual.values()],
            "before_rng": before_rng, "after_rng": rng(), "normal_errors_recovery_full_consumer_external_GT_training": "not_run",
            "new_optimization_count": 0})
        print(json.dumps({"checks": checks, "rows": rows, "difference_count": len(differences), "new_optimization_count": 0}))
        if not all(checks.values()):
            raise AssertionError("The registered exact batch camera semantic gate failed")


if __name__ == "__main__":
    main()
