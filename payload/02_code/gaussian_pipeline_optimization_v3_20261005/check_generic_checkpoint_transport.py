"""Verify unchanged byte transport across heterogeneous native tensor states."""

import argparse
from collections import OrderedDict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from zstd_checkpoint import encode_checkpoint, verified_expand, Zstd


def tensor_schema(state):
    import torch
    rows, storages = [], {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            token = (storage.data_ptr(), storage.nbytes())
            if token not in storages:
                storages[token] = len(storages)
            rows.append({"path": path, "shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "offset": value.storage_offset(), "storage_group": storages[token],
                "storage_bytes": storage.nbytes()})
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, path+"/"+str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, path+"/"+str(index))

    walk(state, "state")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    paths = [Path(__file__), Path(__file__).with_name("zstd_checkpoint.py")]
    identity = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": identity, "system_zstd_version": Zstd().version,
        "opaque_native_complete_bytes_no_object_class_or_geometry": True,
        "heterogeneous_dtype_shape_stride_storage_alias_and_metadata": True,
        "complex_empty_and_large_multiblock_payloads": True,
        "native_bytes_exact_and_Torch_load_unchanged": True, "full_new_object_training_not_executed": True,
        "new_optimization_count": 0})
    generator = torch.Generator().manual_seed(20261005)
    shared = torch.randn((37, 11), generator=generator)
    mixtures = OrderedDict([
        ("float64", torch.randn((19, 4), dtype=torch.float64, generator=generator)),
        ("half", shared.half()), ("bfloat16", shared.bfloat16()), ("signed_ids", torch.arange(23, dtype=torch.int64)*17-1),
        ("boolean", torch.arange(101)%3 == 0), ("empty", torch.empty((0, 7))),
        ("noncontiguous", shared.T), ("offset_alias", shared[3:29:2, 2:9]), ("base_alias", shared),
        ("complex", torch.complex(shared, shared*.25)),
        ("metadata", {"cameras": ["arbitrary/irregular.jpg", "曲面.png"], "RNG_python": random.Random(20261005).getstate(),
            "RNG_numpy": np.random.default_rng(20261005).bit_generator.state, "RNG_Torch_CPU": generator.get_state(),
            "remaining_indices": [2, 0, 4], "scale_is_input_not_GT": 100.})])
    fixtures = [
        ("mixed", mixtures),
        ("empty_model", {"parameters": {"xyz": torch.empty((0, 3)), "sh": torch.empty((0, 25, 3))}, "optimizer": {"state": {}, "param_groups": []}}),
        ("irregular_counts", {"xyz": torch.randn((317, 3), generator=generator),
            "sh": torch.randn((317, 25, 3), generator=generator),
            "optimizer": {"step": torch.tensor(153), "exp_avg": torch.randn((317, 3), generator=generator),
                "exp_avg_sq": torch.rand((317, 3), generator=generator)}, "weighting": None}),
        ("multiblock_high_entropy", {"arbitrary_auxiliary": torch.randint(0, 256, (9*1024**2+31,), dtype=torch.uint8, generator=generator)})]
    checks, rows = {}, []
    for name, state in fixtures:
        target = args.output/name
        target.mkdir()
        native, archive, expanded = target/"native.pth", target/"state.zstd", target/"expanded.pth"
        torch.save(state, native)
        encode = encode_checkpoint(native, archive)
        decode = verified_expand(archive, expanded)
        a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in (native, expanded)]
        before, after = data_identity(native), data_identity(expanded)
        exact = before["sha256"] == after["sha256"] and before["bytes"] == after["bytes"]
        layout = tensor_schema(a) == tensor_schema(b)
        checks[name+"_all_native_bytes_and_shared_tensor_layout_exact"] = exact and layout
        rows.append({"fixture": name, "complete_bytes_exact": exact, "dtype_stride_offset_storage_alias_exact": layout,
            "records": encode["records"], "native_bytes": before["bytes"], "transport_bytes": archive.stat().st_size,
            "decode": decode})
    checks["multiblock_record_boundary_exercised"] = rows[-1]["records"] > 1
    checks["unchanged_accepted_codec_source"] = identity == [data_identity(p) for p in paths]
    result = {"status": "generic_native_checkpoint_transport_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "fixtures": rows, "sources": identity, "new_optimization_count": 0,
        "new_object_training_or_recovery_quality_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An unchanged complete native transport contract failed")


if __name__ == "__main__":
    main()
