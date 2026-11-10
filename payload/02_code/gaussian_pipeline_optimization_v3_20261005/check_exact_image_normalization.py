"""Check native image normalization for varied modes, sizes and explicit routes."""

import argparse
import hashlib
from pathlib import Path
import pickle
import random
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import activate_reference, atomic_json, data_identity
from exact_image_normalization import ExactImageNormalization, exact_image_normalization


def layout(value):
    return (list(value.shape), list(value.stride()), str(value.dtype), str(value.device),
            hashlib.sha256(value.contiguous().numpy().tobytes(order="C")).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(args.runtime)
    from utils.general_utils import PILtoTorch
    import scene.cameras as cameras
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    adapter = ExactImageNormalization(PILtoTorch, True)
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    fixtures = []
    for mode, channels in (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4), ("CMYK", 4)):
        data = codes if channels == 1 else np.stack([np.roll(codes, i, axis=1) for i in range(channels)], axis=-1)
        fixtures.append((mode, Image.fromarray(data, mode=mode), True))
    palette = Image.fromarray(codes, mode="P")
    palette.putpalette(np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 3)).reshape(-1).tolist())
    fixtures.append(("P", palette, True))
    for mode in ("YCbCr", "HSV"):
        fixtures.append((mode, fixtures[2][1].convert(mode), True))
    fixtures.extend((mode, image, False) for mode, image in (
        ("1", Image.fromarray(codes > 120)),
        ("I;16", Image.fromarray(codes.astype(np.uint16)*257)),
        ("F", Image.fromarray(np.linspace(-2, 3, 256, dtype=np.float32).reshape(16, 16))),
        ("I", Image.fromarray(np.arange(-128, 128, dtype=np.int32).reshape(16, 16)))))
    checks, rows = {}, []
    for name, image, admitted in fixtures:
        for resolution in ((16, 16), (1, 1), (19, 7), (4, 3)):
            original_bytes = image.tobytes()
            old_native, old_lookup = adapter.trace["native_calls"], adapter.trace["lookup_calls"]
            reference, candidate = PILtoTorch(image, resolution), adapter(image, resolution)
            passed = (layout(reference) == layout(candidate) and image.tobytes() == original_bytes
                      and adapter.trace["lookup_calls"]-old_lookup == int(admitted)
                      and adapter.trace["native_calls"]-old_native == int(not admitted))
            rows.append({"mode": name, "resolution": resolution, "exact_and_route": passed})
    checks["all_modes_sizes_full256_byte_values_and_layouts_exact"] = all(row["exact_and_route"] for row in rows)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        old = adapter.trace["native_calls"]
        checks["float64_default_preselects_native_exactly"] = (layout(PILtoTorch(fixtures[2][1], (16, 16)))
            == layout(adapter(fixtures[2][1], (16, 16))) and adapter.trace["native_calls"] == old+1)
    finally:
        torch.set_default_dtype(old_dtype)
    small = ExactImageNormalization(PILtoTorch, True, maximum_elements=16)
    checks["over_bound_preselects_native_without_image_truncation"] = (layout(PILtoTorch(fixtures[2][1], (16, 16)))
        == layout(small(fixtures[2][1], (16, 16))) and small.trace["native_calls"] == 1)
    original, threads = cameras.PILtoTorch, cv2.getNumThreads()
    try:
        with exact_image_normalization(True):
            cameras.PILtoTorch(fixtures[2][1], (0, 3))
    except ValueError:
        checks["native_invalid_resize_error_propagated"] = True
    else:
        checks["native_invalid_resize_error_propagated"] = False
    checks["global_function_and_cv2_threads_restored_after_error"] = cameras.PILtoTorch is original and cv2.getNumThreads() == threads
    checks["all_rng_states_unchanged"] = (before[0] == pickle.dumps(random.getstate())
        and before[1] == pickle.dumps(np.random.get_state()) and torch.equal(before[2], torch.get_rng_state()))
    checks["read_only_table"] = not adapter.table.flags.writeable
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "rows": rows,
        "trace": adapter.trace, "counted": False, "full_new_object_pipeline_not_tested": True,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("exact_image_normalization.py"))]}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The native image normalization oracle failed")


if __name__ == "__main__":
    main()
