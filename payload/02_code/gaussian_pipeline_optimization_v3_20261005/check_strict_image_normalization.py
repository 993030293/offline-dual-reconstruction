"""Check original tensor bits and complete normalization calls before a trial."""

import time

STARTED = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import statistics
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from strict_image_normalization import ArithmeticLibrary, StrictImageNormalization, build_library, strict_image_normalization


def state():
    return (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().numpy().tobytes())


def layout(value):
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device),
            value.storage_offset(), value.requires_grad, value.contiguous().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nn001")
    output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=120, maximum_bytes=4 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        from utils.general_utils import PILtoTorch
        sources = [Path(__file__), Path(__file__).with_name("strict_image_normalization.py"),
            Path(__file__).with_name("strict_image_normalization.c"), Path(sys.modules[PILtoTorch.__module__].__file__)]
        identities = [data_identity(path) for path in sources]
        for path in sources:
            shutil.copyfile(path, output / ("source_" + path.name))
        library = build_library(output / "build")
        adapter = StrictImageNormalization(PILtoTorch, library, True)
        native_library = ArithmeticLibrary(library, vector=False)
        checks, rows = {}, []
        codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
        expected = (torch.from_numpy(codes) / 255.0).numpy()
        for route, engine in (("scalar", native_library), ("AVX2", adapter.library)):
            actual = engine.normalize(codes)
            checks[route + "_all256_division_bits_exact"] = actual.tobytes() == expected.tobytes()
        if not all(checks.values()):
            atomic_json(output / "FAILED_NUMERICS.json", {"checks": checks, "counted": False})
            raise AssertionError("Direct arithmetic does not preserve original Torch division bits")
        before = state()
        fixtures = []
        for mode, channels in (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4), ("CMYK", 4), ("P", 1)):
            array = codes if channels == 1 else np.stack([np.roll(codes, index, axis=1) for index in range(channels)], axis=-1)
            fixtures.append((Image.fromarray(array, mode=mode), True))
        fixtures += [(fixtures[2][0].convert(mode), True) for mode in ("YCbCr", "HSV")]
        fixtures += [(image, False) for image in (Image.fromarray(codes > 120),
            Image.fromarray(codes.astype(np.uint16) * 257), Image.fromarray(codes.astype(np.float32)),
            Image.fromarray(codes.astype(np.int32) - 128))]
        for image, admitted in fixtures:
            for size in ((16, 16), (1, 1), (1, 17), (17, 1), (19, 7), (4, 3)):
                a, b = adapter.trace["arithmetic_calls"], adapter.trace["native_calls"]
                original_bytes = image.tobytes()
                left, right = PILtoTorch(image, size), adapter(image, size)
                equal = layout(left) == layout(right) and image.tobytes() == original_bytes
                equal = equal and adapter.trace["arithmetic_calls"] - a == int(admitted) and adapter.trace["native_calls"] - b == int(not admitted)
                rows.append({"mode": image.mode, "size": size, "exact_bits_layout_and_route": equal})
        checks["all72_modes_sizes_tails_resize_and_original_routes_exact"] = len(rows) == 72 and all(row["exact_bits_layout_and_route"] for row in rows)
        saved_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            checks["float64_default_selects_full_original_exactly"] = layout(PILtoTorch(fixtures[2][0], (16, 16))) == layout(adapter(fixtures[2][0], (16, 16)))
        finally:
            torch.set_default_dtype(saved_dtype)
        small = StrictImageNormalization(PILtoTorch, library, True, maximum_elements=16)
        checks["capacity_selects_original_without_pixel_removal"] = layout(small(fixtures[2][0], (16, 16))) == layout(PILtoTorch(fixtures[2][0], (16, 16))) and small.trace["native_calls"] == 1
        one, two = adapter(fixtures[2][0], (16, 16)), adapter(fixtures[2][0], (16, 16))
        preserved = layout(two)
        one[0, 0, 0] = -1.25
        checks["outputs_have_independent_mutable_storage"] = layout(two) == preserved and one.data_ptr() != two.data_ptr()
        import scene.cameras as cameras
        original = cameras.PILtoTorch
        errors = []
        for function in (PILtoTorch, adapter):
            try:
                function(fixtures[2][0], (0, 3))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
        checks["original_invalid_resize_error_propagates"] = len(errors) == 2 and errors[0] == errors[1]
        try:
            with strict_image_normalization(library, True):
                cameras.PILtoTorch(fixtures[2][0], (0, 3))
        except ValueError:
            pass
        checks["context_restores_original_binding_after_actual_error"] = cameras.PILtoTorch is original
        photos = sorted((ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset/images").iterdir())
        if len(photos) != 1016:
            raise AssertionError("The original full dataset membership changed")
        selected = [photos[index] for index in np.linspace(0, len(photos) - 1, 12, dtype=int)]
        photo_checks = []
        for path in selected:
            with Image.open(path) as photo:
                photo_checks.append(layout(PILtoTorch(photo, (812, 620))) == layout(adapter(photo, (812, 620))))
        checks["12_full_real_photos_uniformly_cover_existing1016_exact"] = all(photo_checks)
        image = Image.fromarray(np.tile(codes, (39, 51))[:620, :812]).convert("RGB")
        pairs = []
        for trial in range(3):
            timing = {}
            for name in (("native", "arithmetic") if trial % 2 == 0 else ("arithmetic", "native")):
                begin = time.perf_counter()
                value = (PILtoTorch if name == "native" else adapter)(image, (812, 620))
                timing[name] = time.perf_counter() - begin
                if name == "native":
                    reference = layout(value)
                else:
                    candidate = layout(value)
            if reference != candidate:
                raise AssertionError("A complete conversion component changed tensor contents")
            pairs.append(timing)
        medians = {name: statistics.median(row[name] for row in pairs) for name in ("native", "arithmetic")}
        checks["all_original_rng_states_unchanged"] = state() == before
        checks["original_normalization_and_data_sources_unchanged"] = identities == [data_identity(path) for path in sources]
        atomic_json(output / "RESULT.json", {"status": "generic_direct_arithmetic_exact_component_only" if all(checks.values()) else "failed",
            "checks": checks, "rows": rows, "component_pairs": pairs, "component_medians": medians,
            "component_raw_reduction_not_whole_Scene": 1 - medians["arithmetic"] / medians["native"],
            "trace": adapter.trace, "sources": identities, "runtime": metadata,
            "library": data_identity(library), "actual_seconds_before_publication": time.perf_counter() - STARTED,
            "public_whole_error_cancel_recovery_formal_training_production": "not_run",
            "new_optimization_count": 0})
        print(json.dumps({"checks": checks, "component_medians": medians}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A strict arithmetic normalization oracle failed")


if __name__ == "__main__":
    main()
