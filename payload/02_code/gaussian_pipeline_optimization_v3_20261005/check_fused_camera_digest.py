"""Check arbitrary float bits, native boundaries and real image digest equivalence."""

import ctypes
import json
from pathlib import Path
import pickle
import random
import shutil
import statistics
import sys
import time

import numpy as np
import psutil
from PIL import Image

from fused_camera_digest import DigestLibrary, ImageInput, build_library, original_digest, ROOT
from profile_scene_transform_cost import BoundedSceneGuard
from runtime import activate_reference, atomic_json, data_identity

CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def main():
    output = OUT / "g0"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), CODE / "fused_camera_digest.py", CODE / "fused_camera_digest.c", CODE / "fused_digest_probe_node.py"]
    sources = [data_identity(path) for path in paths]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, snapshots / (f"{index:03d}_" + path.name))
    with BoundedSceneGuard(output, atomic_json) as guard:
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils.general_utils import PILtoTorch
        library = build_library(output / "build")
        native = DigestLibrary(library)
        generator = np.random.default_rng(153)
        shapes = ((1, 1), (1, 7), (3, 11), (1, 8191), (1, 8192), (1, 8193), (83, 107))
        synthetic = []
        special = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fa00001, 1, 0xffffffff], dtype=np.uint32)
        for channels in (1, 2, 3, 4):
            for index, (height, width) in enumerate(shapes):
                size = channels*height*width
                bits = generator.integers(0, 2**32, size=size+16, dtype=np.uint32)
                count = min(len(special), size)
                bits[7:7+count] = special[:count]
                tensor = torch.from_numpy(bits.view(np.float32)).narrow(0, 7, size).reshape(height, width, channels).permute(2, 0, 1)
                if index == 2:
                    tensor.requires_grad_(True)
                synthetic.append(tensor)
        dataset = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
        image_paths = sorted(path for path in (dataset / "images").rglob("*") if path.is_file())[:16]
        if len(image_paths) != 16:
            raise ValueError("The original real image component denominator changed")
        inputs = [data_identity(path) for path in image_paths]
        real = []
        for path in image_paths:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("The original image scope is opaque RGB")
                real.append(PILtoTorch(image, (812, 620)))
        images = synthetic + real
        expected = [original_digest(value) for value in images]
        bits_before = [original_digest(value) for value in images]
        versions = [value._version for value in images]
        pointers = [(value.data_ptr(), value.untyped_storage().data_ptr(), value.storage_offset(), value.stride(), value.requires_grad) for value in images]
        before_rng = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone(),
                      [state.clone() for state in torch.cuda.get_rng_state_all()])
        scalar = native.images(images, workers=1, vectorized=False)
        vector = native.images(images, workers=1)
        four = native.images(images, workers=4)
        duplicate = native.images([images[0], images[0], images[-1], images[-1]], workers=4)
        descriptors = (ImageInput * 1)()
        raw_output = (ctypes.c_ubyte * 32)(*([171] * 32))
        status = (ctypes.c_int * 1)(123)
        illegal = [native.function(descriptors, 1, raw_output, status, 1, 0),
                   native.function(descriptors, 0, raw_output, status, 1, 0),
                   native.function(descriptors, 4097, raw_output, status, 1, 0),
                   native.function(descriptors, 1, raw_output, status, 2, 0),
                   native.function(descriptors, 1, raw_output, status, 1, 2)]
        rejected = 0
        for values, workers in (([], 4), (images, 2), ([torch.zeros((3, 2, 2), dtype=torch.float64)], 4)):
            try:
                native.images(values, workers=workers)
            except ValueError:
                rejected += 1
        unsupported = [torch.zeros((3, 2, 2), dtype=torch.float64), torch.zeros((3, 2, 2)),
                       torch.zeros((3, 0, 2)), torch.zeros((1, 3, 2, 2)), real[0][:, :, ::2]]
        timings = []
        real_expected = expected[-16:]
        for order in (("original", "serial", "four"), ("four", "serial", "original"), ("serial", "original", "four")):
            row = {"order": list(order)}
            for mode in order:
                begin = time.perf_counter()
                actual = [original_digest(value) for value in real] if mode == "original" else native.images(real, workers=1 if mode == "serial" else 4)
                row[mode] = time.perf_counter() - begin
                if actual != real_expected:
                    raise AssertionError("A timed component changed its exact standard digest")
            timings.append(row)
        checks = {
            "all28_arbitrary_bit_layouts_and16_real_images": len(synthetic) == 28 and len(real) == 16,
            "scalar_native_gather_complete_SHA_bytes_exact": scalar == expected,
            "runtime_selected_vector_gather_complete_SHA_bytes_exact": vector == expected,
            "four_worker_original_order_full_SHA_bytes_exact": four == expected,
            "duplicate_and_shared_inputs_keep_distinct_context_order": duplicate == [expected[0], expected[0], expected[-1], expected[-1]],
            "all_original_pixels_versions_pointers_layouts_grad_unchanged": bits_before == [original_digest(value) for value in images]
                and versions == [value._version for value in images] and pointers == [(value.data_ptr(), value.untyped_storage().data_ptr(), value.storage_offset(), value.stride(), value.requires_grad) for value in images],
            "nonzero_offset_all_float_bits_and_chunk_boundaries_exercised": all(value.storage_offset() == 7 for value in synthetic)
                and all(native.eligible(value) for value in synthetic),
            "native_invalid_descriptors_explicit_error_no_partial_output": illegal == [-1]*5 and list(raw_output) == [171]*32 and status[0] == 123,
            "Python_invalid_or_unsupported_native_admission_explicit": rejected == 3 and all(not native.eligible(value) for value in unsupported),
            "all_original_RNG_unchanged": before_rng[0] == pickle.dumps(random.getstate()) and before_rng[1] == pickle.dumps(np.random.get_state())
                and torch.equal(before_rng[2], torch.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(before_rng[3], torch.cuda.get_rng_state_all())),
            "all_three_component_orders_and_exact_outputs": len(timings) == 3 and all(all(row[key] > 0 for key in ("original", "serial", "four")) for row in timings),
            "full_image_pack_scratch_is32KiB_per_worker": native.library.fused_digest_scratch_bytes() == 32768,
            "all_source_library_and_actual_real_inputs_unchanged": sources == [data_identity(path) for path in paths]
                and inputs == [data_identity(path) for path in image_paths] and native.identity == data_identity(library),
        }
        guard.sample()
        result = {"status": "fused_layout_digest_generic44_exact_component_not_full_acceptance0" if all(checks.values()) else "failed_verification",
            "checks": checks, "sources": sources, "inputs": inputs, "library": native.identity, "selected_AVX2": native.avx2,
            "component16_real_three_orders_seconds": timings,
            "component_medians": {mode: statistics.median(row[mode] for row in timings) for mode in ("original", "serial", "four")},
            "scalar_vector_four_all44_digest_equal": scalar == vector == four == expected,
            "before_result_seconds": time.perf_counter()-started, "actual_owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
            "full1016_consumer_normal_errors_recovery_external_GT_training": "not_run", "new_optimization_count": 0}
        atomic_json(output / "RESULT.json", result)
        if not all(checks.values()):
            raise AssertionError("A native exact digest component changed an original contract")
        print(json.dumps({"checks": len(checks), "status": result["status"], "component_medians": result["component_medians"], "library": str(library)}))


if __name__ == "__main__":
    main()
