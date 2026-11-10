"""Check native pixels, layouts, copy-on-write mutation and mapping lifetimes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from mapped_training_images import MappedTrainingImages, mapped_training_images
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json):
        runtime = activate_reference(ROOT/"04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        import torch
        from PIL import Image
        import scene.cameras as cameras
        native = cameras.PILtoTorch
        sources = [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("mapped_training_images.py")),
            data_identity(Path(__file__).with_name("mapped_image_linux_guard.py")),
            data_identity(Path(runtime["reference_source"])/"utils/general_utils.py")]
        checks, retained, traces = {}, [], []
        shapes = [(256, 16), (37, 29), (1, 1), (9, 13)]
        for index, shape in enumerate(shapes):
            data = np.arange(shape[0]*shape[1]*3, dtype=np.uint32).reshape(shape[1], shape[0], 3).astype(np.uint8)
            path = args.output/f"input_{index}.png"
            Image.fromarray(data).save(path)
            resolution = shape if index != 1 else (19, 11)
            with Image.open(path) as image:
                left = native(image, resolution)
            with mapped_training_images(args.output/f"cache_{index}", data_identity) as store:
                with Image.open(path) as image:
                    right = cameras.PILtoTorch(image, resolution)
                    second = cameras.PILtoTorch(image, resolution)
                checks[f"fixture{index}_all_pixels_shape_stride_dtype_CPU"] = torch.equal(left, right) and left.shape == right.shape and left.stride() == right.stride() and left.dtype == right.dtype and right.device.type == "cpu"
                checks[f"fixture{index}_independent_native_instances"] = right.data_ptr() != second.data_ptr() and torch.equal(right, second)
                saved = data_identity(next((args.output/f"cache_{index}").glob("*.f32")))
                before = second.clone()
                right[0, 0, 0] = -3.125
                store.release_clean_pages(right)
                store.release_clean_pages(second)
                checks[f"fixture{index}_dirty_COW_retained_clean_instance_isolated"] = right[0, 0, 0].item() == -3.125 and torch.equal(second, before) and saved == data_identity(saved["path"]) and store.trace["dirty_release_skipped"] == 1
                traces.append(store.receipt())
                retained.append(right)
            checks[f"fixture{index}_original_factory_restored_and_tensor_alive"] = cameras.PILtoTorch is native and retained[-1][0, 0, 0].item() == -3.125
        with mapped_training_images(args.output/"thread_cache", data_identity) as store:
            def prepare(index):
                with Image.open(args.output/f"input_{index%4}.png") as image:
                    value = cameras.PILtoTorch(image, shapes[index%4])
                    return value, native(image, shapes[index%4])
            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(prepare, range(8)))
            checks["four_worker_duplicates_exact_and_serialized_by_key"] = all(torch.equal(a,b) and a.stride() == b.stride() for a,b in values) and store.trace["created_files"] == 4 and store.trace["reused_files"] == 4
            store.release_clean_pages()
            traces.append(store.receipt())
        checks["thread_factory_restored"] = cameras.PILtoTorch is native
        for variant in ("missing", "truncated", "metadata", "budget", "RGBA"):
            cache = args.output/f"reject_{variant}"
            caught = None
            try:
                with mapped_training_images(cache, data_identity) as store:
                    if variant == "budget":
                        store.maximum_bytes = 1
                    if variant == "missing":
                        Image.open(args.output/"does_not_exist.png")
                    elif variant == "RGBA":
                        image = Image.open(args.output/"input_0.png").convert("RGBA")
                        cameras.PILtoTorch(image, shapes[0])
                    else:
                        with Image.open(args.output/"input_0.png") as image:
                            cameras.PILtoTorch(image, shapes[0])
                        payload = next(cache.glob("*.f32"))
                        metadata = next(cache.glob("*.json"))
                        if variant == "truncated":
                            payload.chmod(0o600)
                            with payload.open("r+b") as stream:
                                stream.truncate(4)
                        elif variant == "metadata":
                            metadata.chmod(0o600)
                            value = json.loads(metadata.read_text())
                            value["dtype"] = "<f8"
                            metadata.write_text(json.dumps(value))
                        with Image.open(args.output/"input_0.png") as image:
                            cameras.PILtoTorch(image, shapes[0])
            except (ValueError, MemoryError, FileNotFoundError) as error:
                caught = type(error).__name__
            checks[variant+"_actual_rejection_and_restore"] = caught is not None and cameras.PILtoTorch is native
        checks["pinned_sources_preserved"] = sources == [data_identity(x["path"]) for x in sources]
        result = {"status":"generic_mapped_pixels_and_COW_lifetimes_passed" if all(checks.values()) else "failed_verification",
            "checks":checks,"sources":sources,"runtime":runtime,"fixtures":traces,
            "new_optimization_count":0,"real_Scene_performance_or_quality_acceptance":False}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({"status":result["status"],"checks":checks}))
        if not all(checks.values()):
            raise AssertionError("A native mapped image contract failed")


if __name__ == "__main__":
    main()
