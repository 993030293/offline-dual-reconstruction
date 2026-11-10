"""Diagnose complete image upload bits after clean mapped-page release."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"
sys.path.insert(0, str(V1))
from runtime import activate_reference, atomic_json, data_identity
from mapped_training_images import mapped_training_images
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json) as guard:
        runtime = activate_reference(ROOT/"04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from PIL import Image
        import scene.cameras as cameras
        from scene.colmap_loader import read_extrinsics_binary
        dataset = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
        bindings = read_extrinsics_binary(dataset/"sparse/0/images.bin")
        if len(bindings) != 1016:
            raise ValueError("The fixed admitted image denominator changed")
        sources = [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("mapped_training_images.py")),
            data_identity(Path(__file__).with_name("mapped_image_linux_guard.py")),
            data_identity(Path(runtime["reference_source"])/"utils/general_utils.py")]
        native = cameras.PILtoTorch
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        free, _ = torch.cuda.mem_get_info()
        budget = min(12*1024**3, free-2*1024**3)
        if budget <= 0:
            raise MemoryError("The diagnostic has insufficient original CUDA headroom")

        def bits(value):
            return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

        rows = []
        with mapped_training_images(OUT/"full_cache_001", data_identity) as store:
            for name in sorted(row.name for row in bindings.values()):
                with Image.open(dataset/"images"/name) as image:
                    size = (round(image.width/2), round(image.height/2))
                    left = native(image, size)
                    right = cameras.PILtoTorch(image, size)
                native_CPU, mapped_CPU = bits(left), bits(right)
                left_GPU, right_GPU = left.cuda(), right.cuda()
                store.release_clean_pages(right)
                torch.cuda.synchronize()
                native_GPU, mapped_GPU = bits(left_GPU), bits(right_GPU)
                passed = native_CPU == mapped_CPU == native_GPU == mapped_GPU and left_GPU.stride() == right_GPU.stride()
                rows.append({"image_name": name, "all_CPU_CUDA_bytes_after_release_exact": passed,
                    "native_CPU": native_CPU, "mapped_CPU": mapped_CPU,
                    "native_GPU": native_GPU, "mapped_GPU_after_release": mapped_GPU,
                    "CUDA_strides_equal": left_GPU.stride() == right_GPU.stride()})
                if len(rows) % 128 == 0:
                    atomic_json(args.output/"PROGRESS.json", {"images": len(rows), "failed": sum(not row["all_CPU_CUDA_bytes_after_release_exact"] for row in rows),
                        "elapsed_seconds": time.perf_counter()-started, "counted": False})
                if torch.cuda.max_memory_reserved() > budget:
                    raise MemoryError("The fixed diagnostic CUDA allocation bound was exceeded")
                del left, right, left_GPU, right_GPU
            trace = store.receipt()
        checks = {"complete1016_native_CPU_CUDA_uploads_after_release_exact": len(rows) == 1016 and all(row["all_CPU_CUDA_bytes_after_release_exact"] for row in rows),
            "all1016_actual_clean_releases": trace["clean_page_release_calls"] == trace["prepared"] == 1016,
            "no_unregistered_dirty_pages": trace["dirty_release_skipped"] == 0,
            "factory_restored": cameras.PILtoTorch is native,
            "original_source_identity_preserved": sources == [data_identity(row["path"]) for row in sources]}
        guard.sample()
        result = {"status": "complete_input_upload_diagnostic_passed_quality_cause_unconfirmed" if all(checks.values()) else "failed_input_upload_diagnostic",
            "checks": checks, "rows": rows, "trace": trace, "sources": sources,
            "actual_peak_RSS_bytes": guard.row["peak_tree_RSS_bytes"], "CUDA_peak_bytes": torch.cuda.max_memory_reserved(),
            "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0,
            "negative_quality_gate_overridden": False, "GPU_nondeterminism_proven": False}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "complete_seconds": result["complete_seconds"]}))
        if not all(checks.values()):
            raise AssertionError("The complete uploaded image bytes or original release contract failed")


if __name__ == "__main__":
    main()
