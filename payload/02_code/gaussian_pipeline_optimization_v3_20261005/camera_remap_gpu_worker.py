"""Run a bounded CUDA pixel pass over immutable staged native-array inputs."""

import time
STARTED = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mapped_image_linux_guard import LinuxRSSGuard


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.resolve().relative_to(root / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001")
    source_ids = [identity(Path(__file__)), identity(Path(__file__).with_name("gpu_camera_remap_pixels.py")),
                  identity(Path(__file__).with_name("mapped_image_linux_guard.py"))]
    manifest = json.loads((args.output / "MANIFEST.json").read_text(encoding="utf-8"))
    payloads = [path for path in sorted(args.output.rglob("*.npy"))]
    payload_ids = [identity(path) for path in payloads]
    with LinuxRSSGuard(args.output, save, maximum_bytes=8 * 1024**3):
        import torch
        from gpu_camera_remap_pixels import sample_pixels
        if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
            raise RuntimeError("The pinned CUDA environment is required; no silent CPU fallback is allowed")
        torch.set_num_threads(1)
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if free < 2 * 1024**3:
            raise MemoryError("The original CUDA headroom was unavailable before sampling")
        torch.cuda.reset_peak_memory_stats(device)
        maps, current_group, rows = None, None, []
        try:
            for case in manifest["cases"]:
                directory = args.output / case["case"]
                started = time.perf_counter()
                if current_group != case["group"]:
                    maps = None
                    values = [np.load(args.output / case["group"] / (name + ".npy"), allow_pickle=False)
                              for name in ("x0", "y0", "dx", "dy", "valid")]
                    maps = tuple(torch.from_numpy(value).to(device=device,
                        dtype=torch.int64 if index < 2 else torch.float64 if index < 4 else torch.bool)
                        for index, value in enumerate(values))
                    current_group = case["group"]
                raw = np.load(directory / "raw.npy", allow_pickle=False)
                tensor = torch.from_numpy(raw).to(device=device)
                torch.cuda.synchronize(device)
                prepared = time.perf_counter() - started
                working = time.perf_counter()
                sampled = sample_pixels(tensor, maps)
                torch.cuda.synchronize(device)
                sampled_seconds = time.perf_counter() - working
                completed = sampled.cpu().numpy()
                np.save(directory / "sampled.npy", completed, allow_pickle=False)
                peak = torch.cuda.max_memory_allocated(device)
                free, total = torch.cuda.mem_get_info(device)
                if peak > 12 * 1024**3 or free < 2 * 1024**3:
                    raise MemoryError("An unchanged CUDA allocation or headroom protection failed")
                rows.append({"case": case["case"], "prepare_load_maps_and_H2D_seconds": prepared,
                             "warm_sampling_seconds": sampled_seconds,
                             "load_H2D_sample_D2H_and_file_seconds": time.perf_counter() - started})
                tensor, sampled, completed = None, None, None
        finally:
            maps, tensor, sampled, completed, values = None, None, None, None, None
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.empty_cache()
        if source_ids != [identity(Path(row["path"])) for row in source_ids] or payload_ids != [identity(path) for path in payloads]:
            raise RuntimeError("A staged input or execution source changed in the owned CUDA process")
        result = {"status": "completed_owned_CUDA_pixel_preconditions", "cases": len(rows), "rows": rows,
                  "all_cases_sampled": len(rows) == len(manifest["cases"]),
                  "owned_tensors_released": torch.cuda.memory_allocated(device) == 0,
                  "peak_CUDA_allocated_bytes": peak_allocated, "peak_CUDA_reserved_bytes": peak_reserved,
                  "GPU_name": torch.cuda.get_device_name(device), "Torch": torch.__version__,
                  "sources": source_ids, "input_payloads": payload_ids,
                  "whole_before_publication_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
        save(args.output / "CUDA_RESULT.json", result)
        print(json.dumps({"status": result["status"], "cases": len(rows), "peak_CUDA_allocated_bytes": peak_allocated}), flush=True)


if __name__ == "__main__":
    main()
