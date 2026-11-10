"""Serve exact RGB sampling to one owned caller in the existing CUDA environment."""

import time
STARTED = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

from mapped_image_linux_guard import LinuxRSSGuard

HEADER = struct.Struct("<I")
ROOT = Path(__file__).resolve().parents[2]
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        block = stream.read(count - len(result))
        if not block:
            raise EOFError("The owned RGB request ended before its declared payload")
        result.extend(block)
    return result


def reply(record, payload=b""):
    record = dict(record, payload_bytes=len(payload))
    encoded = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sys.stdout.buffer.write(HEADER.pack(len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWNED)
    source_paths = [Path(__file__), Path(__file__).with_name("gpu_camera_remap_pixels.py"),
                    Path(__file__).with_name("mapped_image_linux_guard.py")]
    sources = [identity(path) for path in source_paths]
    maps, tensor, sampled = None, None, None
    map_ids, raw_ids, rows, current = [], [], [], None
    failure = None
    with LinuxRSSGuard(output, save, maximum_bytes=8 * 1024**3):
        import torch
        from gpu_camera_remap_pixels import sample_pixels
        if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
            raise RuntimeError("The pinned CUDA environment is required without CPU fallback")
        torch.set_num_threads(1)
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if free < 2 * 1024**3:
            raise MemoryError("The original CUDA headroom is unavailable before initialization")
        torch.cuda.reset_peak_memory_stats(device)
        try:
            while True:
                try:
                    size = HEADER.unpack(read_exact(sys.stdin.buffer, HEADER.size))[0]
                    if not 0 < size <= 65536:
                        raise ValueError("The owned RGB request metadata exceeds its bound")
                    record = json.loads(read_exact(sys.stdin.buffer, size))
                    started = time.perf_counter()
                    command = record["command"]
                    if command == "info":
                        free, total = torch.cuda.mem_get_info(device)
                        if free < 2 * 1024**3:
                            raise MemoryError("The original CUDA headroom is unavailable")
                        reply({"status": "ready", "Torch": torch.__version__,
                               "GPU_name": torch.cuda.get_device_name(device), "sources": sources})
                        continue
                    if command == "stop":
                        break
                    if command != "sample_file":
                        raise ValueError("The owned RGB command is unsupported")
                    height, width = record["height"], record["width"]
                    if (type(height) is not int or type(width) is not int or min(height, width) < 2
                            or height * width * 25 > 256 * 1024**2):
                        raise ValueError("The original map and RGB dimensions exceed their bound")
                    group = (output / record["group"]).resolve()
                    group.relative_to(output / "maps")
                    if current != str(group):
                        maps = None
                        paths = [group / (name + ".npy") for name in ("x0", "y0", "dx", "dy", "valid")]
                        expected = record["map_identities"]
                        actual = [identity(path) for path in paths]
                        if [{k: row[k] for k in ("bytes", "sha256")} for row in actual] != expected:
                            raise ValueError("A complete sampling map differs from the caller's identity")
                        values = [np.load(path, allow_pickle=False) for path in paths]
                        dtypes = (np.dtype("int32"), np.dtype("int32"), np.dtype("float64"),
                                  np.dtype("float64"), np.dtype("bool"))
                        if any(value.shape != (height, width) or value.dtype != dtype
                               for value, dtype in zip(values, dtypes)):
                            raise ValueError("A complete sampling map changed layout or precision")
                        if (not np.isfinite(values[2]).all() or not np.isfinite(values[3]).all()
                                or values[0].min() < 0 or values[0].max() >= width - 1
                                or values[1].min() < 0 or values[1].max() >= height - 1):
                            raise ValueError("A complete sampling map contains unsafe gather indices")
                        maps = tuple(torch.from_numpy(value).to(device=device,
                            dtype=torch.int64 if index < 2 else torch.float64 if index < 4 else torch.bool)
                            for index, value in enumerate(values))
                        map_ids.extend(actual)
                        current = str(group)
                    batch = (output / record["batch"]).resolve()
                    batch.relative_to(output / "batches")
                    source_path, result_path = batch / "raw.npy", batch / "sampled.npy"
                    actual_raw = identity(source_path)
                    if {k: actual_raw[k] for k in ("bytes", "sha256")} != record["raw_identity"]:
                        raise ValueError("The raw RGB file differs from its complete caller identity")
                    if result_path.exists():
                        raise FileExistsError("The exact RGB result must have a fresh owned path")
                    raw = np.load(source_path, allow_pickle=False)
                    if raw.dtype != np.uint8 or raw.shape != (height, width, 3):
                        raise ValueError("The complete original raw RGB file changed its layout")
                    raw_ids.append(actual_raw)
                    tensor = torch.from_numpy(raw).to(device=device)
                    sampled = sample_pixels(tensor, maps)
                    torch.cuda.synchronize(device)
                    result = sampled.cpu().numpy()
                    np.save(result_path, result, allow_pickle=False)
                    result_identity = identity(result_path)
                    free, total = torch.cuda.mem_get_info(device)
                    peak = torch.cuda.max_memory_allocated(device)
                    if peak > 12 * 1024**3 or free < 2 * 1024**3:
                        raise MemoryError("An original CUDA allocation or headroom guard failed")
                    row = {"ordinal": len(rows), "width": width, "height": height,
                           "worker_prepare_sample_D2H_seconds": time.perf_counter() - started,
                           "group": str(group)}
                    rows.append(row)
                    tensor, sampled, result, raw = None, None, None, None
                    reply(dict(row, status="sampled", result_identity={k: result_identity[k] for k in ("bytes", "sha256")}))
                except BaseException as error:
                    failure = repr(error)
                    reply({"status": "failed", "error": failure})
                    break
        finally:
            maps, tensor, sampled, values = None, None, None, None
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.empty_cache()
        checks = {"sources_unchanged": sources == [identity(path) for path in source_paths],
                  "all_complete_maps_unchanged": map_ids == [identity(Path(row["path"])) for row in map_ids],
                  "all_complete_raw_files_unchanged": raw_ids == [identity(Path(row["path"])) for row in raw_ids],
                  "owned_CUDA_tensors_released": torch.cuda.memory_allocated(device) == 0}
        if not all(checks.values()) and failure is None:
            failure = "An original execution identity or owned tensor close guard failed"
        result = {"status": "completed" if failure is None else "failed", "error": failure,
                  "checks": checks, "rows": rows, "sources": sources, "map_identities": map_ids, "raw_identities": raw_ids,
                  "peak_CUDA_allocated_bytes": peak_allocated, "peak_CUDA_reserved_bytes": peak_reserved,
                  "whole_worker_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
        save(output / "WORKER_RESULT.json", result)
        if failure is None:
            reply({"status": "stopped", "all_calls": len(rows), "owned_CUDA_tensors_released": checks["owned_CUDA_tensors_released"]})
    if failure is not None:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
