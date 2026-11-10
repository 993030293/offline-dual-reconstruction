"""Retain exact native CPU image tensors in private copy-on-write file mappings."""

from contextlib import contextmanager
import hashlib
import inspect
import json
import mmap
import os
from pathlib import Path
import threading
import time

import numpy as np


class MappedTrainingImages:
    def __init__(self, cache, native_prepare, identity, maximum_bytes=8*1024**3):
        if not hasattr(mmap.mmap, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
            raise RuntimeError("This private image mapping requires native Linux madvise")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.native_prepare, self.identity = native_prepare, identity
        self.prepare_identity = identity(Path(inspect.getsourcefile(native_prepare)))
        self.maximum_bytes = int(maximum_bytes)
        if self.maximum_bytes <= 0:
            raise ValueError("The image mapping byte budget must be positive")
        self.lock = threading.RLock()
        self.key_locks, self.charged, self.records, self.by_pointer = {}, {}, [], {}
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.pagemap_descriptor = os.open("/proc/self/pagemap", os.O_RDONLY)
        self.trace = {"prepared": 0, "created_files": 0, "reused_files": 0,
            "payload_bytes": 0, "mapping_calls": 0, "clean_page_release_calls": 0,
            "dirty_release_skipped": 0, "logical_released_bytes": 0,
            "native_prepare_seconds": 0., "complete_prepare_seconds": 0.,
            "kernel_page_checks": 0, "private_or_swapped_page_release_skipped": 0,
            "kernel_private_or_swapped_pages_observed": 0,
            "file_cache_is_not_process_RSS": True, "precision_or_frame_reduction": False}

    def prepare(self, image, resolution):
        import torch
        started = time.perf_counter()
        if image.mode != "RGB" or len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError("The private mapped image entry requires positive opaque RGB input")
        source = self.identity(Path(image.filename))
        declaration = {"schema_version": 1, "source": source,
            "prepare_source": self.prepare_identity, "resolution": list(resolution),
            "layout": "native_HWC_float32_little_endian_CHW_view", "mode": "RGB"}
        key = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()
        payload, metadata = self.cache/(key+".f32"), self.cache/(key+".json")
        expected_bytes = int(resolution[0])*int(resolution[1])*3*4
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if payload.exists() != metadata.exists():
                raise ValueError("A mapped image cache has an incomplete payload/metadata pair")
            with self.lock:
                if key not in self.charged:
                    if len(self.charged) >= 2048 or sum(self.charged.values())+expected_bytes > self.maximum_bytes:
                        raise MemoryError("The mapped image cache exceeds its full input count or byte budget")
                    self.charged[key] = expected_bytes
            if not payload.exists():
                before = time.perf_counter()
                value = self.native_prepare(image, resolution)
                elapsed = time.perf_counter()-before
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad
                        or tuple(value.shape) != (3, resolution[1], resolution[0])
                        or value.stride() != (1, 3*resolution[0], 3)):
                    raise ValueError("The original PIL preparation left its exact float32 RGB layout")
                raw = value.permute(1, 2, 0).contiguous().numpy().tobytes()
                if len(raw) != expected_bytes:
                    raise ValueError("The complete native image payload changed its declared size")
                if source != self.identity(Path(image.filename)):
                    raise ValueError("The source image changed during exact preparation")
                with payload.open("xb") as stream:
                    stream.write(raw)
                os.chmod(payload, 0o400)
                info = {**declaration, "payload_sha256": hashlib.sha256(raw).hexdigest(),
                    "payload_bytes": len(raw), "dtype": "<f4", "physical_shape": [resolution[1], resolution[0], 3]}
                with metadata.open("x", encoding="utf-8") as stream:
                    json.dump(info, stream, sort_keys=True)
                os.chmod(metadata, 0o400)
                with self.lock:
                    self.trace["created_files"] += 1
                    self.trace["native_prepare_seconds"] += elapsed
            else:
                info = json.loads(metadata.read_text(encoding="utf-8"))
                if any(info.get(k) != v for k, v in declaration.items()):
                    raise ValueError("The mapped image cache identity or native preparation changed")
                with self.lock:
                    self.trace["reused_files"] += 1
            actual = self.identity(payload)
            if (actual["bytes"] != expected_bytes or info["payload_bytes"] != expected_bytes
                    or info["payload_sha256"] != actual["sha256"] or info["dtype"] != "<f4"
                    or info["physical_shape"] != [resolution[1], resolution[0], 3]
                    or set(info) != {*declaration, "payload_sha256", "payload_bytes", "dtype", "physical_shape"}):
                raise ValueError("A complete mapped float32 image payload changed")
            physical = np.memmap(payload, mode="c", dtype="<f4", shape=(resolution[1], resolution[0], 3))
            value = torch.from_numpy(physical.transpose(2, 0, 1))
            record = {"tensor": value, "mapping": physical._mmap,
                "version": value._version, "bytes": expected_bytes, "path": str(payload)}
            with self.lock:
                self.records.append(record)
                self.by_pointer[value.untyped_storage().data_ptr()] = record
                self.trace["prepared"] += 1
                self.trace["mapping_calls"] += 1
                self.trace["payload_bytes"] = sum(self.charged.values())
                self.trace["complete_prepare_seconds"] += time.perf_counter()-started
            return value

    def private_page_count(self, row):
        if self.pagemap_descriptor is None:
            raise ValueError("The mapped-page observer was already closed")
        pointer = row["tensor"].untyped_storage().data_ptr()
        first = pointer//self.page_size
        last = (pointer+row["bytes"]+self.page_size-1)//self.page_size
        raw = os.pread(self.pagemap_descriptor, (last-first)*8, first*8)
        if len(raw) != (last-first)*8:
            raise ValueError("The complete mapped image page table could not be inspected")
        entries = np.frombuffer(raw, dtype="<u8")
        present = (entries & np.uint64(1 << 63)) != 0
        swapped = (entries & np.uint64(1 << 62)) != 0
        file_page = (entries & np.uint64(1 << 61)) != 0
        # Tensor.numpy() writes bypass _version; private COW pages must survive.
        count = int(np.count_nonzero(swapped | (present & ~file_page)))
        self.trace["kernel_page_checks"] += 1
        self.trace["kernel_private_or_swapped_pages_observed"] += count
        return count

    def release_clean_pages(self, value=None):
        with self.lock:
            rows = self.records if value is None else [self.by_pointer[value.untyped_storage().data_ptr()]]
            for row in rows:
                if row["tensor"]._version != row["version"]:
                    self.trace["dirty_release_skipped"] += 1
                    continue
                if not row["mapping"].closed:
                    if self.private_page_count(row):
                        self.trace["dirty_release_skipped"] += 1
                        self.trace["private_or_swapped_page_release_skipped"] += 1
                        continue
                    row["mapping"].madvise(mmap.MADV_DONTNEED)
                    self.trace["clean_page_release_calls"] += 1
                    self.trace["logical_released_bytes"] += row["bytes"]

    def receipt(self):
        with self.lock:
            return {**self.trace, "maximum_cache_bytes": self.maximum_bytes,
                "mapping_count": len(self.records), "unique_payloads": len(self.charged),
                "retained_native_base_tensors": len(self.records),
                "cache": str(self.cache), "prepare_source": self.prepare_identity,
                "native_tensor_lifetime_preserved_after_context": True}


@contextmanager
def mapped_training_images(cache, identity, enabled=True):
    import scene.cameras as cameras
    original = cameras.PILtoTorch
    if not enabled:
        yield None
        return
    if original.__name__ == "consume":
        raise ValueError("The mapped preparer must be installed before bounded prefetch")
    store = MappedTrainingImages(cache, original, identity)
    cameras.PILtoTorch = store.prepare
    try:
        yield store
    finally:
        cameras.PILtoTorch = original
        # NumPy ownership retained by each native tensor keeps its mapping valid.
        store.records.clear()
        store.by_pointer.clear()
        os.close(store.pagemap_descriptor)
        store.pagemap_descriptor = None
