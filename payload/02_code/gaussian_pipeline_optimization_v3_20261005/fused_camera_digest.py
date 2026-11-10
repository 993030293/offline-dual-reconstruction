"""Bind a checked private layout-gather and standard SHA-256 implementation."""

import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


class ImageInput(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("pixels", ctypes.c_uint64),
                ("channels", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("available_bytes", ctypes.c_uint64)]


def build_library(output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    source = Path(__file__).with_suffix(".c")
    copied = output / source.name
    shutil.copyfile(source, copied)
    library = output / "fused_camera_digest.so"
    command = ["cc", "-std=c11", "-O3", "-fno-strict-aliasing", "-fPIC", "-shared", "-fopenmp",
               str(copied), "-lcrypto", "-o", str(library)]
    started = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    (output / "COMPILER.log").write_text(result.stdout, encoding="utf-8")
    atomic_json(output / "BUILD.json", {"command": command, "returncode": result.returncode,
        "actual_compile_seconds": time.perf_counter()-started, "source": data_identity(source),
        "source_copy": data_identity(copied), "no_environment_install": True})
    if result.returncode:
        raise RuntimeError("The private native digest build failed; compiler log preserved")
    return library


class DigestLibrary:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.identity = data_identity(self.path)
        self.library = ctypes.CDLL(str(self.path))
        for name in ("fused_digest_abi", "fused_digest_avx2"):
            getattr(self.library, name).argtypes = []
            getattr(self.library, name).restype = ctypes.c_int
        self.library.fused_digest_scratch_bytes.argtypes = []
        self.library.fused_digest_scratch_bytes.restype = ctypes.c_uint64
        self.library.fused_digest_descriptor_bytes.argtypes = []
        self.library.fused_digest_descriptor_bytes.restype = ctypes.c_uint32
        self.function = self.library.fused_digest_batch
        self.function.argtypes = [ctypes.POINTER(ImageInput), ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte),
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_uint32, ctypes.c_uint32]
        self.function.restype = ctypes.c_int
        if (self.library.fused_digest_abi() != 1 or self.library.fused_digest_scratch_bytes() != 32768
            or self.library.fused_digest_descriptor_bytes() != ctypes.sizeof(ImageInput)):
            raise ValueError("The private digest ABI or bounded scratch layout changed")
        self.avx2 = bool(self.library.fused_digest_avx2())

    @staticmethod
    def eligible(value):
        import torch
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 3 or value.is_conj() or value.is_neg()):
            return False
        channels, height, width = value.shape
        if not (1 <= channels <= 4 and height > 0 and width > 0
                and value.stride() == (1, width*channels, channels) and value.numel()*4 <= 256*1024**2):
            return False
        storage = value.untyped_storage()
        available = storage.data_ptr() + storage.nbytes() - value.data_ptr()
        return value.storage_offset() >= 0 and available >= value.numel()*4

    def images(self, images, workers=4, vectorized=None):
        images = list(images)
        if type(workers) is not int or workers not in (1, 4) or not 0 < len(images) <= 4096:
            raise ValueError("A bounded original image batch and one or four workers are required")
        if not all(self.eligible(value) for value in images):
            raise ValueError("The explicit native layout path requires admitted interleaved float32 views")
        vectorized = self.avx2 if vectorized is None else vectorized
        if type(vectorized) is not bool or vectorized and not self.avx2:
            raise ValueError("The explicit gather ISA is unavailable")
        versions = [value._version for value in images]
        descriptors = (ImageInput * len(images))()
        for index, value in enumerate(images):
            storage = value.untyped_storage()
            descriptors[index] = ImageInput(value.data_ptr(), value.shape[1]*value.shape[2], value.shape[0], 0,
                                           storage.data_ptr()+storage.nbytes()-value.data_ptr())
        output = (ctypes.c_ubyte * (32*len(images)))()
        statuses = (ctypes.c_int * len(images))()
        code = self.function(descriptors, len(images), output, statuses, workers, int(vectorized))
        if code or any(statuses):
            raise RuntimeError("A native digest error was not replaced with stale or fallback output: " + repr((code, list(statuses))))
        if versions != [value._version for value in images]:
            raise RuntimeError("An owned image changed during native layout digest")
        return [bytes(output[index*32:(index+1)*32]).hex() for index in range(len(images))]


def original_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
