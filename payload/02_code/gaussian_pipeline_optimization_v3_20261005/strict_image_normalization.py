"""Normalize owned Pillow byte images with checked strict native arithmetic."""

from contextlib import contextmanager
import ctypes
import inspect
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
from PIL import Image
import torch


def build_library(output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    source = Path(__file__).with_suffix(".c")
    copied = output / source.name
    shutil.copyfile(source, copied)
    library = output / "strict_image_normalization.so"
    command = ["cc", "-std=c11", "-O3", "-fPIC", "-shared", "-ffp-contract=off",
               "-fno-fast-math", "-frounding-math", str(copied), "-o", str(library)]
    started = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=60)
    (output / "COMPILER.log").write_text(result.stdout, encoding="utf-8")
    from runtime import atomic_json, data_identity
    atomic_json(output / "BUILD.json", {"command": command, "returncode": result.returncode,
        "actual_compile_seconds": time.perf_counter() - started,
        "source": data_identity(source), "source_copy": data_identity(copied),
        "no_environment_install": True})
    if result.returncode:
        raise RuntimeError("The private strict normalization compiler failed")
    return library


class ArithmeticLibrary:
    def __init__(self, path, vector=True):
        self.path = Path(path).resolve()
        self.native = ctypes.CDLL(str(self.path))
        for name in ("strict_normalization_abi", "strict_normalization_avx2"):
            getattr(self.native, name).argtypes = []
            getattr(self.native, name).restype = ctypes.c_int
        self.function = self.native.strict_normalization
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.function.restype = ctypes.c_int
        if self.native.strict_normalization_abi() != 1:
            raise ValueError("The strict arithmetic ABI changed")
        self.avx2 = bool(self.native.strict_normalization_avx2())
        if vector and not self.avx2:
            raise ValueError("Choose the original route before execution on unsupported CPUs")
        self.vector = vector

    def normalize(self, array):
        if array.dtype != np.uint8 or not array.flags.c_contiguous or array.size > 64 * 1024**2:
            raise ValueError("The owned byte normalization layout or capacity changed")
        output = np.empty(array.shape, dtype=np.float32, order="C")
        code = self.function(array.ctypes.data, output.ctypes.data, array.size, int(self.vector))
        if code:
            raise RuntimeError("Strict arithmetic normalization failed with code " + str(code))
        return output


class StrictImageNormalization:
    def __init__(self, native, library, enabled, maximum_elements=64 * 1024**2):
        source = inspect.getsource(native)
        anchors = ("resized_image_PIL = pil_image.resize(resolution)",
                   "resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0",
                   "return resized_image.permute(2, 0, 1)",
                   "return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)")
        if any(source.count(anchor) != 1 for anchor in anchors):
            raise ValueError("The original full image normalization source changed")
        if type(enabled) is not bool or maximum_elements <= 0:
            raise ValueError("An explicit route and positive capacity are required")
        self.native, self.enabled = native, enabled
        self.maximum_elements = maximum_elements
        started = time.perf_counter()
        self.library = ArithmeticLibrary(library) if enabled else None
        self.lock = threading.Lock()
        self.trace = {"enabled": enabled, "calls": 0, "arithmetic_calls": 0, "native_calls": 0,
                      "call_seconds": 0., "setup_seconds": time.perf_counter() - started,
                      "original_resize_float32_division_HWC_storage_and_CPU_residency": True,
                      "selected_avx2": self.library.avx2 if enabled else False}

    def __call__(self, image, resolution):
        started = time.perf_counter()
        byte_modes = ("L", "LA", "RGB", "RGBA", "P", "CMYK", "YCbCr", "HSV")
        supported_size = (isinstance(resolution, (tuple, list)) and len(resolution) == 2
                          and all(isinstance(value, int) and value > 0 for value in resolution))
        admitted = (self.enabled and isinstance(image, Image.Image) and image.mode in byte_modes
                    and supported_size and torch.get_default_dtype() == torch.float32
                    and resolution[0] * resolution[1] * len(image.getbands()) <= self.maximum_elements)
        if admitted:
            array = np.array(image.resize(resolution))
            physical = self.library.normalize(array)
            tensor = torch.from_numpy(physical)
            result = tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor.unsqueeze(-1).permute(2, 0, 1)
        else:
            result = self.native(image, resolution)
        with self.lock:
            self.trace["calls"] += 1
            self.trace["arithmetic_calls" if admitted else "native_calls"] += 1
            self.trace["call_seconds"] += time.perf_counter() - started
        return result


@contextmanager
def strict_image_normalization(library, enabled):
    import scene.cameras as cameras
    original = cameras.PILtoTorch
    adapter = StrictImageNormalization(original, library, enabled)
    cameras.PILtoTorch = adapter
    try:
        yield adapter.trace
    finally:
        cameras.PILtoTorch = original
        adapter.trace["original_binding_restored"] = cameras.PILtoTorch is original
