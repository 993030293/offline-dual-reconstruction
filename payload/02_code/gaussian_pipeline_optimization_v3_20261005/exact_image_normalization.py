"""Replace native uint8 CPU normalization with an exact bounded lookup pass."""

from contextlib import contextmanager
import inspect
import threading
import time

import cv2
import numpy as np
import torch


class ExactImageNormalization:
    def __init__(self, native, enabled, maximum_elements=64*1024**2):
        if type(enabled) is not bool or maximum_elements <= 0:
            raise ValueError("An explicit normalization route and positive bound are required")
        source = inspect.getsource(native)
        for anchor in ("resized_image_PIL = pil_image.resize(resolution)",
                       "resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0",
                       "return resized_image.permute(2, 0, 1)",
                       "return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)"):
            if source.count(anchor) != 1:
                raise ValueError("The original image normalization contract changed")
        self.native, self.enabled = native, enabled
        self.maximum_elements = maximum_elements
        self.lock = threading.Lock()
        self.trace = {"enabled": enabled, "calls": 0, "lookup_calls": 0, "native_calls": 0,
                      "call_seconds": 0., "setup_seconds": 0., "maximum_elements": maximum_elements,
                      "native_resize_and_CPU_float32_HWC_strides_preserved": True,
                      "CPU_original_float_images_remain_resident": True,
                      "cv2_version": cv2.__version__, "torch_version": torch.__version__}
        started = time.perf_counter()
        self.table = None
        if enabled:
            if torch.__version__ != "2.7.1+cu128" or cv2.__version__ != "4.11.0":
                raise ValueError("Select native normalization before execution outside the fixed runtime")
            self.table = (torch.from_numpy(np.arange(256, dtype=np.uint8))/255.0).numpy()
            self.table.setflags(write=False)
        self.trace["setup_seconds"] = time.perf_counter()-started

    def __call__(self, image, resolution):
        started = time.perf_counter()
        supported_modes = ("L", "LA", "RGB", "RGBA", "P", "CMYK", "YCbCr", "HSV")
        shape_supported = (isinstance(resolution, (list, tuple)) and len(resolution) == 2
                           and all(isinstance(x, int) and x > 0 for x in resolution))
        admitted = (self.enabled and image.mode in supported_modes and shape_supported
                    and torch.get_default_dtype() == torch.float32
                    and resolution[0]*resolution[1]*len(image.getbands()) <= self.maximum_elements)
        if admitted:
            array = np.array(image.resize(resolution))
            if array.dtype != np.uint8 or array.ndim not in (2, 3):
                raise ValueError("An admitted Pillow byte image changed its array representation")
            # The table retains CPU division bit patterns without recomputing arithmetic.
            physical = cv2.LUT(array.reshape(-1, 1), self.table).reshape(array.shape)
            tensor = torch.from_numpy(physical)
            result = tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor.unsqueeze(-1).permute(2, 0, 1)
        else:
            result = self.native(image, resolution)
        seconds = time.perf_counter()-started
        with self.lock:
            self.trace["calls"] += 1
            self.trace["lookup_calls" if admitted else "native_calls"] += 1
            self.trace["call_seconds"] += seconds
        return result


@contextmanager
def exact_image_normalization(enabled):
    import scene.cameras as cameras
    original = cameras.PILtoTorch
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        adapter = ExactImageNormalization(original, enabled)
        cameras.PILtoTorch = adapter
        yield adapter.trace
    finally:
        cameras.PILtoTorch = original
        cv2.setNumThreads(original_threads)
