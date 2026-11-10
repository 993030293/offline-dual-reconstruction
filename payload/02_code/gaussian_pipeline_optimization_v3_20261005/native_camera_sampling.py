"""Sample private array-origin RGB bitmaps through an owned strict-float Windows DLL."""

from contextlib import contextmanager
import ctypes
import _ctypes
import json
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap

from camera_pixel_remap import CameraPixelRemap, SUPPORTED, camera_fields


class NativeCameraPixelRemap(CameraPixelRemap):
    def __init__(self, original, library, maximum_bytes=256*1024**2):
        if sys.platform != "win32":
            raise ValueError("This source-pinned sampling implementation requires the existing Windows host")
        super().__init__(original, maximum_bytes)
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.native_camera_sampling_abi.argtypes = []
        self.library.native_camera_sampling_abi.restype = ctypes.c_int
        if self.library.native_camera_sampling_abi() != 1:
            self.close()
            raise ValueError("The private RGB sampling ABI changed")
        self.sample = self.library.native_camera_sample_rgb8
        self.sample.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_size_t]*2
        self.sample.restype = ctypes.c_int
        self.trace.update(native_DLL=True, native_DLL_closed=False, sampler_calls=0)

    def sample_array(self, raw):
        if self.closed or self.maps is None:
            raise ValueError("A closed or unbuilt native sampling map cannot accept pixels")
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3 or not raw.flags.c_contiguous:
            raise ValueError("Complete native RGB sampling requires a contiguous HWC uint8 source")
        expected = (np.dtype(np.int32), np.dtype(np.int32), np.dtype(np.float64), np.dtype(np.float64), np.dtype(bool))
        if len(self.maps) != len(expected) or any(array.shape != raw.shape[:2] or array.dtype != dtype or not array.flags.c_contiguous
               or array.flags.writeable for array, dtype in zip(self.maps, expected)):
            raise ValueError("The private immutable sampling map layout changed")
        output = np.empty_like(raw)
        result = self.sample(raw.ctypes.data, *[array.ctypes.data for array in self.maps],
                             output.ctypes.data, raw.shape[1], raw.shape[0])
        if result:
            raise ValueError("The native RGB sampling boundary guard failed with code " + str(result))
        self.trace["sampler_calls"] += 1
        return output

    def __call__(self, options, bitmap, camera):
        if self.closed:
            raise RuntimeError("An owned sampling map cannot be used after close")
        started = time.perf_counter()
        supported = (camera.model_name in SUPPORTED and bitmap.is_rgb
            and bitmap.width == camera.width and bitmap.height == camera.height
            and camera.width >= 2 and camera.height >= 2
            and camera.width*camera.height*25 <= self.maximum_bytes)
        if not supported:
            self.trace["routes"].append("original_before_mapping")
            return self.original(options, bitmap, camera)
        key = json.dumps({"camera": camera_fields(camera), "options": options.todict()}, sort_keys=True, allow_nan=False)
        if self.key != key:
            calibration = pycolmap.undistort_camera(options, camera)
            self.build(camera, calibration)
            self.key = key
        raw = bitmap.to_array()
        if raw.shape != (camera.height, camera.width, 3):
            raise ValueError("An owned array-origin RGB bitmap changed its native dimensions")
        output = self.sample_array(raw)
        result = pycolmap.Bitmap.from_array(output)
        calibration = pycolmap.Camera(**self.calibration)
        if (calibration.width, calibration.height) != (camera.width, camera.height):
            result.rescale(calibration.width, calibration.height)
        self.trace["routes"].append("owned_native_calibration_map")
        self.trace["calls"].append({"seconds": time.perf_counter()-started, "camera": camera_fields(camera),
            "result_camera": camera_fields(calibration), "map_bytes": self.trace["map_bytes"]})
        return result, calibration

    def close(self):
        super().close()
        library = getattr(self, "library", None)
        if library is not None:
            self.sample = None
            self.library = None
            _ctypes.FreeLibrary(library._handle)
        self.trace["native_DLL_closed"] = True


@contextmanager
def native_camera_sampling(enabled, library):
    started = time.perf_counter()
    original = pycolmap.undistort_image
    operator = NativeCameraPixelRemap(original, library) if enabled else None
    trace = {"enabled": enabled, "setup_seconds": time.perf_counter()-started,
             "close_seconds": 0., "closed": False, "original_binding_restored": False, "operator": None}
    try:
        if operator is not None:
            pycolmap.undistort_image = operator
        yield trace
    finally:
        started = time.perf_counter()
        pycolmap.undistort_image = original
        if operator is not None:
            operator.close()
            trace["operator"] = operator.trace
        trace.update(close_seconds=time.perf_counter()-started, closed=True,
                     original_binding_restored=pycolmap.undistort_image is original)
