"""Gate the strict SIMD sampler before any map or image output is constructed."""

import _ctypes
import ctypes
from pathlib import Path

from camera_pixel_remap import CameraPixelRemap
from native_camera_sampling import NativeCameraPixelRemap


def library_supports_simd(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    try:
        abi = library.native_camera_sampling_abi
        capabilities = library.native_camera_sampling_simd_capabilities
        for function in (abi, capabilities):
            function.argtypes = []
            function.restype = ctypes.c_int
        if abi() != 1:
            raise ValueError("The strict RGB sampling ABI changed")
        return capabilities() == 1
    finally:
        _ctypes.FreeLibrary(library._handle)


class VectorCameraPixelRemap(NativeCameraPixelRemap):
    def __init__(self, original, library, maximum_bytes=256 * 1024**2):
        self.vector_admitted = library_supports_simd(library)
        if self.vector_admitted:
            super().__init__(original, library, maximum_bytes)
        else:
            CameraPixelRemap.__init__(self, original, maximum_bytes)
            self.trace.update(native_DLL=False, native_DLL_closed=True)
        self.trace["actual_vector_ISA_admitted"] = self.vector_admitted

    def __call__(self, options, bitmap, camera):
        if self.closed:
            raise RuntimeError("An owned sampling map cannot be used after close")
        if not self.vector_admitted:
            self.trace["routes"].append("original_before_unsupported_ISA")
            return self.original(options, bitmap, camera)
        return super().__call__(options, bitmap, camera)

    def sample_array(self, raw):
        if not self.vector_admitted:
            raise RuntimeError("A nonadmitted SIMD sampler cannot accept a direct array call")
        return super().sample_array(raw)
