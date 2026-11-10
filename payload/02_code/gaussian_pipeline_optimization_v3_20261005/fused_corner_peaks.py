"""Replace only response threshold and peak materialization inside the current owner."""

from contextlib import contextmanager
import ctypes
from pathlib import Path

import numpy as np
import planar_corner


class FusedPeakCornerWorkspace(planar_corner.PlanarCornerWorkspace):
    def __init__(self, response_library, peak_library):
        super().__init__(response_library)
        self.peak_dll = None
        self.peak_select = None
        try:
            if not self.available:
                raise RuntimeError("The original AVX2 response capability is unavailable")
            self.peak_dll = ctypes.CDLL(str(Path(peak_library).resolve(strict=True)))
            abi = self.peak_dll.fused_corner_peaks_abi
            abi.argtypes, abi.restype = [], ctypes.c_int
            if abi() != 1:
                raise ValueError("The private fused peak ABI changed")
            self.peak_select = self.peak_dll.fused_corner_peaks_select
            self.peak_select.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.peak_select.restype = ctypes.c_int
        except BaseException:
            self.close()
            raise

    def corners(self, image):
        self.enter()
        try:
            eigen = self.response(image)
            output = np.empty((800, 1, 2), np.float32)
            count = ctypes.c_int()
            code = self.peak_select(eigen.ctypes.data, *image.shape, output.ctypes.data, ctypes.byref(count))
            if code or not 0 <= count.value <= 800:
                raise ValueError("The private fused corner peak extraction failed with code " + str(code))
            with self.lock:
                self.native_calls += 1
            return output[:count.value].copy() if count.value else None
        finally:
            self.leave()

    def close(self):
        super().close()
        self.peak_select = None
        if self.peak_dll is not None:
            import _ctypes
            _ctypes.FreeLibrary(self.peak_dll._handle)
            self.peak_dll._handle = 0
            self.peak_dll = None


@contextmanager
def fused_peak_execution(library, enabled):
    if type(enabled) is not bool:
        raise TypeError("The private peak extraction route requires an explicit flag")
    original = planar_corner.PlanarCornerWorkspace
    owned = []
    trace = {"enabled": enabled, "workspaces": [], "original_factory_restored": False,
        "original_response_and_std_sort_grid_preserved": True, "new_optimization_count": 0}

    def workspace(response_library):
        value = FusedPeakCornerWorkspace(response_library, library)
        owned.append(value)
        return value

    if enabled:
        planar_corner.PlanarCornerWorkspace = workspace
    try:
        yield trace
    finally:
        expected = workspace if enabled else original
        changed = planar_corner.PlanarCornerWorkspace is not expected
        planar_corner.PlanarCornerWorkspace = original
        trace["original_factory_restored"] = planar_corner.PlanarCornerWorkspace is original
        trace["workspaces"] = [dict(value.receipt(), peak_DLL_unloaded=value.peak_dll is None) for value in owned]
        if changed or any(not row["closed"] or not row["peak_DLL_unloaded"] or row["active_calls"] for row in trace["workspaces"]):
            raise RuntimeError("The peak-extraction factory did not restore after callers drained")
