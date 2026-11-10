"""Own exact vectorized gray moments and compose with the accepted RAW consumer."""

from contextlib import contextmanager
import ctypes
import inspect
import math
from pathlib import Path
import threading

import cv2
import numpy as np

ABI_VERSION = 2


class VectorGrayStats:
    def __init__(self, library):
        self.closed = False
        self.local = threading.local()
        self.lock = threading.Lock()
        self.states = []
        self.active_calls = self.native_calls = self.fallback_calls = 0
        self.dll = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.function = None
        try:
            abi = self.dll.vector_gray_stats_abi
            abi.argtypes, abi.restype = [], ctypes.c_int
            if abi() != ABI_VERSION:
                raise ValueError("The private vector gray statistics ABI differs")
            available = self.dll.vector_gray_stats_avx2_available
            available.argtypes, available.restype = [], ctypes.c_int
            self.available = bool(available())
            self.function = self.dll.vector_gray_stats_u8
            self.function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
            self.function.restype = ctypes.c_int
        except BaseException:
            self.close()
            raise

    def supported(self, gray):
        return (not self.closed and self.available and cv2.__version__ == "4.13.0" and cv2.useOptimized()
                and isinstance(gray, np.ndarray) and gray.dtype == np.uint8 and gray.ndim == 2
                and gray.flags.c_contiguous and 0 < gray.size <= 2**24)

    def moments(self, gray):
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed vector gray statistics DLL cannot execute")
            if not self.supported(gray):
                raise ValueError("The private vector statistics input is unsupported")
            self.active_calls += 1
        try:
            state = getattr(self.local, "state", None)
            if state is None:
                state = {"owner_thread": threading.get_ident(), "output": np.empty(4, np.int64), "calls": 0}
                self.local.state = state
                with self.lock:
                    self.states.append(state)
            if state["owner_thread"] != threading.get_ident():
                raise RuntimeError("A private vector output crossed its thread owner")
            code = self.function(gray.ctypes.data, *gray.shape, state["output"].ctypes.data, 4)
            if code:
                raise ValueError("The private vector statistics failed with code " + str(code))
            state["calls"] += 1
            with self.lock:
                self.native_calls += 1
            return tuple(int(value) for value in state["output"])
        finally:
            with self.lock:
                self.active_calls -= 1

    def quality_statistics(self, gray):
        if self.closed:
            raise RuntimeError("A closed vector gray statistics consumer cannot execute")
        if not self.supported(gray):
            from quality_histogram import gray_moments
            with self.lock:
                self.fallback_calls += 1
            _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
            brightness, clipped = gray_moments(gray)
            return float(deviation[0, 0] ** 2), brightness, clipped
        brightness_sum, clipped_count, derivative_sum, squared_sum = self.moments(gray)
        scale = 1.0 / gray.size
        mean = float(derivative_sum) * scale
        deviation = math.sqrt(max(float(squared_sum) * scale - mean * mean, 0.0))
        return float(np.float64(deviation) ** 2), brightness_sum / gray.size, clipped_count / gray.size

    def receipt(self):
        return {"closed": self.closed, "DLL_unloaded": self.dll is None, "active_calls": self.active_calls,
                "native_calls": self.native_calls, "fallback_calls": self.fallback_calls,
                "private_states": len(self.states), "retained_output_bytes": sum(
                    state["output"].nbytes for state in self.states if state["output"] is not None)}

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active_calls:
                raise RuntimeError("The vector statistics DLL cannot unload before callers drain")
            import _ctypes
            self.function = None
            _ctypes.FreeLibrary(self.dll._handle)
            self.dll._handle = 0
            self.dll = None
            for state in self.states:
                state["output"] = None
            self.closed = True


def compile_vector_quality(original, workspace, histogram, engine, fallback):
    source = inspect.getsource(original)
    changes = (
        ("gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)", "gray = _workspace.gray(image)"),
        ("    derivative = cv2.Laplacian(gray, cv2.CV_16S)\n    _, deviation = cv2.meanStdDev(derivative)\n    sharpness = float(deviation[0, 0] ** 2)\n    clipped = float(np.mean((gray <= 5) | (gray >= 250)))\n",
         "    sharpness, brightness, clipped = _vector_gray_stats.quality_statistics(gray)\n"),
        ("reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)", "reduced = _workspace.reduced(gray)"),
        ("float(gray.mean())", "brightness"),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("An original vector quality source anchor changed")
        source = source.replace(before, after)
    namespace = dict(fallback.__globals__)
    namespace.update(_workspace=workspace, _vector_gray_stats=engine)
    exec(compile(source, "<owned_exact_vector_gray_statistics>", "exec"), namespace)
    implementation = namespace[original.__name__]

    def quality(image):
        if engine.closed:
            raise RuntimeError("A closed vector quality consumer cannot execute")
        if (cv2.__version__ == "4.13.0" and cv2.useOptimized() and engine.available
                and isinstance(image, np.ndarray) and image.dtype == np.uint8
                and image.ndim in (2, 3) and (image.ndim == 2 or image.shape[2] == 3)
                and 0 < image.shape[0] * image.shape[1] <= 2**24):
            return implementation(image)
        return fallback(image)

    return quality


@contextmanager
def vector_gray_stats_execution(library):
    import exact_bayer_gray as accepted
    original = accepted.compile_quality
    engine = VectorGrayStats(library)

    def compile_quality(native, workspace, histogram):
        fallback = original(native, workspace, histogram)
        return compile_vector_quality(native, workspace, histogram, engine, fallback)

    accepted.compile_quality = compile_quality
    trace = {"counted": False}
    try:
        yield trace
    finally:
        changed = accepted.compile_quality is not compile_quality
        accepted.compile_quality = original
        engine.close()
        trace.update(original_factory_restored=accepted.compile_quality is original, engine=engine.receipt())
        if changed:
            raise RuntimeError("Another writer changed the owned vector statistics compiler")
