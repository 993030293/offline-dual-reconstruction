"""Own a thread-private exact RGGB scoring plane and its isolated native DLL."""

from contextlib import contextmanager
import ctypes
import inspect
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from native_quality_workspace import NativeQualityWorkspace


class BayerGrayEngine:
    def __init__(self, path):
        self.dll = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.dll.bayer_gray_avx2_available.argtypes = []
        self.dll.bayer_gray_avx2_available.restype = ctypes.c_int
        self.available = bool(self.dll.bayer_gray_avx2_available())
        self.compute = self.dll.bayer_gray_exact_u8
        self.compute.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_ssize_t]
        self.compute.restype = ctypes.c_int
        self.lock = threading.Lock()
        self.active_calls = self.calls = 0
        self.closed = False

    def supported(self, raw):
        return (not self.closed and self.available and cv2.__version__ == "4.13.0" and isinstance(raw, np.ndarray)
                and raw.ndim == 2 and raw.dtype == np.uint8 and raw.shape[0] > 0 and raw.shape[1] > 0
                and raw.size <= 2**24 and raw.strides[1] == 1 and raw.strides[0] >= raw.shape[1])

    def convert(self, raw, destination):
        if not self.supported(raw):
            raise ValueError("The exact RGGB DLL received an unsupported raw input")
        if destination.shape != raw.shape or destination.dtype != np.uint8 or not destination.flags.c_contiguous:
            raise ValueError("The exact RGGB destination must own the complete contiguous gray plane")
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed exact RGGB engine cannot execute")
            self.active_calls += 1
        try:
            code = self.compute(raw.ctypes.data, raw.strides[0], raw.shape[0], raw.shape[1],
                                destination.ctypes.data, destination.strides[0])
            if code:
                raise ValueError("The exact RGGB native input or destination bounds failed")
            with self.lock:
                self.calls += 1
            return destination
        finally:
            with self.lock:
                self.active_calls -= 1

    def receipt(self):
        return {"closed": self.closed, "DLL_unloaded": self.closed, "AVX2_available": self.available,
                "active_calls": self.active_calls, "calls": self.calls}

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active_calls:
                raise RuntimeError("The exact RGGB DLL cannot unload before all workers drain")
            import _ctypes
            handle = self.dll._handle
            self.compute = None
            _ctypes.FreeLibrary(handle)
            self.dll._handle = 0
            self.dll = None
            self.closed = True


class BayerGrayWorkspace(NativeQualityWorkspace):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def gray(self, raw):
        state = self.state()
        key = raw.shape, raw.dtype.str
        if state["key"] != key:
            for name in ("gray", "derivative", "reduced"):
                state[name] = None
            state["key"] = key
            state["shape_changes"] += 1
        created = state["gray"] is None
        if created:
            state["gray"] = np.empty(raw.shape, dtype=np.uint8)
        self.engine.convert(raw, state["gray"])
        state["calls"] += 1
        state["created" if created else "reused"] += 1
        self.measure(state)
        return state["gray"]


def compile_quality(original, workspace, histogram):
    source = inspect.getsource(original)
    changes = (
        ("gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)", "gray = _workspace.gray(image)"),
        ("derivative = cv2.Laplacian(gray, cv2.CV_16S)", "derivative = _workspace.derivative(gray)"),
        ("reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)", "reduced = _workspace.reduced(gray)"),
        ("    clipped = float(np.mean((gray <= 5) | (gray >= 250)))\n", "    brightness, clipped = _gray_moments(gray)\n"),
        ("float(gray.mean())", "brightness"),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("An original accepted quality calculation anchor changed")
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    namespace.update(_workspace=workspace, _gray_moments=histogram.gray_moments)
    exec(compile(source, "<exact_RGGB_original_quality>", "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def raw_gray_quality_execution(library, demosaic):
    import quality_histogram as histogram
    started = time.perf_counter()
    original = histogram.make_quality
    engine = BayerGrayEngine(library)
    fused_workspace, fallback_workspace = BayerGrayWorkspace(engine), NativeQualityWorkspace()
    traces = []

    def make_quality(native):
        fused = compile_quality(native, fused_workspace, histogram)
        fallback_histogram = compile_quality(native, fallback_workspace, histogram)
        trace = {"histogram_calls": 0, "native_calls": 0, "fallback_calls": 0, "setup_seconds": time.perf_counter() - started}
        traces.append(trace)

        def quality(raw):
            if engine.closed or fused_workspace.closed:
                raise RuntimeError("A closed exact RAW quality execution cannot score")
            if engine.supported(raw):
                trace["histogram_calls"] += 1
                return fused(raw)
            trace["fallback_calls"] += 1
            image = demosaic(raw)
            if (cv2.__version__ == "4.13.0" and image.ndim == 3 and image.dtype == np.uint8
                    and image.shape[2] == 3 and 0 < image.shape[0]*image.shape[1] <= 2**24):
                trace["histogram_calls"] += 1
                return fallback_histogram(image)
            trace["native_calls"] += 1
            return native(image)

        return quality, trace

    histogram.make_quality = make_quality
    trace = {"setup_seconds": time.perf_counter() - started, "counted": False}
    try:
        yield trace
    finally:
        before = time.perf_counter()
        changed = histogram.make_quality is not make_quality
        histogram.make_quality = original
        fused_workspace.close()
        fallback_workspace.close()
        engine.close()
        trace.update(builder_restored=histogram.make_quality is original, restore_seconds=time.perf_counter() - before,
                     fused_workspace=fused_workspace.receipt(), fallback_workspace=fallback_workspace.receipt(),
                     engine=engine.receipt(), consumer_traces=traces)
        if changed:
            raise RuntimeError("Another writer changed the owned exact RAW quality builder")
