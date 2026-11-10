"""Fuse original gray statistics in an owned integer Windows DLL."""

from contextlib import contextmanager
import ctypes
import _ctypes
import inspect
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

import quality_histogram as histogram
from native_quality_workspace import native_quality_workspace_execution


class NativeGrayMoments:
    def __init__(self, path):
        if sys.platform != "win32":
            raise ValueError("The private gray moments DLL requires Windows")
        self.path = Path(path).resolve(strict=True)
        self.closed = False
        self.local = threading.local()
        self.lock = threading.Lock()
        self.states = []
        self.active = 0
        self.library = ctypes.CDLL(str(self.path))
        try:
            abi = self.library.gray_moments_abi
            abi.restype = ctypes.c_uint64
            abi.argtypes = []
            if abi() != 1:
                raise ValueError("The private gray moments ABI differs")
            self.function = self.library.gray_moments
            self.function.restype = ctypes.c_int
            self.function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                      ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        except BaseException:
            self.function = None
            self.close()
            raise

    def state(self):
        state = getattr(self.local, "state", None)
        if state is None:
            state = {"owner_thread": threading.get_ident(), "calls": 0,
                     "histogram": np.empty(256, np.uint64), "sum": np.empty(1, np.int64),
                     "squared_sum": np.empty(1, np.uint64)}
            self.local.state = state
            with self.lock:
                self.states.append(state)
        if state["owner_thread"] != threading.get_ident():
            raise RuntimeError("A fused output crossed its original private thread owner")
        return state

    def moments(self, gray):
        if not (isinstance(gray, np.ndarray) and gray.dtype == np.uint8 and gray.ndim == 2
                and gray.flags.c_contiguous and 0 < gray.size <= histogram.PIXEL_LIMIT):
            raise ValueError("Private gray statistics require a nonempty contiguous uint8 plane within the original budget")
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed private gray moments DLL cannot execute")
            self.active += 1
        try:
            state = self.state()
            code = self.function(gray.ctypes.data, gray.shape[0], gray.shape[1],
                                 state["histogram"].ctypes.data, 256, state["sum"].ctypes.data,
                                 state["squared_sum"].ctypes.data)
            if code:
                raise RuntimeError("The owned integer gray statistics failed with code " + str(code))
            counts = state["histogram"]
            if int(counts.sum(dtype=np.uint64)) != gray.size:
                raise ValueError("The complete gray histogram denominator changed")
            state["calls"] += 1
            return counts, int(state["sum"][0]), int(state["squared_sum"][0])
        finally:
            with self.lock:
                self.active -= 1

    def quality_statistics(self, gray):
        counts, derivative_sum, squared_sum = self.moments(gray)
        scale = 1.0 / gray.size
        mean = float(derivative_sum) * scale
        deviation = math.sqrt(max(float(squared_sum) * scale - mean * mean, 0.0))
        sharpness = float(np.float64(deviation) ** 2)
        brightness = int(counts @ histogram.LEVELS) / gray.size
        clipped = (int(counts[:6].sum(dtype=np.uint64)) + int(counts[250:].sum(dtype=np.uint64))) / gray.size
        return sharpness, float(brightness), float(clipped)

    def receipt(self):
        return {"closed": self.closed, "DLL_unloaded": self.library is None,
                "active_calls": self.active, "thread_count": len(self.states),
                "calls": sum(state["calls"] for state in self.states),
                "thread_states": [{key: state[key] for key in ("owner_thread", "calls")} for state in self.states],
                "owned_output_bytes_after_close": sum(state[name].nbytes for state in self.states
                    for name in ("histogram", "sum", "squared_sum") if state[name] is not None)}

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active:
                raise RuntimeError("The worker pool must drain before the private DLL closes")
            self.function = None
            handle = self.library._handle if self.library else None
            self.library = None
            if handle:
                _ctypes.FreeLibrary(handle)
            for state in self.states:
                for name in ("histogram", "sum", "squared_sum"):
                    state[name] = None
            self.closed = True


def make_fused_quality(original, workspace, engine):
    started = time.perf_counter()
    source = inspect.getsource(original)
    changes = (
        ("gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)", "gray = _native_quality_workspace.gray(image)"),
        ("    derivative = cv2.Laplacian(gray, cv2.CV_16S)\n    _, deviation = cv2.meanStdDev(derivative)\n    sharpness = float(deviation[0, 0] ** 2)\n    clipped = float(np.mean((gray <= 5) | (gray >= 250)))\n",
         "    sharpness, brightness, clipped = _fused_gray_moments.quality_statistics(gray)\n"),
        ("reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)", "reduced = _native_quality_workspace.reduced(gray)"),
        ("float(gray.mean())", "brightness"),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("The original complete gray statistics source anchor changed")
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    namespace.update(_native_quality_workspace=workspace, _fused_gray_moments=engine)
    exec(compile(source, "<owned_fused_gray_statistics>", "exec"), namespace)
    implementation = namespace[original.__name__]
    trace = {"histogram_calls": 0, "native_calls": 0, "setup_seconds": time.perf_counter() - started}

    def quality(image):
        supported = (cv2.__version__ == "4.13.0" and isinstance(image, np.ndarray) and image.dtype == np.uint8
            and image.ndim == 3 and image.shape[2] == 3 and image.shape[0] > 0 and image.shape[1] > 0
            and image.shape[0] * image.shape[1] <= histogram.PIXEL_LIMIT)
        trace["histogram_calls" if supported else "native_calls"] += 1
        return implementation(image) if supported else original(image)

    return quality, trace


@contextmanager
def fused_quality_execution(enabled, library):
    started = time.perf_counter()
    trace = {"enabled": enabled, "counted": False}
    native_trace = None
    try:
        with native_quality_workspace_execution(True) as native_trace:
            original = histogram.make_quality
            engine = NativeGrayMoments(library) if enabled else None
            workspace = original.__globals__["_private_quality_workspace"]

            def builder(function):
                return make_fused_quality(function, workspace, engine)

            trace["setup_seconds"] = time.perf_counter() - started
            if enabled:
                histogram.make_quality = builder
            try:
                yield trace
            finally:
                before = time.perf_counter()
                changed = histogram.make_quality is not (builder if enabled else original)
                histogram.make_quality = original
                if engine:
                    engine.close()
                trace.update(restore_seconds=time.perf_counter() - before,
                             fused_builder_restored=histogram.make_quality is original,
                             engine=engine.receipt() if engine else None)
                if changed:
                    raise RuntimeError("Another writer changed the owned fused quality builder")
    finally:
        trace["workspace"] = native_trace
