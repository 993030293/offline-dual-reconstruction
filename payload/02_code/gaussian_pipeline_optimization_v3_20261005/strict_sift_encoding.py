"""Fuse checked descriptor conversion and trace successful NumPy output bytes."""

import ctypes
import inspect
from pathlib import Path
import threading
import warnings

import numpy as np

import integer_sift_matching as native

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/build0"
DEFAULT_ERRORS = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


class StrictSIFTEncoder:
    def __init__(self, enabled=False):
        self.enabled, self.owner, self.calls = enabled, threading.get_ident(), []
        self.active = None
        source = inspect.getsource(native.bytes_exact)
        edits = [
            ('np.isfinite(values)', '_record(np.isfinite(values), "finite")'),
            ('scaled = values*512', 'scaled = _record(values*512, "scaled")'),
            ('np.rint(scaled)', '_record(np.rint(scaled), "rint")'),
            ('result = scaled.astype(np.uint8)', 'result = _record(scaled.astype(np.uint8), "packed")'),
            ('result.astype(np.float32)/512', '_record(_record(result.astype(np.float32), "restored_float")/512, "restored_scaled")'),
        ]
        for before, after in edits:
            if source.count(before) != 1:
                raise ValueError("The original strict descriptor conversion changed: " + before)
            source = source.replace(before, after)
        namespace = dict(native.bytes_exact.__globals__)
        namespace["_record"] = self.record
        exec(compile(source, str(Path(__file__)), "exec"), namespace)
        self.original = namespace["bytes_exact"]
        self.library = None
        if enabled:
            self.library = ctypes.CDLL(str(BUILD / "native_sift_encoding.dll"))
            self.function = self.library.encode_sift_bytes
            self.function.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
            self.function.restype = ctypes.c_int

    def record(self, value, name):
        self.active["actual_successful_numpy_outputs"].append({"operation": name, "shape": list(value.shape),
            "dtype": str(value.dtype), "strides": list(value.strides), "bytes": value.nbytes,
            "owns_data": bool(value.flags.owndata)})
        return value

    def supported(self, values):
        return (self.enabled and threading.get_ident() == self.owner and type(values) is np.ndarray
                and values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 128
                and 1 <= len(values) <= 12000 and np.geterr() == DEFAULT_ERRORS
                and all(abs(stride)*max(size, 1) < 2**62 for stride, size in zip(values.strides, values.shape)))

    def __call__(self, values):
        row = {"route": "original", "status": "running", "actual_successful_numpy_outputs": []}
        if self.active is not None:
            raise RuntimeError("The owned encoder cannot recursively replace a descriptor call")
        self.active = row
        try:
            if not self.supported(values):
                result = self.original(values)
            else:
                row["route"] = "native_single_scan"
                result = self.record(np.empty_like(values, dtype=np.uint8, order="K", subok=False), "packed_native_output")
                code = self.function(values.ctypes.data, len(values), *values.strides,
                                     result.ctypes.data, *result.strides)
                row["native_returncode"] = code
                if code == 1:
                    raise ValueError("The original finite float32 native query contract changed")
                if code == 3:
                    warnings.warn("overflow encountered in multiply", RuntimeWarning, stacklevel=2)
                if code in (2, 3):
                    raise ValueError("A native SIFT component cannot be represented exactly as a byte")
                if code != 0:
                    raise RuntimeError("The private checked descriptor encoder returned an unsupported status")
            row["status"] = "completed"
            return result
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["actual_recorded_output_bytes"] = sum(value["bytes"] for value in row["actual_successful_numpy_outputs"])
            self.calls.append(row)
            self.active = None


def receipt(encoder):
    return {"calls": encoder.calls, "actual_recorded_successful_output_bytes": sum(
        row["actual_recorded_output_bytes"] for row in encoder.calls),
        "counts_only_six_explicit_native_numpy_array_outputs_not_hardware_peak_or_total_allocations": True,
        "no_failed_execution_fallback": True}
