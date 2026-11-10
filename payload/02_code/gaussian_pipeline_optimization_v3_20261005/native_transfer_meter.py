"""Read actual CUDA runtime copy calls from the explicit private preload."""

import ctypes
import os
from pathlib import Path


class Transfer(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_uint64), ("kind", ctypes.c_int32),
                ("result", ctypes.c_int32), ("api", ctypes.c_int32),
                ("reserved", ctypes.c_int32)]


class NativeTransferMeter:
    def __init__(self):
        root = Path(__file__).resolve().parents[2]
        expected = root / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build/native_transfer_counter.so"
        if os.environ.get("LD_PRELOAD") != str(expected) or not expected.is_file():
            raise RuntimeError("The native transfer meter requires its sole explicit private preload")
        self.library = ctypes.CDLL(None)
        self.library.owned_transfer_begin.argtypes = []
        self.library.owned_transfer_begin.restype = ctypes.c_int
        self.library.owned_transfer_end.argtypes = [ctypes.POINTER(Transfer), ctypes.c_uint64]
        self.library.owned_transfer_end.restype = ctypes.c_uint64
        self.library.owned_transfer_backend.argtypes = []
        self.library.owned_transfer_backend.restype = ctypes.c_char_p
        self.backend = self.library.owned_transfer_backend().decode()
        if not self.backend or "libcudart.so" not in Path(self.backend).name:
            raise RuntimeError("The actual native runtime copy backend is unavailable")
        self.active = False

    def begin(self):
        if self.active or self.library.owned_transfer_begin() != 0:
            raise RuntimeError("An owned native transfer scope is already active")
        self.active = True

    def end(self):
        if not self.active:
            raise RuntimeError("The owned native transfer scope is not active")
        rows = (Transfer * 64)()
        count = self.library.owned_transfer_end(rows, 64)
        self.active = False
        if count > 64:
            raise RuntimeError("The real native copy event capacity overflowed")
        values = [{name: int(getattr(row, name)) for name, _ in Transfer._fields_}
                  for row in rows[:count]]
        if any(row["kind"] not in (0, 1, 2, 3) or row["reserved"] for row in values):
            raise RuntimeError("An actual native transfer direction is unsupported or ambiguous")
        return values
