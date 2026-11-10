"""Read the existing NVIDIA memory API with explicit ABI and process ownership."""

import ctypes
from pathlib import Path
import subprocess

LIBRARY = Path("C:/Windows/System32/nvml.dll")
MIB = 1024**2


class MemoryV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class MemoryV2(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def cli_memory():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    return tuple(map(int, text.strip().split(",")))


class NativeMemory:
    """Own one NVML initialization and one explicitly admitted device handle."""

    def __init__(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8 or not LIBRARY.is_file():
            raise ValueError("The declared existing Windows64 NVML library is unavailable")
        self.library = ctypes.CDLL(str(LIBRARY))
        self.closed, self.initialized = False, False
        self.handle = ctypes.c_void_p()
        specs = {
            "nvmlInit_v2": [], "nvmlShutdown": [],
            "nvmlDeviceGetCount_v2": [ctypes.POINTER(ctypes.c_uint)],
            "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(MemoryV1)],
            "nvmlDeviceGetMemoryInfo_v2": [ctypes.c_void_p, ctypes.POINTER(MemoryV2)],
            "nvmlSystemGetDriverVersion": [ctypes.c_char_p, ctypes.c_uint],
            "nvmlDeviceGetName": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
        }
        for name, arguments in specs.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_uint
        try:
            self.check(self.library.nvmlInit_v2())
            self.initialized = True
            count = ctypes.c_uint()
            self.check(self.library.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            if count.value != 1:
                raise ValueError("The preserved original single-device CSV contract is unsupported")
            self.check(self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
            driver, name = ctypes.create_string_buffer(96), ctypes.create_string_buffer(128)
            self.check(self.library.nvmlSystemGetDriverVersion(driver, len(driver)))
            self.check(self.library.nvmlDeviceGetName(self.handle, name, len(name)))
            self.metadata = {"device_count": count.value, "device_name": name.value.decode(),
                "driver_version": driver.value.decode(), "library": str(LIBRARY),
                "memory_v1_struct_bytes": ctypes.sizeof(MemoryV1), "memory_v2_struct_bytes": ctypes.sizeof(MemoryV2)}
        except BaseException:
            if self.initialized:
                self.check(self.library.nvmlShutdown())
                self.initialized = False
            self.closed = True
            raise

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError("The admitted native NVML call failed with status "+str(code))

    def read(self, version):
        if self.closed or not self.initialized:
            raise RuntimeError("The owned NVML session is closed")
        if version == "v1":
            value = MemoryV1()
            self.check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)))
            return {"total": value.total, "free": value.free, "used": value.used}
        if version != "v2":
            raise ValueError("A declared memory ABI version is required")
        value = MemoryV2()
        value.version = ctypes.sizeof(MemoryV2) | 2 << 24
        self.check(self.library.nvmlDeviceGetMemoryInfo_v2(self.handle, ctypes.byref(value)))
        return {"total": value.total, "free": value.free, "used": value.used, "reserved": value.reserved}

    def mib(self, version):
        value = self.read(version)
        return value["used"]//MIB, value["free"]//MIB

    def close(self):
        if not self.closed:
            self.check(self.library.nvmlShutdown())
            self.closed, self.initialized = True, False
