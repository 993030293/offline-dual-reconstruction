"""Release unused glibc pages at an explicit completed loading boundary."""

import ctypes
import sys
import time


def trim_unused_host_pages():
    if sys.platform != "linux":
        raise RuntimeError("This explicit allocator adapter is verified only for Linux glibc")
    libc = ctypes.CDLL("libc.so.6")
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    started = time.monotonic()
    result = trim(0)
    if result not in (0, 1):
        raise RuntimeError("Unexpected glibc trim result")
    return {"api": "glibc_malloc_trim", "returned": result, "seconds": time.monotonic()-started,
            "scope": "unused host pages after completed scene loading", "persistent_allocator_setting_changed": False}
