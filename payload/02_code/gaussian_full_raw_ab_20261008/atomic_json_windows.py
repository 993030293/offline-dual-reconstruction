"""Publish complete receipts with bounded retries for Windows sharing violations."""

import json
import os
from pathlib import Path
import tempfile
import time


def delete_sharing_evidence(path):
    """Distinguish an incompatible open reader from a real delete-access denial."""
    path = Path(path)
    if os.name != "nt":
        return {"path": str(path), "verified_delete_sharing_conflict": False}
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    handle = create(str(path), 0x10000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        code = ctypes.get_last_error()
    else:
        close(handle)
        code = 0
    return {"path": str(path), "delete_access_probe_winerror": code,
            "verified_delete_sharing_conflict": code in (32, 33)}


def shared_text(path):
    if os.name != "nt":
        return path.read_text(encoding="utf-8-sig")
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "r", encoding="utf-8-sig") as stream:
        return stream.read()


def read_json(path, maximum_retry_seconds=2.0):
    path = Path(path)
    started, retries = time.monotonic(), 0
    while True:
        try:
            return json.loads(shared_text(path))
        except OSError as error:
            elapsed = time.monotonic() - started
            sharing_error = (error.winerror in (5, 32, 33) or
                isinstance(error, PermissionError) and error.errno == 13 and error.winerror is None)
            if os.name != "nt" or not sharing_error or elapsed >= maximum_retry_seconds:
                raise
            time.sleep(min(0.01 * 2 ** min(retries, 5), maximum_retry_seconds - elapsed))
            retries += 1


def atomic_json(path, value, maximum_retry_seconds=2.0):
    path = Path(path)
    if maximum_retry_seconds < 0:
        raise ValueError("A publication retry bound cannot be negative")
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    started = time.monotonic()
    retries = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
        while True:
            try:
                os.replace(temporary, path)
                return {"retries": retries, "elapsed_seconds": time.monotonic() - started}
            except OSError as error:
                elapsed = time.monotonic() - started
                if os.name != "nt" or error.winerror not in (5, 32, 33) or elapsed >= maximum_retry_seconds:
                    if os.name == "nt" and getattr(error, "winerror", None) in (5, 32, 33):
                        error.publication_sharing_evidence = delete_sharing_evidence(path)
                    raise
                delay = min(0.01 * 2 ** min(retries, 5), maximum_retry_seconds - elapsed)
                retries += 1
                time.sleep(delay)
    finally:
        if temporary.exists():
            temporary.unlink()
