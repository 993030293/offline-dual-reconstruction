"""Reuse full digests only while a Windows byte-immutable read handle remains held."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import sys


class FileInfo(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
        ("access", wintypes.FILETIME), ("write", wintypes.FILETIME), ("volume", wintypes.DWORD),
        ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
        ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]


class ReadHandle:
    def __init__(self, path):
        if os.name != "nt":
            raise RuntimeError("The selected snapshot implementation requires Windows sharing semantics")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self.api.CreateFileW.restype = wintypes.HANDLE
        self.api.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInfo)]
        self.api.GetFileInformationByHandle.restype = wintypes.BOOL
        self.api.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.api.ReadFile.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateFileW(str(Path(path).resolve()), 0x80000000, 1, None, 3, 0x08000000, None)
        if self.handle == ctypes.c_void_p(-1).value:
            self.handle = None
            raise ctypes.WinError(ctypes.get_last_error())

    def identity(self):
        value = FileInfo()
        if not self.api.GetFileInformationByHandle(self.handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (value.volume, value.index_high, value.index_low, value.size_high, value.size_low)

    def digest(self):
        value, count, result, size = ctypes.create_string_buffer(4*1024**2), wintypes.DWORD(), hashlib.sha256(), 0
        view = memoryview(value).cast("B")
        while True:
            if not self.api.ReadFile(self.handle, value, len(value), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                break
            result.update(view[:count.value])
            size += count.value
        return {"bytes": size, "sha256": result.hexdigest()}

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


class LockedInputs:
    def __init__(self, allowed_roots, original):
        self.roots = tuple(Path(p).resolve() for p in allowed_roots)
        self.original, self.rows = original, {}
        self.hits = self.misses = self.outside = self.avoided_logical_hash_bytes = 0

    def identity(self, path):
        raw, actual = str(Path(path)), Path(path).resolve()
        if not any(actual == root or actual.is_relative_to(root) for root in self.roots):
            self.outside += 1
            return self.original(path)
        if actual in self.rows:
            owned, file_id, digest = self.rows[actual]
            probe = ReadHandle(actual)
            try:
                if probe.identity() != file_id or owned.identity() != file_id:
                    raise ValueError("The held input path was replaced or its protected byte identity changed")
            finally:
                probe.close()
            self.hits += 1
            self.avoided_logical_hash_bytes += digest["bytes"]
        else:
            owned = ReadHandle(actual)
            try:
                file_id = owned.identity()
                digest = owned.digest()
                if owned.identity() != file_id or digest["bytes"] != (file_id[3]<<32)+file_id[4]:
                    raise ValueError("The complete held input hash and byte length disagree")
            except BaseException:
                owned.close()
                raise
            self.rows[actual] = (owned, file_id, digest)
            self.misses += 1
        return dict(path=raw, **digest)

    def verify(self):
        for path, (owned, identity, _) in self.rows.items():
            probe = ReadHandle(path)
            try:
                if probe.identity() != identity or owned.identity() != identity:
                    raise ValueError("A held source path identity changed before completing the snapshot")
            finally:
                probe.close()

    def receipt(self):
        return {"hits": self.hits, "full_hashes": self.misses, "uncached_mutable_or_unselected": self.outside,
            "held_read_handles": len(self.rows), "avoided_logical_hash_bytes": self.avoided_logical_hash_bytes,
            "protection": "GENERIC_READ_FILE_SHARE_READ_without_write_or_delete_sharing",
            "every_hit_reopens_and_checks_volume_file_id_and_size": True,
            "hash_read_from_owned_handle_not_racy_path": True, "metadata_only_hash_cache": False}

    def close(self):
        errors = []
        for owned, _, _ in self.rows.values():
            try:
                owned.close()
            except OSError as error:
                errors.append(error)
        self.rows.clear()
        if errors:
            raise errors[0]


@contextmanager
def locked_input_identity(roots, enabled=True):
    import runtime
    original = runtime.data_identity
    snapshot = LockedInputs(roots, original) if enabled else None

    def identity(path):
        return snapshot.identity(path)

    def replace(old, new):
        for module in list(sys.modules.values()):
            if module is not None and getattr(module, "data_identity", None) is old:
                setattr(module, "data_identity", new)

    if enabled:
        replace(original, identity)
    try:
        yield snapshot
        if snapshot is not None:
            snapshot.verify()
    finally:
        if enabled:
            replace(identity, original)
            snapshot.close()
