"""Publish original quality-cache bytes through an owned native file lifecycle."""

from contextlib import contextmanager
import ctypes
import errno
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_cache_writer_001"
AUDIT_RENAME = ctypes.CFUNCTYPE(ctypes.c_int)
ABI_VERSION = 1


class NativeQualityCacheWriter:
    def __init__(self, library):
        self.closed = False
        self.local = threading.local()
        self.lock = threading.Lock()
        self.states = []
        self.active_calls = self.native_calls = self.fallback_calls = 0
        self.create_attempts = self.actual_flushes = self.closed_handles = self.bytes_written = 0
        self.dump = json.dump
        self.dll = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.function = None
        self.callback = AUDIT_RENAME(self.audit_rename)
        try:
            abi = self.dll.quality_cache_writer_abi
            abi.argtypes, abi.restype = [], ctypes.c_int
            if abi() != ABI_VERSION:
                raise ValueError("The private cache writer ABI differs")
            self.function = self.dll.quality_cache_publish
            self.function.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint64,
                                      AUDIT_RENAME, ctypes.c_void_p, ctypes.c_uint64]
            self.function.restype = ctypes.c_int
        except BaseException:
            self.close()
            raise

    def supported(self, cache, key, scores):
        from quality_engine import FIELDS
        if not (sys.platform == "win32" and not self.closed and type(key) is str and re.fullmatch(r"[0-9a-f]{64}", key)
                and type(scores) is dict and set(scores) == set(FIELDS)
                and all(type(value) is int and value.bit_length() <= 64
                        or type(value) is float and math.isfinite(value) for value in scores.values())
                and cache.directory.is_absolute() and json.dump is self.dump):
            return False
        try:
            cache.directory.resolve().relative_to(BASE.resolve())
        except ValueError:
            return False
        return len(str(cache.directory)) < 200

    def state(self):
        state = getattr(self.local, "state", None)
        if state is None:
            state = {"owner": threading.get_ident(), "output": np.empty(5, np.uint64), "audit_error": None,
                     "temporary": None, "target": None}
            self.local.state = state
            with self.lock:
                self.states.append(state)
        if state["owner"] != threading.get_ident():
            raise RuntimeError("A native cache output crossed its thread owner")
        return state

    def audit_rename(self):
        state = self.local.state
        try:
            sys.audit("os.rename", str(state["temporary"]), str(state["target"]), -1, -1)
            return 1
        except BaseException as error:
            state["audit_error"] = error
            return 0

    def publish_record(self, cache, key, record):
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed native cache writer cannot publish")
            self.active_calls += 1
        state = None
        try:
            state = self.state()
            text_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_BINARY | os.O_NOINHERIT
            sys.audit("open", str(cache.directory), "w", text_flags)
            with io.StringIO() as buffer:
                self.dump(record, buffer, allow_nan=False)
                body = buffer.getvalue().encode("utf-8")
            if not 0 < len(body) <= 16384:
                raise ValueError("The original native cache record exceeded its private byte budget")
            state["target"] = cache.directory / (key + ".json")
            names = tempfile._get_candidate_names()
            for _ in range(tempfile.TMP_MAX):
                state["temporary"] = cache.directory / (key + "." + next(names) + ".tmp")
                state["audit_error"] = None
                sys.audit("tempfile.mkstemp", str(state["temporary"]))
                sys.audit("open", str(state["temporary"]), None, tempfile._bin_openflags | os.O_NOINHERIT)
                pointer = ctypes.c_char_p(body)
                code = self.function(str(state["temporary"]), str(state["target"]), pointer, len(body), self.callback,
                                     state["output"].ctypes.data, 5)
                error, created, flushed, closed, written = (int(value) for value in state["output"])
                with self.lock:
                    self.create_attempts += 1
                    self.actual_flushes += flushed
                    self.closed_handles += closed
                    self.bytes_written += written
                if code == 1 and (error in (80, 183) or error == 5 and cache.directory.is_dir() and os.access(cache.directory, os.W_OK)):
                    continue
                if state["audit_error"] is not None:
                    raise state["audit_error"]
                if code:
                    if not error:
                        raise RuntimeError("The owned native cache writer failed with code " + str(code))
                    translated = ctypes.WinError(error)
                    source = str(state["temporary"]) if code in (1, 5) else None
                    target = str(state["target"]) if code == 5 else None
                    if code == 5:
                        raise OSError(translated.errno, translated.strerror, source, error, target)
                    # The original CRT-backed open/write/fsync/close report errno only.
                    raise OSError(translated.errno, os.strerror(translated.errno), source)
                if (created, flushed, closed, written) != (1, 1, 1, len(body)):
                    raise ValueError("The actual native write, flush or handle close was incomplete")
                with self.lock:
                    self.native_calls += 1
                return
            raise FileExistsError(errno.EEXIST, "No usable temporary file name found")
        finally:
            if state is not None:
                state["temporary"] = state["target"] = state["audit_error"] = None
            with self.lock:
                self.active_calls -= 1

    def receipt(self):
        return {"closed": self.closed, "DLL_unloaded": self.dll is None, "active_calls": self.active_calls,
            "native_calls": self.native_calls, "fallback_calls": self.fallback_calls, "create_attempts": self.create_attempts,
            "actual_flushes": self.actual_flushes, "closed_handles": self.closed_handles, "bytes_written": self.bytes_written,
            "private_states": len(self.states), "retained_output_bytes": sum(state["output"].nbytes for state in self.states
                if state["output"] is not None)}

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active_calls:
                raise RuntimeError("The native cache DLL cannot unload before callers drain")
            import _ctypes
            self.function = None
            self.callback = None
            _ctypes.FreeLibrary(self.dll._handle)
            self.dll._handle = 0
            self.dll = None
            for state in self.states:
                state["output"] = None
            self.closed = True


@contextmanager
def native_quality_cache_execution(library):
    import quality_engine
    started = time.perf_counter()
    original = quality_engine.ScoreCache.write
    source = inspect.getsource(original)
    begin = source.index("        # Unique temporary names")
    if source.count("        temporary.replace(path)") != 1:
        raise ValueError("The original native cache lifecycle anchor changed")
    modified = source[:begin] + "        _private_cache_writer.publish_record(self, key, record)\n"
    namespace = dict(original.__globals__)
    engine = NativeQualityCacheWriter(library)
    namespace["_private_cache_writer"] = engine
    try:
        exec(compile("class OwnedCache:\n" + modified, "<owned_native_quality_cache_writer>", "exec"), namespace)
    except BaseException:
        engine.close()
        raise
    native = namespace["OwnedCache"].write

    def write(cache, key, scores):
        if engine.closed:
            raise RuntimeError("A closed native cache interface cannot execute")
        if engine.supported(cache, key, scores):
            return native(cache, key, scores)
        with engine.lock:
            engine.fallback_calls += 1
        return original(cache, key, scores)

    quality_engine.ScoreCache.write = write
    trace = {"setup_seconds": time.perf_counter() - started, "counted": False}
    try:
        yield trace
    finally:
        before = time.perf_counter()
        changed = quality_engine.ScoreCache.write is not write
        quality_engine.ScoreCache.write = original
        engine.close()
        trace.update(original_write_restored=quality_engine.ScoreCache.write is original, engine=engine.receipt(),
                     restore_seconds=time.perf_counter() - before)
        if changed:
            raise RuntimeError("Another writer changed the private cache-write owner")
