"""Count successful native source-image read opens across a complete entry."""

import builtins
from contextlib import contextmanager
import io
import os
from pathlib import Path
import threading


@contextmanager
def image_source_open_census(output, dataset):
    from runtime import atomic_json

    paths = {str(path) for path in (Path(dataset) / "images").rglob("*") if path.is_file()}
    native_builtin, native_io = builtins.open, io.open
    lock = threading.Lock()
    rows = {}
    failed = []

    def counted(native, api, fp, *args, **kwargs):
        name = os.fspath(fp) if isinstance(fp, (str, os.PathLike)) else None
        mode = args[0] if args else kwargs.get("mode", "r")
        registered = name in paths and "r" in mode
        try:
            result = native(fp, *args, **kwargs)
        except BaseException as error:
            if registered:
                with lock:
                    failed.append({"path": name, "api": api, "error_type": type(error).__name__, "message": str(error)})
            raise
        if registered:
            key = (name, api, threading.current_thread().name)
            with lock:
                rows[key] = rows.get(key, 0) + 1
        return result

    builtin_wrapper = lambda fp, *args, **kwargs: counted(native_builtin, "builtins.open", fp, *args, **kwargs)
    io_wrapper = lambda fp, *args, **kwargs: counted(native_io, "io.open", fp, *args, **kwargs)
    builtins.open, io.open = builtin_wrapper, io_wrapper
    succeeded = False
    try:
        yield rows
        succeeded = True
    finally:
        bindings_unchanged = builtins.open is builtin_wrapper and io.open is io_wrapper
        builtins.open, io.open = native_builtin, native_io
        atomic_json(Path(output) / "SOURCE_IMAGE_OPEN_WORK.json", {
            "status": "complete" if succeeded else "original_failure_preserved",
            "registered_source_images": len(paths), "successful_native_read_opens": sum(rows.values()),
            "rows": [{"path": name, "api": api, "thread": thread, "opens": count} for (name, api, thread), count in sorted(rows.items())],
            "failed_open_attempts": failed, "native_functions_restored": bindings_unchanged and builtins.open is native_builtin and io.open is native_io,
            "actual_native_call_count_not_Image_open_estimate": True, "new_optimization_count": 0})
