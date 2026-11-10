"""Use native one-shot JSON encoding for an explicitly machine-consumed path."""

from contextlib import contextmanager
import json
from pathlib import Path
import struct
import time


def compact_atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"),
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def exact_json_values(first, second):
    if type(first) is not type(second):
        raise AssertionError("A decoded JSON leaf type changed")
    if isinstance(first, dict):
        if list(first) != list(second):
            raise AssertionError("A decoded JSON key order changed")
        for key in first:
            exact_json_values(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError("A decoded JSON sequence length changed")
        for left, right in zip(first, second, strict=True):
            exact_json_values(left, right)
    elif isinstance(first, float):
        if struct.pack("!d", first) != struct.pack("!d", second):
            raise AssertionError("A decoded JSON floating-point bit changed")
    elif first != second:
        raise AssertionError("A decoded JSON scalar changed")


@contextmanager
def machine_json_execution(module, enabled=True):
    started = time.perf_counter()
    original = module.atomic_json
    trace = {"calls": [], "compact_calls": 0, "native_calls": 0}

    def write(path, value):
        begin = time.perf_counter()
        path = Path(path)
        chosen = compact_atomic_json if enabled and path.name == "camera_path.json" else original
        chosen(path, value)
        write_seconds = time.perf_counter() - begin
        before = time.perf_counter()
        decoded = json.loads(path.read_text(encoding="utf-8"))
        exact_json_values(value, decoded)
        trace["compact_calls" if chosen is compact_atomic_json else "native_calls"] += 1
        trace["calls"].append({"path_name": path.name, "write_seconds": write_seconds,
            "complete_validation_seconds": time.perf_counter() - before,
            "output_bytes": path.stat().st_size,
            "all_decoded_types_values_order_and_float_bits_exact": True})

    module.atomic_json = write
    trace["setup_seconds"] = time.perf_counter() - started
    try:
        yield trace
    finally:
        begin = time.perf_counter()
        module.atomic_json = original
        trace["restore_seconds"] = time.perf_counter() - begin
        trace["original_binding_restored"] = module.atomic_json is original
