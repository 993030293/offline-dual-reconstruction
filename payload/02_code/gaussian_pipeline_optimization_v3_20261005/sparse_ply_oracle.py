"""Assemble official sparse PLY round-trip rows without per-point concatenations."""

from contextlib import contextmanager
import inspect
import time

import numpy as np


def imported_point_rows(points):
    return np.column_stack((np.asarray([p.xyz for p in points]),
                            np.asarray([p.color for p in points])))


@contextmanager
def sparse_ply_execution(native, enabled):
    original = native.write_initial_ply
    started = time.perf_counter()
    source = inspect.getsource(original)
    anchor = "    actual_rows = np.asarray([np.r_[p.xyz, p.color] for p in actual])"
    if source.count(anchor) != 1:
        raise ValueError("The exact official sparse PLY validation source changed")
    candidate = original
    if enabled:
        namespace = dict(original.__globals__)
        namespace["_imported_point_rows"] = imported_point_rows
        modified = source.replace(anchor, "    actual_rows = _imported_point_rows(actual)")
        exec(compile(modified, "<official_sparse_PLY_roundtrip_column_assembly>", "exec"), namespace)
        candidate = namespace[original.__name__]
    trace = {"enabled": enabled, "setup_seconds": time.perf_counter()-started,
             "calls": [], "counted": False}

    def writer(model, path):
        started_call = time.perf_counter()
        # Empty models retain the original rejection and partial-output behavior.
        use_candidate = enabled and model.num_points3D() > 0
        row = {"points": model.num_points3D(), "candidate": use_candidate,
               "path": str(path), "status": "running"}
        trace["calls"].append(row)
        try:
            result = (candidate if use_candidate else original)(model, path)
            row["status"] = "passed_complete_official_roundtrip"
            return result
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            row["complete_call_seconds"] = time.perf_counter()-started_call

    native.write_initial_ply = writer
    try:
        yield trace
    finally:
        started_restore = time.perf_counter()
        native.write_initial_ply = original
        trace["restore_seconds"] = time.perf_counter()-started_restore
        trace["original_binding_restored"] = native.write_initial_ply is original

