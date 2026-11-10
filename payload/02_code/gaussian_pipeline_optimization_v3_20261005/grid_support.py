"""Count the original clipped image grid using bounded exact integer IDs."""

from contextlib import contextmanager
import inspect
import time

import numpy as np


GRID_SOURCE = """        cells = {
            (
                min(columns - 1, int(columns * x / float(camera.width))),
                min(rows - 1, int(rows * y / float(camera.height))),
            )
            for x, y in clipped
            if x < camera.width and y < camera.height
        }
        occupied_cells = len(cells)"""


def exact_grid_count(clipped, width, height, rows, columns):
    # Preserve Python conversion and overflow behavior outside safe integer IDs.
    safe = (0 < rows <= 2**31-1 and 0 < columns <= 2**31-1
            and 0 < float(width) and 0 < float(height)
            and np.isfinite(float(width)*columns)
            and np.isfinite(float(height)*rows))
    if not safe:
        return len({(min(columns-1, int(columns*x/float(width))),
                     min(rows-1, int(rows*y/float(height))))
                    for x, y in clipped if x < width and y < height})
    selected = clipped[(clipped[:, 0] < width) & (clipped[:, 1] < height)]
    x = np.minimum(columns-1, (columns*selected[:, 0]/float(width)).astype(np.int64))
    y = np.minimum(rows-1, (rows*selected[:, 1]/float(height)).astype(np.int64))
    return int(np.unique(y*columns+x).size)


@contextmanager
def grid_support_execution(module, enabled):
    original = module.camera_geometry_metrics
    started = time.perf_counter()
    source = inspect.getsource(original)
    if source.count(GRID_SOURCE) != 1:
        raise ValueError("The original complete camera grid support block changed")
    candidate = original
    if enabled:
        namespace = dict(original.__globals__)
        namespace["_exact_grid_count"] = exact_grid_count
        modified = source.replace(GRID_SOURCE,
            "        occupied_cells = _exact_grid_count(clipped, camera.width, camera.height, rows, columns)")
        exec(compile(modified, "<exact_complete_camera_grid_support>", "exec"), namespace)
        candidate = namespace[original.__name__]
    trace = {"enabled": enabled, "setup_seconds": time.perf_counter()-started,
             "calls": [], "counted": False}

    def metrics(reconstruction, image, policy):
        call_started = time.perf_counter()
        row = {"image_name": image.name, "status": "running", "candidate": enabled}
        trace["calls"].append(row)
        try:
            result = candidate(reconstruction, image, policy)
            row["status"] = "passed_original_full_camera_support_function"
            return result
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            row["complete_call_seconds"] = time.perf_counter()-call_started

    module.camera_geometry_metrics = metrics
    try:
        yield trace
    finally:
        restore_started = time.perf_counter()
        module.camera_geometry_metrics = original
        trace["restore_seconds"] = time.perf_counter()-restore_started
        trace["original_binding_restored"] = module.camera_geometry_metrics is original
