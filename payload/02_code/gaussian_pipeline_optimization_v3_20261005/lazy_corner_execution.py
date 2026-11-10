"""Own only the private candidate ordering factory until original callers drain."""

from contextlib import contextmanager

import planar_corner
from lazy_corner_order import LazyCornerWorkspace


@contextmanager
def lazy_corner_execution(library, enabled):
    if type(enabled) is not bool:
        raise TypeError("The private corner ordering route must be explicit")
    original = planar_corner.PlanarCornerWorkspace
    owned = []
    trace = {"enabled": enabled, "workspaces": [], "original_factory_restored": False,
        "only_candidate_order_changed": True, "new_optimization_count": 0}

    def workspace(response_library):
        value = LazyCornerWorkspace(response_library, library)
        owned.append(value)
        return value

    if enabled:
        planar_corner.PlanarCornerWorkspace = workspace
    try:
        yield trace
    finally:
        expected = workspace if enabled else original
        changed = planar_corner.PlanarCornerWorkspace is not expected
        planar_corner.PlanarCornerWorkspace = original
        trace["original_factory_restored"] = planar_corner.PlanarCornerWorkspace is original
        trace["workspaces"] = [dict(value.receipt(), priority_DLL_unloaded=value.order_dll is None) for value in owned]
        if changed or any(not row["closed"] or not row["priority_DLL_unloaded"] or row["active_calls"] for row in trace["workspaces"]):
            raise RuntimeError("The private corner ordering owner did not restore after all callers drained")
