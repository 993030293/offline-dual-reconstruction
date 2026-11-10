"""Own a chosen exact-key selector while retaining the original response owner."""

from contextlib import contextmanager
import ctypes
from pathlib import Path

import planar_corner


class KeyCornerWorkspace(planar_corner.PlanarCornerWorkspace):
    def __init__(self, response_library, order_library, mode):
        if mode not in ("packed", "radix"):
            raise ValueError("The exact-key selector must be explicitly chosen")
        super().__init__(response_library)
        self.order_dll = None
        self.order_mode = mode
        try:
            self.order_dll = ctypes.CDLL(str(Path(order_library).resolve(strict=True)))
            abi = self.order_dll.exact_key_corner_abi
            abi.argtypes, abi.restype = [], ctypes.c_int
            if abi() != 2:
                raise ValueError("The private exact-key ABI changed")
            function = getattr(self.order_dll, mode + "_corner_select")
            function.argtypes, function.restype = self.select.argtypes, self.select.restype
            self.select = function
        except BaseException:
            self.close()
            raise

    def close(self):
        super().close()
        if self.order_dll is not None:
            import _ctypes
            _ctypes.FreeLibrary(self.order_dll._handle)
            self.order_dll._handle = 0
            self.order_dll = None


@contextmanager
def key_corner_execution(library, mode, enabled):
    if mode not in ("packed", "radix") or type(enabled) is not bool:
        raise ValueError("The exact-key selector and explicit enabled flag are required")
    original = planar_corner.PlanarCornerWorkspace
    owned = []
    trace = {"enabled": enabled, "mode": mode, "workspaces": [], "original_factory_restored": False,
        "only_candidate_order_changed": True, "new_optimization_count": 0}

    def workspace(response_library):
        value = KeyCornerWorkspace(response_library, library, mode)
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
        trace["workspaces"] = [dict(value.receipt(), mode=value.order_mode, priority_DLL_unloaded=value.order_dll is None) for value in owned]
        if changed or any(not row["closed"] or not row["priority_DLL_unloaded"] or row["active_calls"] for row in trace["workspaces"]):
            raise RuntimeError("The exact-key owner did not restore after callers drained")
