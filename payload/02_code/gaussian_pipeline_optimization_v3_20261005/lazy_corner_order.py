"""Replace only candidate ordering while retaining the accepted response owner."""

import ctypes
from pathlib import Path

from planar_corner import PlanarCornerWorkspace


class LazyCornerWorkspace(PlanarCornerWorkspace):
    def __init__(self, response_library, order_library):
        super().__init__(response_library)
        self.order_dll = None
        try:
            self.order_dll = ctypes.CDLL(str(Path(order_library).resolve(strict=True)))
            abi = self.order_dll.lazy_corner_order_abi
            abi.argtypes, abi.restype = [], ctypes.c_int
            if abi() != 1:
                raise ValueError("The private candidate priority ABI changed")
            function = self.order_dll.lazy_corner_select
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
