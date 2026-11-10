"""Use the existing native library for one strictly bounded complete frame."""

import ctypes
import ctypes.util
from pathlib import Path
import sys

MAXIMUM_RAW_BYTES = 8 * 1024**2


class ZstdPayloadCodec:
    def __init__(self):
        root = Path(__file__).resolve().parents[2]
        if sys.platform == "win32":
            library = str(root / "05_envs/colmap-4.1.1-cuda/bin/zstd.dll")
            if not Path(library).is_file():
                raise ImportError("The existing explicit COLMAP Zstd library is missing")
        else:
            library = ctypes.util.find_library("zstd")
            if library is None:
                raise ImportError("The existing system Zstd library is missing")
        self.library = ctypes.CDLL(library)
        signatures = {"ZSTD_compressBound": ([ctypes.c_size_t], ctypes.c_size_t),
            "ZSTD_compress": ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_size_t),
            "ZSTD_decompress": ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
            "ZSTD_findFrameCompressedSize": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
            "ZSTD_getFrameContentSize": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ulonglong),
            "ZSTD_isError": ([ctypes.c_size_t], ctypes.c_uint),
            "ZSTD_getErrorName": ([ctypes.c_size_t], ctypes.c_char_p),
            "ZSTD_versionNumber": ([], ctypes.c_uint)}
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, result
        self.version = int(self.library.ZSTD_versionNumber())
        self.library_name = library

    def checked(self, size):
        if self.library.ZSTD_isError(size):
            raise ValueError("The existing native payload codec rejected its input: "
                             + self.library.ZSTD_getErrorName(size).decode())
        return int(size)

    def encode(self, raw, level=1):
        if type(raw) is not bytes or not 0 < len(raw) <= MAXIMUM_RAW_BYTES or level != 1:
            raise ValueError("The registered complete native payload bound or level changed")
        buffer = ctypes.create_string_buffer(self.library.ZSTD_compressBound(len(raw)))
        count = self.checked(self.library.ZSTD_compress(buffer, len(buffer), raw, len(raw), level))
        return buffer.raw[:count]

    def decode(self, encoded, expected):
        if (type(encoded) is not bytes or type(expected) is not int or not 0 < expected <= MAXIMUM_RAW_BYTES
                or not 0 < len(encoded) <= self.library.ZSTD_compressBound(MAXIMUM_RAW_BYTES)):
            raise ValueError("A native request exceeds its complete allocation bound")
        frame_bytes = self.checked(self.library.ZSTD_findFrameCompressedSize(encoded, len(encoded)))
        content_bytes = int(self.library.ZSTD_getFrameContentSize(encoded, len(encoded)))
        if frame_bytes != len(encoded) or content_bytes != expected:
            raise ValueError("The single complete frame has trailing bytes or a wrong content size")
        buffer = ctypes.create_string_buffer(expected)
        count = self.checked(self.library.ZSTD_decompress(buffer, expected, encoded, len(encoded)))
        if count != expected:
            raise ValueError("The full native request decoded to a wrong byte count")
        return buffer.raw
