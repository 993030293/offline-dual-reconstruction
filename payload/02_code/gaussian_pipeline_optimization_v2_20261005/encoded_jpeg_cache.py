"""Reuse exact OpenCV JPEG artifacts by pixel content and encoder identity."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import uuid

import cv2
import numpy as np


MAGIC = b"JPGC0001"
ENTRY_LIMIT = 32*1024**2


def digest(value):
    return hashlib.sha256(value).hexdigest()


class EncodedJpegCache:
    """Serialize cache ownership; publish each complete artifact atomically."""

    def __init__(self, directory, maximum_bytes=2*1024**3):
        if not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
            raise ValueError("A positive cache byte limit is required")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.maximum_bytes = maximum_bytes
        self.lock = self.directory/"OWNER.json"
        self.token = uuid.uuid4().hex
        self.native = cv2.imencode
        self.started = time.monotonic()
        self.closed = True
        owner = {"pid": os.getpid(), "token": self.token, "created_ns": time.time_ns()}
        with self.lock.open("x", encoding="utf-8") as stream:
            json.dump(owner, stream)
        self.closed = False
        try:
            self.bytes = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file() and path != self.lock)
            if self.bytes > maximum_bytes:
                raise MemoryError("The existing cache exceeds its declared disk budget")
            self.contract = {"schema": 1, "opencv": cv2.__version__, "numpy": np.__version__,
                "opencv_build_sha256": digest(cv2.getBuildInformation().encode()),
                "implementation_sha256": digest(Path(__file__).read_bytes())}
            self.version = json.dumps(self.contract, sort_keys=True, separators=(",", ":")).encode()
            self.hits = self.misses = self.calls = 0
            self.key_seconds = self.read_seconds = self.encode_seconds = self.publish_seconds = 0.
        except BaseException:
            self.close()
            raise

    def encode(self, ext, image, params=None):
        if self.closed:
            raise RuntimeError("The JPEG cache is closed")
        if (ext != ".jpg" or not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
            raise ValueError("This exact export cache supports nonempty uint8 BGR JPEG inputs only")
        params = [] if params is None else list(params)
        if len(params) % 2 or not all(isinstance(value, int) and not isinstance(value, bool) for value in params):
            raise ValueError("Encoder parameters must be integer key/value pairs")
        if image.nbytes > ENTRY_LIMIT:
            raise MemoryError("The source image exceeds the bounded export contract")
        self.calls += 1
        started = time.perf_counter()
        prefix = json.dumps({"ext": ext, "shape": image.shape, "dtype": image.dtype.str,
            "params": params}, sort_keys=True, separators=(",", ":")).encode()
        hasher = hashlib.sha256(self.version+prefix)
        hasher.update(memoryview(np.ascontiguousarray(image)).cast("B"))
        key = hasher.hexdigest()
        path = self.directory/(key+".jpgc")
        self.key_seconds += time.perf_counter()-started
        started = time.perf_counter()
        if path.exists():
            if path.stat().st_size > ENTRY_LIMIT or path.stat().st_size < 12:
                raise ValueError("Invalid committed JPEG cache entry length")
            payload = path.read_bytes()
            length = struct.unpack_from("<I", payload, 8)[0]
            if payload[:8] != MAGIC or length > 4096 or 12+length >= len(payload):
                raise ValueError("Invalid committed JPEG cache envelope")
            record = json.loads(payload[12:12+length].decode("utf-8"))
            encoded = payload[12+length:]
            if record != {"key": key, "jpeg_bytes": len(encoded), "jpeg_sha256": digest(encoded)}:
                raise ValueError("Committed JPEG content or key differs from its cache record")
            self.hits += 1
            self.read_seconds += time.perf_counter()-started
            return True, np.frombuffer(encoded, dtype=np.uint8).copy()
        self.read_seconds += time.perf_counter()-started
        self.misses += 1
        started = time.perf_counter()
        ok, encoded = self.native(ext, image, params)
        self.encode_seconds += time.perf_counter()-started
        if not ok:
            raise RuntimeError("The native JPEG encoder failed")
        body = encoded.tobytes()
        header = json.dumps({"key": key, "jpeg_bytes": len(body), "jpeg_sha256": digest(body)},
            sort_keys=True, separators=(",", ":")).encode()
        payload = MAGIC+struct.pack("<I", len(header))+header+body
        if len(payload) > ENTRY_LIMIT or self.bytes+len(payload) > self.maximum_bytes:
            raise MemoryError("Publishing the JPEG would exceed the cache budget")
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, prefix=key+".", suffix=".tmp", delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
        self.bytes += len(payload)
        self.publish_seconds += time.perf_counter()-started
        return True, encoded

    def receipt(self):
        return {"contract": self.contract, "calls": self.calls, "hits": self.hits, "misses": self.misses,
            "cache_bytes": self.bytes, "maximum_bytes": self.maximum_bytes, "key_seconds": self.key_seconds,
            "read_seconds": self.read_seconds, "native_encode_seconds": self.encode_seconds,
            "publish_seconds": self.publish_seconds, "closed": self.closed,
            "scope": "exact JPEG artifacts; no quality scores, decisions, poses, or feature reuse"}

    def close(self):
        if self.closed:
            return
        owner = json.loads(self.lock.read_text(encoding="utf-8"))
        if owner["token"] != self.token or owner["pid"] != os.getpid():
            raise RuntimeError("JPEG cache ownership changed; refusing to release another owner")
        self.lock.unlink()
        self.closed = True


@contextmanager
def cached_jpeg_encoding(directory, maximum_bytes=2*1024**3):
    cache = EncodedJpegCache(directory, maximum_bytes)
    original = cv2.imencode
    try:
        cv2.imencode = cache.encode
        yield cache
    finally:
        cv2.imencode = original
        cache.close()
