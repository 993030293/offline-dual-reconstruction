"""Restore complete wire payloads before the unchanged native matcher reads them."""

import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys

from bounded_zstd_payload import ZstdPayloadCodec, MAXIMUM_RAW_BYTES

HEADER = struct.Struct("<I")


class DecodedInput:
    def __init__(self, original, codec):
        self.original, self.codec = original, codec
        self.phase, self.metadata_size = "header", 0
        self.metadata_remaining, self.metadata_parts = 0, []
        self.body_size = self.body_remaining = 0
        self.request = None
        self.raw_parts, self.rows = [], []

    def read(self, count):
        if self.phase == "header":
            value = self.original.read(count)
            if count == HEADER.size and len(value) == HEADER.size:
                self.metadata_size = HEADER.unpack(value)[0]
                self.metadata_remaining = self.metadata_size
                self.phase = "metadata"
            return value
        if self.phase == "metadata":
            if count != self.metadata_remaining or not 0 < count <= 4096:
                raise ValueError("The original complete native metadata read changed")
            value = self.original.read(count)
            self.metadata_parts.append(value)
            self.metadata_remaining -= len(value)
            if self.metadata_remaining:
                return value
            self.request = json.loads(b"".join(self.metadata_parts))
            self.metadata_parts.clear()
            self.phase = "header"
            if not isinstance(self.request, dict):
                return value
            command = self.request.get("command")
            if command == "batch":
                nq, counts = self.request.get("query_rows"), self.request.get("train_counts")
                if (isinstance(nq, int) and 1 <= nq <= 12000 and type(counts) is list
                        and 1 <= len(counts) <= 12 and all(type(n) is int and 2 <= n <= 12000 for n in counts)):
                    self.body_size = (nq + sum(counts)) * 128
                    self.body_remaining = self.body_size
                    self.phase = "body"
            elif command == "nearest":
                nq, nt = self.request.get("query_rows"), self.request.get("train_rows")
                if isinstance(nq, int) and isinstance(nt, int) and 1 <= nq <= 12000 and 2 <= nt <= 12000:
                    self.body_size = (nq + nt) * 128
                    self.body_remaining = self.body_size
                    self.phase = "body"
            return value
        codec_name = self.request.get("wire_codec")
        if codec_name is None:
            value = self.original.read(count)
            self.raw_parts.append(value)
            self.body_remaining -= len(value)
            if self.body_remaining == 0:
                raw = b"".join(self.raw_parts)
                self.raw_parts.clear()
                self.rows.append({"codec": "raw", "raw_bytes": len(raw), "wire_payload_bytes": len(raw),
                                  "raw_sha256": hashlib.sha256(raw).hexdigest()})
                self.phase = "header"
            return value
        if codec_name != "zstd1" or count != self.body_size or self.body_size > MAXIMUM_RAW_BYTES:
            raise ValueError("An unsupported complete wire payload was requested")
        encoded_size = self.request.get("wire_payload_bytes")
        if type(encoded_size) is not int or not 0 < encoded_size <= self.body_size:
            raise ValueError("The encoded complete input byte count changed")
        encoded = self.original.read(encoded_size)
        if len(encoded) != encoded_size:
            raise EOFError("A complete owned matching request was truncated")
        raw = self.codec.decode(encoded, self.body_size)
        self.rows.append({"codec": "zstd1", "raw_bytes": len(raw), "wire_payload_bytes": len(encoded),
                          "raw_sha256": hashlib.sha256(raw).hexdigest()})
        self.phase = "header"
        return raw


class InputFacade:
    def __init__(self, original, buffer):
        self.original, self.buffer = original, buffer

    def __getattr__(self, name):
        return getattr(self.original, name)


def main():
    original = sys.stdin
    codec = ZstdPayloadCodec()
    decoded = DecodedInput(original.buffer, codec)
    sys.stdin = InputFacade(original, decoded)
    try:
        runpy.run_path(str(Path(__file__).with_name("integer_sift_worker.py")), run_name="__main__")
    finally:
        sys.stdin = original
        print("OWNED_WIRE_TRACE " + json.dumps({"rows": decoded.rows, "native_codec_version": codec.version,
              "native_codec_library": codec.library_name, "stdin_alias_restored": sys.stdin is original}),
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
