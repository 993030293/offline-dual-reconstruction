"""Transport complete native bytes with the installed bounded LZ4 block API."""

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

MAGIC = b"E26L401\n"
RECORD = struct.Struct("<II")
BLOCK = 8 * 1024**2
LIMIT = 4 * 1024**3
CODEC = "native_complete_torch_lz4_default_blocks_v1"


class Lz4:
    def __init__(self):
        name = ctypes.util.find_library("lz4")
        if name is None:
            raise ImportError("The existing LZ4 system library is required")
        self.library = ctypes.CDLL(name)
        signatures = {
            "LZ4_versionNumber": ([], ctypes.c_int),
            "LZ4_compressBound": ([ctypes.c_int], ctypes.c_int),
            "LZ4_compress_default": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "LZ4_decompress_safe": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        }
        for key, (arguments, result) in signatures.items():
            method = getattr(self.library, key)
            method.argtypes, method.restype = arguments, result
        self.version = int(self.library.LZ4_versionNumber())
        if self.version != 10902:
            raise ValueError("Only the currently verified existing LZ4 version is admitted")

    def bound(self, size):
        if not 0 < size <= BLOCK:
            raise ValueError("A complete native block exceeds its allocation bound")
        result = int(self.library.LZ4_compressBound(size))
        if result <= 0:
            raise ValueError("The existing LZ4 library rejected the block bound")
        return result

    def compress(self, raw):
        target = ctypes.create_string_buffer(self.bound(len(raw)))
        count = int(self.library.LZ4_compress_default(raw, target, len(raw), len(target)))
        if not 0 < count <= len(target):
            raise ValueError("The existing LZ4 library rejected compression")
        return target.raw[:count]

    def decompress(self, raw, size):
        if not 0 < size <= BLOCK or not 0 < len(raw) <= self.bound(BLOCK):
            raise ValueError("An encoded block exceeds its frozen bounds")
        target = ctypes.create_string_buffer(size)
        count = int(self.library.LZ4_decompress_safe(raw, target, len(raw), size))
        if count != size:
            raise ValueError("The complete LZ4 block is malformed or has a different size")
        return target.raw


def encode_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix(target.suffix + ".json").exists() or not 0 < source.stat().st_size <= LIMIT:
        raise ValueError("The transport requires a unique bounded complete native state")
    identity = data_identity(source)
    started = time.perf_counter()
    codec, records = Lz4(), 0
    with source.open("rb") as original, target.open("xb") as output:
        output.write(MAGIC)
        while raw := original.read(BLOCK):
            encoded = codec.compress(raw)
            output.write(RECORD.pack(len(raw), len(encoded)))
            output.write(encoded)
            records += 1
        output.flush()
    if data_identity(source) != identity:
        raise ValueError("The immutable complete native state changed during encoding")
    native_receipt = source.with_suffix(".json")
    receipt = {
        "schema_version": 1, "codec": CODEC, "archive": data_identity(target), "native_payload": identity,
        "native_receipt": json.loads(native_receipt.read_text()) if native_receipt.exists() else None,
        "decoder_source": data_identity(Path(__file__)), "system_lz4_version": codec.version,
        "compression_seconds": time.perf_counter() - started, "block_bytes": BLOCK, "records": records,
        "maximum_native_payload_bytes": LIMIT, "complete_native_state_no_quantization": True,
    }
    atomic_json(target.with_suffix(target.suffix + ".json"), receipt)
    return receipt


def verified_expand(path, target):
    path, target = Path(path), Path(target)
    record = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    codec = Lz4()
    if (record.get("schema_version") != 1 or record.get("codec") != CODEC
            or record.get("decoder_source") != data_identity(Path(__file__))
            or record.get("system_lz4_version") != codec.version
            or record.get("maximum_native_payload_bytes") != LIMIT or record.get("block_bytes") != BLOCK
            or not 0 < record["native_payload"]["bytes"] <= LIMIT or target.exists()
            or data_identity(path) != record["archive"]):
        raise ValueError("The archive, decoder, payload bound or output ownership changed")
    started = time.perf_counter()
    count, records, digest = 0, 0, hashlib.sha256()
    with path.open("rb") as source, target.open("xb") as output:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("The complete checkpoint codec magic changed")
        while header := source.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError("A complete checkpoint record header is truncated")
            native, compressed = RECORD.unpack(header)
            if not 0 < native <= BLOCK or not 0 < compressed <= codec.bound(BLOCK):
                raise ValueError("An archive record exceeds its declared bounds")
            if count + native > record["native_payload"]["bytes"]:
                raise ValueError("The archive expanded beyond its complete frozen state")
            encoded = source.read(compressed)
            if len(encoded) != compressed:
                raise ValueError("A complete encoded block is truncated")
            raw = codec.decompress(encoded, native)
            output.write(raw)
            digest.update(raw)
            count, records = count + native, records + 1
    if count != record["native_payload"]["bytes"] or records != record["records"] or digest.hexdigest() != record["native_payload"]["sha256"]:
        raise ValueError("The complete decoded native state differs from its frozen identity")
    expanded = data_identity(target)
    if expanded != dict(record["native_payload"], path=str(target)):
        raise ValueError("The expanded native state differs from the complete verified payload")
    return {
        "native_payload": record["native_payload"], "expanded": expanded,
        "verification_and_expansion_seconds": time.perf_counter() - started, "records": records,
        "temporary_native_disk_bytes": count, "bounded_decode_block_bytes": BLOCK,
        "native_complete_loader_unchanged": True,
    }
