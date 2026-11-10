"""Encode complete native checkpoints with a bounded existing-system zstd codec."""

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

MAGIC = b"E26ZS01\n"
RECORD = struct.Struct("<II")
BLOCK = 8*1024**2
LIMIT = 4*1024**3
CODEC = "native_complete_torch_zstd_blocks_level3_v1"


class Zstd:
    def __init__(self):
        library = ctypes.util.find_library("zstd")
        if library is None:
            raise ImportError("Use only the existing system zstd library, never install a fallback")
        self.library = ctypes.CDLL(library)
        signatures = {"ZSTD_compressBound": ([ctypes.c_size_t], ctypes.c_size_t),
            "ZSTD_compress": ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_size_t),
            "ZSTD_decompress": ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
            "ZSTD_isError": ([ctypes.c_size_t], ctypes.c_uint), "ZSTD_getErrorName": ([ctypes.c_size_t], ctypes.c_char_p),
            "ZSTD_versionNumber": ([], ctypes.c_uint)}
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, result
        self.version = int(self.library.ZSTD_versionNumber())

    def checked(self, size):
        if self.library.ZSTD_isError(size):
            raise ValueError("Existing zstd codec rejected its payload: "+self.library.ZSTD_getErrorName(size).decode())
        return int(size)

    def compress(self, raw):
        if not 0 < len(raw) <= BLOCK:
            raise ValueError("A native codec block is outside its bounded admission")
        target = ctypes.create_string_buffer(self.library.ZSTD_compressBound(len(raw)))
        size = self.checked(self.library.ZSTD_compress(target, len(target), raw, len(raw), 3))
        return target.raw[:size]

    def decompress(self, raw, size):
        if not 0 < size <= BLOCK or not 0 < len(raw) <= self.library.ZSTD_compressBound(BLOCK):
            raise ValueError("A native compressed record exceeds its explicit allocation bound")
        target = ctypes.create_string_buffer(size)
        count = self.checked(self.library.ZSTD_decompress(target, size, raw, len(raw)))
        if count != size:
            raise ValueError("A complete native block decompressed to the wrong byte count")
        return target.raw


def encode_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix(target.suffix+".json").exists() or not 0 < source.stat().st_size <= LIMIT:
        raise ValueError("The zstd transport requires a unique bounded native checkpoint")
    identity = data_identity(source)
    before = time.perf_counter()
    codec = Zstd()
    records = 0
    with source.open("rb") as original, target.open("xb") as output:
        output.write(MAGIC)
        while raw := original.read(BLOCK):
            encoded = codec.compress(raw)
            output.write(RECORD.pack(len(raw), len(encoded)))
            output.write(encoded)
            records += 1
        output.flush()
    if data_identity(source) != identity:
        raise ValueError("The immutable native checkpoint changed during compression")
    native_receipt = source.with_suffix(".json")
    receipt = {"schema_version": 1, "codec": CODEC, "archive": data_identity(target), "native_payload": identity,
        "native_receipt": json.loads(native_receipt.read_text()) if native_receipt.exists() else None,
        "decoder_source": data_identity(Path(__file__)), "system_zstd_version": codec.version,
        "compression_seconds": time.perf_counter()-before, "block_bytes": BLOCK, "records": records,
        "maximum_native_payload_bytes": LIMIT, "complete_native_state_no_quantization": True}
    atomic_json(target.with_suffix(target.suffix+".json"), receipt)
    return receipt


def verified_expand(path, target):
    path, target = Path(path), Path(target)
    record = json.loads(path.with_suffix(path.suffix+".json").read_text())
    codec = Zstd()
    if (record.get("schema_version") != 1 or record.get("codec") != CODEC or record.get("decoder_source") != data_identity(Path(__file__))
            or record.get("system_zstd_version") != codec.version or record.get("maximum_native_payload_bytes") != LIMIT
            or record.get("block_bytes") != BLOCK or not 0 < record["native_payload"]["bytes"] <= LIMIT
            or target.exists() or data_identity(path) != record["archive"]):
        raise ValueError("The archive, decoder, native size or unique expansion ownership changed")
    started = time.perf_counter()
    count, records, digest = 0, 0, hashlib.sha256()
    with path.open("rb") as source, target.open("xb") as output:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("The complete checkpoint codec magic changed")
        while header := source.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError("A complete checkpoint record header is truncated")
            native, compressed = RECORD.unpack(header)
            if not 0 < native <= BLOCK or not 0 < compressed <= codec.library.ZSTD_compressBound(BLOCK):
                raise ValueError("An archive record exceeds the declared byte bounds")
            if count+native > record["native_payload"]["bytes"]:
                raise ValueError("The archive expanded beyond its frozen complete state")
            encoded = source.read(compressed)
            if len(encoded) != compressed:
                raise ValueError("A complete compressed state record is truncated")
            raw = codec.decompress(encoded, native)
            output.write(raw)
            digest.update(raw)
            count, records = count+native, records+1
    if count != record["native_payload"]["bytes"] or digest.hexdigest() != record["native_payload"]["sha256"] or records != record["records"]:
        raise ValueError("The complete decoded native state differs from its frozen identity")
    expanded = data_identity(target)
    native = dict(record["native_payload"], path=str(target))
    if expanded != native:
        raise ValueError("The expanded file differs from the verified complete state")
    return {"native_payload": record["native_payload"], "expanded": expanded,
        "verification_and_expansion_seconds": time.perf_counter()-started, "records": records,
        "temporary_native_disk_bytes": count, "bounded_decode_block_bytes": BLOCK, "native_complete_loader_unchanged": True}
