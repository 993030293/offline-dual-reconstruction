"""Store complete native checkpoints as lossless bounded gzip transport artifacts."""

import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

LIMIT = 4*1024**3
CODEC = "native_torch_zip_gzip_level1_v1"


def encode_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix(target.suffix+".json").exists() or source.stat().st_size > LIMIT:
        raise ValueError("The native checkpoint archive requires a unique output within its declared bound")
    identity = data_identity(source)
    before = time.perf_counter()
    with source.open("rb") as original, target.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=1, mtime=0) as compressed:
            while block := original.read(4*1024**2):
                compressed.write(block)
        output.flush()
    if data_identity(source) != identity:
        raise ValueError("The immutable checkpoint source changed during archive creation")
    native_receipt = source.with_suffix(".json")
    receipt = {"schema_version": 1, "codec": CODEC, "archive": data_identity(target), "native_payload": identity,
        "native_receipt": json.loads(native_receipt.read_text()) if native_receipt.exists() else None,
        "decoder_source": data_identity(Path(__file__)), "compression_seconds": time.perf_counter()-before,
        "maximum_native_payload_bytes": LIMIT, "lossless_complete_state_no_quantization": True}
    atomic_json(target.with_suffix(target.suffix+".json"), receipt)
    return receipt


def verified_checkpoint_stream(path):
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix+".json").read_text())
    if (receipt.get("schema_version") != 1 or receipt.get("codec") != CODEC or
            receipt.get("decoder_source") != data_identity(Path(__file__)) or
            receipt.get("maximum_native_payload_bytes") != LIMIT):
        raise ValueError("The native checkpoint archive codec or decoder identity changed")
    if not 0 < receipt["native_payload"]["bytes"] <= LIMIT or data_identity(path) != receipt["archive"]:
        raise ValueError("The checkpoint archive identity or decoded byte bound changed")
    before = time.perf_counter()
    count, digest = 0, hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(4*1024**2):
            count += len(block)
            if count > receipt["native_payload"]["bytes"]:
                raise ValueError("The checkpoint archive expanded beyond its frozen native payload")
            digest.update(block)
    if count != receipt["native_payload"]["bytes"] or digest.hexdigest() != receipt["native_payload"]["sha256"]:
        raise ValueError("The complete decompressed checkpoint differs from its native archive")
    stream = gzip.open(path, "rb")
    return stream, {"verified_native_payload_bytes": count, "verification_seconds": time.perf_counter()-before,
        "native_payload_sha256": digest.hexdigest(), "no_uncompressed_disk_copy": True}


def load_checkpoint_archive(path, **kwargs):
    import torch
    stream, trace = verified_checkpoint_stream(path)
    before = time.perf_counter()
    try:
        payload = torch.load(stream, **kwargs)
    finally:
        stream.close()
    trace["torch_read_seconds"] = time.perf_counter()-before
    return payload, trace
