"""Meter actual writes and optionally encode each complete native request."""

import hashlib
import inspect
from pathlib import Path
import threading
import time

import metric_branch_localization_003 as native
from bounded_zstd_payload import ZstdPayloadCodec, MAXIMUM_RAW_BYTES


class PipeWriteMeter:
    def __init__(self, original):
        self.original, self.written_bytes = original, 0

    def write(self, value):
        count = self.original.write(value)
        if count is not None:
            if type(count) is not int or not 0 <= count <= len(value):
                raise ValueError("The actual native write returned an invalid byte count")
            self.written_bytes += count
        return count

    def __getattr__(self, name):
        return getattr(self.original, name)


def constructor():
    import textwrap
    source = textwrap.dedent(inspect.getsource(native.ObservedWorker.__init__))
    before = 'CODE/"integer_sift_worker.py"'
    if source.count(before) != 1:
        raise ValueError("The original owned worker initialization changed")
    source = source.replace(before, 'CODE/"sift_payload_input_worker.py"')
    namespace = dict(native.__dict__)
    namespace["RESOURCE_WRAPPER"] = Path(__file__).with_name("codec_owned_matching_worker.py")
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["__init__"]


class CodecObservedWorker(native.ObservedWorker):
    def __init__(self, log, enabled=False):
        self.codec_enabled = enabled
        self.codec = ZstdPayloadCodec() if enabled else None
        self.wire = []
        self.written_input_bytes = 0
        (constructor() if enabled else native.ObservedWorker.__init__)(self, log)
        self.receipt["wire"] = self.wire
        self.receipt["actual_written_input_bytes"] = self.written_input_bytes

    def rpc(self, record, payload=b""):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the original caller owns the matching pipe")
        original_record, original_payload = record, payload
        codec_name, encode_seconds = "raw", 0.
        if self.codec_enabled and record.get("command") == "batch" and 0 < len(payload) <= MAXIMUM_RAW_BYTES:
            start = time.perf_counter()
            encoded = self.codec.encode(payload)
            encode_seconds = time.perf_counter() - start
            if len(encoded) < len(payload):
                record = dict(record, wire_codec="zstd1", wire_payload_bytes=len(encoded))
                payload, codec_name = encoded, "zstd1"
        pipe = self.process.stdin
        meter = PipeWriteMeter(pipe)
        self.process.stdin = meter
        row = {"command": original_record.get("command"), "raw_payload_bytes": len(original_payload),
               "transmitted_payload_bytes": len(payload), "codec": codec_name,
               "raw_sha256": hashlib.sha256(original_payload).hexdigest(), "encode_seconds": encode_seconds,
               "status": "running"}
        try:
            result = native.ObservedWorker.rpc(self, record, payload)
            row["status"] = "completed"
            return result
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if self.process.stdin is not meter:
                raise RuntimeError("The private input write alias changed before restoration")
            self.process.stdin = pipe
            row["actual_written_input_bytes"] = meter.written_bytes
            self.written_input_bytes += meter.written_bytes
            self.wire.append(row)
            self.receipt["actual_written_input_bytes"] = self.written_input_bytes
            self.receipt["wire"] = self.wire


def read_server_trace(log):
    rows = [line.removeprefix("OWNED_WIRE_TRACE ") for line in Path(log).read_text(encoding="utf-8").splitlines()
            if line.startswith("OWNED_WIRE_TRACE ")]
    if len(rows) != 1:
        raise ValueError("Exactly one real native decoder terminal trace is required")
    import json
    return json.loads(rows[0])
