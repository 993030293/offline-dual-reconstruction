"""Keep native Zstd blocks exact while overlapping compression and one writer."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import zstd_checkpoint as native
from runtime import atomic_json, data_identity


def verified_expand(path, target):
    return native.verified_expand(path, target)


def encode_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix(target.suffix + ".json").exists() or not 0 < source.stat().st_size <= native.LIMIT:
        raise ValueError("The transport requires a unique bounded native checkpoint")
    identity = data_identity(source)
    started = time.perf_counter()
    codec, records, peak_pending = native.Zstd(), 0, 0
    workers, intervals, writes = set(), [], []
    worker_lock = threading.Lock()

    def compress(raw):
        before = time.perf_counter()
        with worker_lock:
            workers.add(threading.current_thread())
        value = codec.compress(raw)
        with worker_lock:
            intervals.append(dict(started=before, ended=time.perf_counter(), thread=threading.get_ident()))
        return value

    with source.open("rb") as original, target.open("xb") as output, ThreadPoolExecutor(max_workers=2, thread_name_prefix="r137_zstd") as pool:
        output.write(native.MAGIC)
        pending, exhausted = deque(), False
        while pending or not exhausted:
            while not exhausted and len(pending) < 4:
                raw = original.read(native.BLOCK)
                if not raw:
                    exhausted = True
                    break
                pending.append((len(raw), pool.submit(compress, raw)))
                peak_pending = max(peak_pending, len(pending))
            if pending:
                length, future = pending.popleft()
                encoded = future.result()
                before = time.perf_counter()
                output.write(native.RECORD.pack(length, len(encoded)))
                output.write(encoded)
                writes.append(dict(started=before, ended=time.perf_counter()))
                records += 1
        output.flush()
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError("An owned compression worker survived complete join")
    if data_identity(source) != identity:
        raise ValueError("The immutable native state changed during compression")
    native_receipt = source.with_suffix(".json")
    receipt = {
        "schema_version": 1, "codec": native.CODEC, "archive": data_identity(target), "native_payload": identity,
        "native_receipt": json.loads(native_receipt.read_text()) if native_receipt.exists() else None,
        "decoder_source": data_identity(Path(native.__file__)), "system_zstd_version": codec.version,
        "compression_seconds": time.perf_counter() - started, "block_bytes": native.BLOCK, "records": records,
        "maximum_native_payload_bytes": native.LIMIT, "complete_native_state_no_quantization": True,
        "encoder_source": data_identity(Path(__file__)),
        "ordered_pipeline": {"workers": 2, "capacity": 4, "peak_pending": peak_pending,
            "worker_count_observed": len(workers), "all_owned_workers_joined": True, "single_ordered_writer": True,
            "compression_intervals": intervals, "write_intervals": writes,
            "actual_compression_write_overlaps": sum(a["started"] < b["ended"] and b["started"] < a["ended"] for a in intervals for b in writes)},
    }
    atomic_json(target.with_suffix(target.suffix + ".json"), receipt)
    return receipt
