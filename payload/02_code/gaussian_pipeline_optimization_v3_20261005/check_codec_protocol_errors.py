"""Check real damaged streams and complete requests outside codec size bounds."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/R170/g2"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from bounded_zstd_payload import ZstdPayloadCodec
from codec_sift_matching import CodecObservedWorker, PipeWriteMeter, read_server_trace
from integer_sift_matching import HEADER, read_exact
from check_codec_sift_protocol import vector_identity, ended


def broken_stream(label, enabled, metadata, payload=b"", metadata_truncated=False):
    folder = OUT / label
    folder.mkdir()
    log = folder / "worker.log"
    worker = CodecObservedWorker(log, enabled)
    body = json.dumps(metadata, separators=(",", ":")).encode()
    blocks = [HEADER.pack(len(body)), body[:-1] if metadata_truncated else body, payload]
    pipe = worker.process.stdin
    meter = PipeWriteMeter(pipe)
    worker.process.stdin = meter
    try:
        for block in blocks:
            view = memoryview(block)
            while view:
                count = meter.write(view)
                if not count:
                    raise EOFError("The actual diagnostic request stopped accepting bytes")
                view = view[count:]
        meter.flush()
        meter.close()
        header = worker.process.stdout.read(HEADER.size)
        if header:
            if len(header) != HEADER.size:
                raise AssertionError("A real server failure returned a truncated response header")
            reply = json.loads(read_exact(worker.process.stdout, HEADER.unpack(header)[0]))
            if reply["status"] != "failed" or reply["payload_bytes"] != 0:
                raise AssertionError("A malformed actual input was accepted")
        else:
            reply = {"status": "closed_before_response"}
    finally:
        worker.process.stdin = pipe
        resource = ended(worker, failed=True)
    server = read_server_trace(log) if enabled else None
    row = {"name": label, "enabled": enabled, "reply": reply,
           "actual_written_input_bytes": worker.receipt["actual_written_input_bytes"] + meter.written_bytes,
           "normal_info_receipt": worker.receipt, "damaged_write_bytes": meter.written_bytes,
           "returncode": worker.process.returncode, "resource": resource, "server": server,
           "no_normal_stop_or_resend": "shutdown" not in worker.receipt,
           "stderr": log.read_text(encoding="utf-8")}
    atomic_json(folder / "RESULT.json", row)
    return row


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[CODE / name for name in (
        "check_codec_sift_protocol.py", "codec_sift_matching.py", "sift_payload_input_worker.py",
        "bounded_zstd_payload.py", "integer_sift_worker.py", "integer_sift_matching.py",
        "metric_branch_localization_003.py", "codec_owned_matching_worker.py",
        "codec_localization_node.py", "sift_process_node_003.py", "fused_owned_matching_worker.py")],
        ROOT / "05_envs/colmap-4.1.1-cuda/bin/zstd.dll"]
    source_ids = [data_identity(path) for path in sources]
    archive = OUT / "source_snapshot"
    archive.mkdir()
    for number, path in enumerate(sources):
        shutil.copyfile(path, archive / (f"{number:03d}_" + path.name))
    normal = []
    for enabled in (False, True):
        folder = OUT / ("codec" if enabled else "raw")
        folder.mkdir()
        log = folder / "worker.log"
        worker = CodecObservedWorker(log, enabled)
        try:
            query = np.zeros((1, 128), np.float32)
            trains = [np.zeros((5462, 128), np.float32) for _ in range(12)]
            before = [vector_identity(value) for value in [query, *trains]]
            result = worker.batch(query, trains)
            values = [[vector_identity(part) for part in value] for value in result]
            unchanged = before == [vector_identity(value) for value in [query, *trains]]
        finally:
            resource = ended(worker)
        normal.append({"enabled": enabled, "values": values,
            "inputs_unchanged": unchanged, "receipt": worker.receipt, "resource": resource,
            "server": read_server_trace(log) if enabled else None})
    codec = ZstdPayloadCodec()
    raw = bytes(384)
    encoded = codec.encode(raw)
    native = {"command": "batch", "query_rows": 1, "train_counts": [2]}
    encoded_meta = dict(native, wire_codec="zstd1", wire_payload_bytes=len(encoded))
    errors = []
    for enabled in (False, True):
        errors.append(broken_stream("raw_truncated_" + str(int(enabled)), enabled, native, raw[:-1]))
        errors.append(broken_stream("metadata_truncated_" + str(int(enabled)), enabled, native,
                                    metadata_truncated=True))
    for label, record, payload in (
        ("encoded_truncated", encoded_meta, encoded[:-1]),
        ("invalid_frame", dict(encoded_meta, wire_payload_bytes=9), b"notframe!"),
        ("frame_trailing", dict(encoded_meta, wire_payload_bytes=len(encoded)+1), encoded+b"x"),
        ("unknown_codec", dict(encoded_meta, wire_codec="unsupported"), encoded),
        ("wrong_decoded_length", dict(encoded_meta, wire_payload_bytes=len(codec.encode(raw[:-1]))), codec.encode(raw[:-1])),
    ):
        errors.append(broken_stream(label, True, record, payload))
    boolean_errors = [broken_stream("boolean_query_" + str(int(enabled)), enabled,
        {"command": "batch", "query_rows": True, "train_counts": [2]}, raw) for enabled in (False, True)]
    a, b = normal
    large = [row for row in b["receipt"]["wire"] if row["raw_payload_bytes"] > 8*1024**2]
    checks = {
        "original_boolean_query_native_error_preserved": boolean_errors[0]["reply"] == boolean_errors[1]["reply"]
            and boolean_errors[0]["reply"]["status"] == "failed",
        "full_large_original_request_all_twelve_vectors_bits_preserved": a["values"] == b["values"]
            and len(a["values"]) == 12 and all(row["inputs_unchanged"] for row in normal),
        "large_request_raw_before_execution_no_feature_crop": len(large) == 1
            and large[0]["codec"] == "raw" and large[0]["raw_payload_bytes"] == (1+12*5462)*128
            and large[0]["transmitted_payload_bytes"] == large[0]["raw_payload_bytes"],
        "raw_truncation_original_error_preserved": errors[0]["reply"] == errors[2]["reply"],
        "metadata_truncation_original_EOF_preserved": errors[1]["reply"] == errors[3]["reply"]
            and errors[1]["reply"]["status"] == "closed_before_response",
        "all_nine_actual_damaged_native_processes_failed_and_closed": len(errors) == 9
            and all(row["returncode"] != 0 and row["resource"]["closed"] and row["no_normal_stop_or_resend"] for row in errors),
        "all_five_encoded_stream_errors_explicitly_rejected": all(row["reply"]["status"] == "failed" for row in errors[4:]),
        "every_decoder_alias_restored_on_success_and_failure": b["server"]["stdin_alias_restored"]
            and all(row["server"]["stdin_alias_restored"] for row in errors + boolean_errors if row["enabled"]),
        "both_boolean_native_failed_processes_closed": all(row["returncode"] != 0
            and row["resource"]["closed"] and row["no_normal_stop_or_resend"] for row in boolean_errors),
        "all_normal_native_stop_and_resource_guards": all(row["receipt"]["returncode"] == 0
            and row["resource"]["closed"] and row["resource"]["peak_RSS_bytes"] <= 16*1024**3 for row in normal),
        "all_real_decoded_complete_normal_payload_hashes_exact": len(b["server"]["rows"]) == 1
            and all(server["raw_sha256"] == client["raw_sha256"] for server, client in zip(
                b["server"]["rows"], [row for row in b["receipt"]["wire"] if row["command"] == "batch"])),
        "all_current_sources_library_unchanged": source_ids == [data_identity(path) for path in sources],
    }
    result = {"status": "native_damaged_and_large_protocol_passed_pending_full_localization0"
                if all(checks.values()) else "failed_verification", "checks": checks,
              "normal": normal, "errors": errors, "boolean_native_errors": boolean_errors, "sources": source_ids,
              "extra_actual_written_input_bytes": sum(row["receipt"]["actual_written_input_bytes"] for row in normal)
                  + sum(row["actual_written_input_bytes"] for row in errors + boolean_errors),
              "complete_component_seconds": time.perf_counter()-started, "new_optimization_count": 0}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "extra_actual_written_input_bytes", "complete_component_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An actual damaged or large native protocol check failed")


if __name__ == "__main__":
    main()
