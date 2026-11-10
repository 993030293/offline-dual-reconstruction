"""Check record boundaries and rejection behavior using independent tiny fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from mapped_raw_reader import iter_mapped_messages
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payloads = [b"independent-payload-one", b"second-payload"]
    records = [struct.pack("<BQHIQQ", 5, 22+len(payload), 7, i+10, i+100, i+200)+payload
               for i, payload in enumerate(payloads)]
    original = b"".join(records)
    checks = []

    def fixture(name, data=original, *, compression="", declared_size=None, chunk_size=None):
        path = args.output / (name+".bin")
        path.write_bytes(data)
        return SimpleNamespace(path=path, size=len(data) if declared_size is None else declared_size,
            channel_id=lambda topic: 7, locations={7: [(0, 0), (0, len(records[0]))]},
            chunks=[SimpleNamespace(compression=compression, uncompressed_size=len(data) if chunk_size is None else chunk_size)],
            _chunk_records_start=lambda stream, chunk: 0)

    valid = fixture("valid")
    rows = list(iter_mapped_messages(valid, "camera", [1, 0, 1]))
    expected = [(i, {"sequence": i+10, "log_ns": i+100, "publish_ns": i+200,
                     "payload": payload, "body_length": 22+len(payload)}) for i, payload in enumerate(payloads)]
    if rows != expected:
        raise AssertionError("A valid record changed or order/deduplication differed")
    checks.append("independent_message_oracle_and_sorted_unique_ids")
    for length in [0, 3, 100]:
        values = list(iter_mapped_messages(valid, "camera", [0, 1], prefix_bytes=length))
        if any(row[1]["payload"] != payloads[i][:length] for i, row in enumerate(values)):
            raise AssertionError("Prefix semantics changed")
    checks.append("zero_short_and_oversize_prefix")
    if hashlib.sha256(valid.path.read_bytes()).digest() != hashlib.sha256(original).digest():
        raise AssertionError("Read-only source changed")
    checks.append("fixture_bytes_unchanged")

    def reject(name, index, ids=(0,), prefix=None):
        try:
            list(iter_mapped_messages(index, "camera", ids, prefix_bytes=prefix))
        except (ValueError, IndexError, NotImplementedError):
            checks.append(name)
        else:
            raise AssertionError("Malformed fixture was accepted: " + name)

    bad_opcode = bytes([0x06])+original[1:]
    bad_channel = original[:9]+struct.pack("<H", 8)+original[11:]
    small_body = original[:1]+struct.pack("<Q", 21)+original[9:]
    cross_boundary = original[:1]+struct.pack("<Q", len(original))+original[9:]
    reject("wrong_opcode", fixture("wrong_opcode", bad_opcode))
    reject("wrong_channel", fixture("wrong_channel", bad_channel))
    reject("short_message_header", fixture("short_header", small_body))
    reject("cross_chunk_record", fixture("cross_boundary", cross_boundary))
    reject("truncated_body", fixture("truncated", original[:25]))
    reject("compressed_chunk", fixture("compressed", compression="zstd"))
    reject("stale_file_size", fixture("stale", declared_size=len(original)+1))
    reject("negative_id", valid, ids=(-1,))
    reject("too_large_id", valid, ids=(2,))
    reject("negative_prefix", valid, prefix=-1)
    receipt = {"status": "synthetic_record_checks_passed_real_MCAP_pending", "checks": checks,
               "scope": "tiny_message_records_with_independent_index_fixture_not_full_MCAP_container",
               "performance_claim": False, "counted": False}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
