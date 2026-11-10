"""Measure complete captured payloads without starting the native GPU worker."""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from bounded_zlib_payload import encode_payload, decode_payload, MAXIMUM_RAW_BYTES

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/a0"
REQUESTS = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rc001_requests"


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    manifest_file = REQUESTS / "REQUESTS.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["status"] != "passed" or not all(manifest["checks"].values()) or len(manifest["records"]) != 48:
        raise ValueError("The original complete request capture contract changed")
    paths = [manifest_file, *[REQUESTS / value["file"] for value in manifest["blobs"].values()]]
    identities = [data_identity(path) for path in paths]
    sources = [Path(__file__), CODE / "bounded_zlib_payload.py", CODE / "integer_sift_worker.py", CODE / "integer_sift_matching.py"]
    source_ids = [data_identity(path) for path in sources]
    archive = OUT / "source_snapshot"
    archive.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    blobs = {}
    for key, value in manifest["blobs"].items():
        path = (REQUESTS / value["file"]).resolve()
        if path.parent != REQUESTS.resolve() or path.is_symlink():
            raise ValueError("A frozen descriptor payload escaped its original folder")
        raw = path.read_bytes()
        if len(raw) != value["rows"] * 128 or hashlib.sha256(raw).hexdigest() != key:
            raise ValueError("An original captured byte payload identity changed")
        bytes_array = np.frombuffer(raw, np.uint8)
        if not np.array_equal((bytes_array.astype(np.float32) / 512 * 512).astype(np.uint8), bytes_array):
            raise ValueError("The original byte to float32 representation no longer recovers exactly")
        blobs[key] = raw
    payloads = []
    for index, record in enumerate(manifest["records"]):
        if record["ordinal"] != index:
            raise ValueError("The complete original request order changed")
        payloads.append(blobs[record["query"]] + b"".join(blobs[key] for key in record["trains"]))
    rows = []
    for level in (1, 3):
        for repetition in range(3):
            encode_seconds = decode_seconds = 0.
            encoded_bytes = 0
            for raw in payloads:
                mark = time.perf_counter()
                encoded = encode_payload(raw, level)
                encode_seconds += time.perf_counter() - mark
                mark = time.perf_counter()
                recovered = decode_payload(encoded, len(raw))
                decode_seconds += time.perf_counter() - mark
                if recovered != raw:
                    raise AssertionError("A complete native byte payload did not recover exactly")
                encoded_bytes += len(encoded)
            rows.append({"level": level, "repetition": repetition, "complete_request_count": 48,
                         "raw_bytes": sum(map(len, payloads)), "encoded_bytes": encoded_bytes,
                         "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                         "all_complete_payload_bytes_exact": True})
    sample = bytes(range(256)) * 16
    encoded = encode_payload(sample, 1)
    negatives = [("truncated", encoded[:-1], len(sample)), ("trailing", encoded + b"x", len(sample)),
                 ("concatenated", encoded + encoded, len(sample)), ("wrong_short", encoded, len(sample) - 1),
                 ("wrong_long", encoded, len(sample) + 1), ("bad_stream", b"not-zlib", len(sample)),
                 ("zero_bound", encoded, 0), ("above_bound", encoded, MAXIMUM_RAW_BYTES + 1)]
    errors = []
    for label, value, expected in negatives:
        try:
            decode_payload(value, expected)
        except (ValueError, zlib.error) as error:
            errors.append({"name": label, "type": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("A malformed bounded request was accepted: " + label)
    checks = {"complete48_original_requests_all_six_passes_exact": len(rows) == 6 and all(row["all_complete_payload_bytes_exact"] and row["complete_request_count"] == 48 for row in rows),
              "all_eight_negative_stream_bounds_rejected": len(errors) == 8,
              "all_original_input_bytes_unchanged": identities == [data_identity(path) for path in paths],
              "all_source_identities_unchanged": source_ids == [data_identity(path) for path in sources],
              "no_Torch_CUDA_loaded": "torch" not in sys.modules,
              "disk_budget": sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 2 * 1024**3
                and shutil.disk_usage(OUT).free >= 20 * 1024**3}
    summaries = []
    for level in (1, 3):
        group = [row for row in rows if row["level"] == level]
        summaries.append({"level": level, "raw_bytes": group[0]["raw_bytes"], "encoded_bytes": group[0]["encoded_bytes"],
                          "reduction": 1 - group[0]["encoded_bytes"] / group[0]["raw_bytes"],
                          "complete_encode_decode_median_seconds": statistics.median(row["encode_seconds"] + row["decode_seconds"] for row in group)})
    result = {"status": "CPU_codec_diagnostic_passed0" if all(checks.values()) else "failed",
              "checks": checks, "sources": source_ids, "inputs": identities, "rows": rows,
              "summaries": summaries, "negative_streams": errors, "component_seconds": time.perf_counter() - started,
              "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION, "new_optimization_count": 0,
              "not_actual_pipe_transfer_whole_speed_worker_or_full_localization": True}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "summaries", "component_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A complete CPU codec diagnostic check failed")


if __name__ == "__main__":
    main()
