"""Capture every real original matcher request without changing its result."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare
import integer_sift_matching as native
from integer_sift_matching import bytes_exact
import localize_resident_descriptors


def entry(folder):
    found = list(folder.rglob("H_LOCALIZATION.json"))
    if len(found) != 1 or found[0].parent.name != "localization":
        raise ValueError("The actual original full pose output is ambiguous")
    return found[0].parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "requests", "original-control"):
        parser.add_argument("--"+name, type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.output.exists():
        raise FileExistsError("The original request capture pose output must be new")
    args.requests.mkdir(parents=True, exist_ok=False)
    records, blobs = [], {}
    paths = [Path(__file__), Path(native.__file__), Path(localize_resident_descriptors.__file__)]
    sources = [data_identity(path) for path in paths]
    original, old_argv = native.DistanceWorker, sys.argv

    def save(array):
        raw = bytes_exact(array).tobytes()
        key = hashlib.sha256(raw).hexdigest()
        path = args.requests/(key+".bin")
        if key not in blobs:
            with path.open("xb") as stream:
                stream.write(raw)
            blobs[key] = {"rows": len(array), "file": path.name, "identity": data_identity(path)}
        elif blobs[key]["rows"] != len(array) or path.read_bytes() != raw:
            raise ValueError("A complete captured descriptor content identity changed")
        recovered = np.frombuffer(raw, np.uint8).reshape(len(array), 128).astype(np.float32)/512
        if not np.array_equal(recovered, array):
            raise ValueError("The captured original float32 descriptor did not recover exactly")
        return key

    class CaptureWorker(original):
        def batch(self, query, trains):
            record = {"ordinal": len(records), "query": save(query), "trains": [save(value) for value in trains]}
            result = super().batch(query, trains)
            records.append(record)
            return result

    try:
        native.DistanceWorker = CaptureWorker
        sys.argv = [str(Path(localize_resident_descriptors.__file__)), "--transport-mode", "incumbent",
            "--output", str(args.output), *remaining]
        localize_resident_descriptors.main()
    finally:
        native.DistanceWorker, sys.argv = original, old_argv
    left, right = entry(args.original_control), entry(args.output)
    protections = compare(left, right, left/"NEIGHBOR_TRACE.json", right/"NEIGHBOR_TRACE.json")
    atomic_json(args.requests/"INVARIANTS.json", protections)
    checks = {"all_original24_48_real_pose_entry_outputs_protected": protections["all_protected"],
        "all48_original_requests_captured_in_original_order": len(records) == 48 and [row["ordinal"] for row in records] == list(range(48)),
        "all_original_float32_inputs_exactly_recovered": all(blobs[key]["identity"] == data_identity(args.requests/blobs[key]["file"]) for key in blobs),
        "original_worker_factory_restored": native.DistanceWorker is original,
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "records": records, "blobs": blobs, "sources": sources,
        "capture_has_extra_IO_and_conversion_and_is_not_a_speed_result": True,
        "current_original_queries_never_used_for_initial_selection": True, "counted": False}
    atomic_json(args.requests/"REQUESTS.json", result)
    print({key: result[key] for key in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original complete descriptor capture output guard failed")


if __name__ == "__main__":
    main()
