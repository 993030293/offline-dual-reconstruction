"""Compare complete actual MCAP payloads and exercise read-only map boundaries."""

import argparse
import copy
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
sys.path.insert(0, str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002"))
from runtime import atomic_json, data_identity
from windowed_raw_reader import iter_windowed_messages as iter_mapped_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    index = McapIndex(baseline.RAW)
    count = index.message_count(baseline.CAMERA_TOPIC)
    if count != 7338:
        raise ValueError("The complete original camera message denominator changed")
    original_stat = (index.path.stat().st_size, index.path.stat().st_mtime_ns)

    def identities(iterator):
        rows = []
        for number, value in iterator:
            rows.append((number, value["sequence"], value["log_ns"], value["publish_ns"],
                value["body_length"], len(value["payload"]), hashlib.sha256(value["payload"]).hexdigest()))
        return rows

    original = identities(index.iter_selected_messages(baseline.CAMERA_TOPIC, range(count)))
    mapped = identities(iter_mapped_messages(index, baseline.CAMERA_TOPIC, range(count)))
    checks = {"all7338_raw_payloads_and_metadata_exact": original == mapped and len(mapped) == count}
    iterator = iter_mapped_messages(index, baseline.CAMERA_TOPIC, [0, 1])
    _, first = next(iterator)
    retained = first["payload"]
    iterator.close()
    checks["early_close_keeps_independent_immutable_bytes"] = type(retained) is bytes and hashlib.sha256(retained).hexdigest() == original[0][-1]
    checks["sorted_duplicate_ids_match_original_contract"] = [row[0] for row in iter_mapped_messages(index, baseline.CAMERA_TOPIC, [1, 0, 1])] == [0, 1]
    checks["empty_selection_closes_normally"] = list(iter_mapped_messages(index, baseline.CAMERA_TOPIC, [])) == []

    def reject(label, selected, changed=None, expected=Exception, **kwargs):
        try:
            list(iter_mapped_messages(changed or index, baseline.CAMERA_TOPIC, selected, **kwargs))
            checks[label] = False
        except expected:
            checks[label] = True

    reject("negative_message_rejected", [-1], expected=IndexError)
    reject("out_of_range_rejected", [count], expected=IndexError)
    reject("noninteger_message_rejected", [.5], expected=IndexError)
    reject("negative_prefix_rejected", [0], expected=ValueError, prefix_bytes=-1)
    changed = copy.copy(index)
    changed.size -= 1
    reject("changed_source_size_rejected", [0], changed, ValueError)
    channel = index.channel_id(baseline.CAMERA_TOPIC)
    changed = copy.copy(index)
    changed.locations = dict(index.locations)
    chunk_number, _ = index.locations[channel][0]
    changed.locations[channel] = [(chunk_number, -1)]
    reject("negative_relative_offset_rejected", [0], changed, ValueError)
    changed = copy.copy(index)
    changed.chunks = list(index.chunks)
    old = index.chunks[chunk_number]
    changed.chunks[chunk_number] = SimpleNamespace(compression="unsupported-fixture", uncompressed_size=old.uncompressed_size)
    reject("compressed_chunk_rejected_without_silent_fallback", [0], changed, NotImplementedError)
    checks["source_stat_unchanged"] = original_stat == (index.path.stat().st_size, index.path.stat().st_mtime_ns)
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_lifecycle", "checks": checks,
        "actual_raw_source": str(index.path), "source_bytes": index.size, "all_raw_messages": count,
        "raw_source_modified": False, "all_payload_hashes": original,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("windowed_raw_reader.py"))],
        "window_allocation_max_bytes": 64*1024**2+65536, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The actual mapped sensor reader failed a raw input boundary")


if __name__ == "__main__":
    main()
