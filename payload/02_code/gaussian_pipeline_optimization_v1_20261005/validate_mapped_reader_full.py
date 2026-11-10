"""Verify all selected raw topics and prefixes against the original indexed reader."""

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys
import time

import psutil

from mapped_raw_reader import iter_mapped_messages
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-result", type=Path, required=True)
    args = parser.parse_args()
    small = json.loads(args.small_result.read_text())
    if small["status"] != "small_read_stage_passed_pending_integration":
        raise ValueError("The registered small speed prerequisite has not passed")
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    args.output.mkdir(parents=True, exist_ok=False)
    index = McapIndex(baseline.RAW)
    stat = baseline.RAW.stat()
    topics = [baseline.CAMERA_TOPIC, "/localization/pose/odom", "/sensing/camera/camera_info", "/tf_static"]
    atomic_json(args.output / "CONTRACT.json", {
        "source": data_identity(Path(__file__).with_name("mapped_raw_reader.py")),
        "small_result": data_identity(args.small_result), "raw_bytes": stat.st_size, "raw_mtime_ns": stat.st_mtime_ns,
        "maximum_seconds": 300, "rss_budget_bytes": 32*1024**3, "topics": topics,
        "timing_role": "equivalence_only_interleaved_read_not_performance_comparison"})
    started = time.perf_counter()
    results = []
    process = psutil.Process()
    for topic, prefix in [(topic, None) for topic in topics] + [(baseline.CAMERA_TOPIC, 256)]:
        count = index.message_count(topic)
        original = index.iter_selected_messages(topic, range(count), prefix_bytes=prefix)
        candidate = iter_mapped_messages(index, topic, range(count), prefix_bytes=prefix)
        digest, payload_bytes, compared = hashlib.sha256(), 0, 0
        for a, b in itertools.zip_longest(original, candidate):
            if a is None or b is None or a != b:
                raise AssertionError("Actual MCAP message/payload mismatch on topic " + topic)
            frame, row = a
            digest.update(str(frame).encode()+b"\0"+row["payload"])
            payload_bytes += len(row["payload"])
            compared += 1
            if time.perf_counter()-started > 300 or process.memory_info().rss > 32*1024**3:
                raise RuntimeError("Full equivalence exceeded its registered resource bound")
        if compared != count:
            raise AssertionError("Topic denominator changed")
        results.append({"topic": topic, "prefix_bytes": prefix, "count": count, "payload_bytes": payload_bytes,
                        "equal_payload_stream_sha256": digest.hexdigest()})
        atomic_json(args.output / "PROGRESS.json", {"completed": results})
    receipt = {"status": "passed_full_raw_messages_and_prefix_equivalence", "results": results,
               "elapsed_seconds": time.perf_counter()-started, "counted": False,
               "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", process.memory_info().rss),
               "actual_initial_export_pending": True}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
