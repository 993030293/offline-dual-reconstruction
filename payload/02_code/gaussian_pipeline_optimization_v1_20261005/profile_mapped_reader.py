"""Compare complete raw payload reads without changing downstream interpretation."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import psutil

from mapped_raw_reader import iter_mapped_messages
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    input_path = project / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p1_raw_combined_001/INITIAL_FRAME_MANIFEST.json"
    frames = json.loads(input_path.read_text())["selected_frames"]
    ids = [int(frames[i]["camera_frame"]) for i in np.linspace(0, len(frames)-1, 128, dtype=int)]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("mapped_raw_reader.py"), baseline.TOOLS / "mcap_audit.py"]
    source_dir = args.output / "source_snapshot"
    source_dir.mkdir()
    for source in sources:
        shutil.copyfile(source, source_dir / source.name)
    raw_stat = baseline.RAW.stat()
    atomic_json(args.output / "CONTRACT.json", {
        "frames": ids, "source_selection": data_identity(input_path), "sources": [data_identity(p) for p in sources],
        "raw": {"path": str(baseline.RAW), "bytes": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns},
        "trials": 3, "reverse_order_trial": 1, "maximum_trial_seconds": 300, "rss_budget_bytes": 32 * 1024**3,
        "speed_gate": "at_least10percent_median_complete_payload_read_and_hash_stage",
        "cache": "OS_cache_not_cleared_separate_first_unmeasured_read", "training_parallel": False})
    index = McapIndex(baseline.RAW)
    process = psutil.Process()

    def read(label):
        before = time.perf_counter()
        source = index.iter_selected_messages(baseline.CAMERA_TOPIC, ids) if label == "reference" else iter_mapped_messages(index, baseline.CAMERA_TOPIC, ids)
        rows, byte_count = [], 0
        for number, message in source:
            payload = message["payload"]
            byte_count += len(payload)
            rows.append({"number": number, "sha256": hashlib.sha256(payload).hexdigest(),
                         **{key: message[key] for key in ["sequence", "log_ns", "publish_ns", "body_length"]}})
            if time.perf_counter() - before > 300 or process.memory_info().rss > 32 * 1024**3:
                raise RuntimeError("Reader diagnostic exceeded its registered resource bounds")
        return {"elapsed_seconds": time.perf_counter()-before, "bytes": byte_count, "rows": rows,
                "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", process.memory_info().rss)}

    first = read("reference")
    atomic_json(args.output / "FIRST_ACCESS.json", first)
    results = []
    for trial in range(3):
        for label in (["mapped", "reference"] if trial == 1 else ["reference", "mapped"]):
            result = read(label)
            if result["rows"] != first["rows"] or result["bytes"] != first["bytes"]:
                raise ValueError("Raw payload or message identity changed")
            result.pop("rows")
            results.append(dict(result, trial=trial, label=label))
            atomic_json(args.output / "PROGRESS.json", {"results": results})
    medians = {label: statistics.median(row["elapsed_seconds"] for row in results if row["label"] == label)
               for label in ["reference", "mapped"]}
    reduction = 1 - medians["mapped"] / medians["reference"]
    receipt = {"status": "small_read_stage_passed_pending_integration" if reduction >= .1 else "small_read_speed_guard_failed",
               "results": results, "medians": medians, "reduction": reduction,
               "all_payloads_and_metadata_identical": True, "counted": False,
               "full7338_and_actual_pipeline_not_run": True}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
