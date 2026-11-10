"""Compare real raw JPEG bytes and ordered-queue lifecycle without changing selection."""

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import initial_selection
sys.path.insert(0, str(initial_selection.TOOLS))
from mcap_audit import McapIndex
from camera_motion import decode_bayer
from color_contract import demosaic_rggb8_to_bgr
import run_initial_pipeline
from parallel_raw_export import prepared_exports, parallel_export_preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["selected_frames"][:16]
    index = McapIndex(initial_selection.RAW)
    messages = list(index.iter_selected_messages(initial_selection.CAMERA_TOPIC, [r["camera_frame"] for r in rows]))
    cv2.setNumThreads(1)
    traces, values = {}, {}
    for name, workers, encoded in (("serial", 1, True), ("queue", 4, True), ("main_encoder", 4, False)):
        trace = {}
        output = list(prepared_exports(iter(messages), decode_bayer, demosaic_rggb8_to_bgr,
            manifest["policy"]["jpeg_quality"], workers, 8, encoded, trace))
        values[name] = [(frame, stamp, bool(ok), data.tobytes()) for frame, stamp, ok, data in output]
        traces[name] = trace
    checks = {"actual16_raw_source_outputs_exact": len(messages) == 16 and values["serial"] == values["queue"] == values["main_encoder"],
        "raw_frame_and_timestamp_order_matches_manifest": [(r[0], r[1]) for r in values["queue"]] ==
            [(r["camera_frame"], r["camera_header_ns"]) for r in rows],
        "all_actual_queues_consumed_and_closed": all(r["submitted"] == r["consumed"] == 16 and r["closed"] for r in traces.values()),
        "all_actual_pending_queues_bounded": all(r["maximum_pending"] <= 8 for r in traces.values())}
    original = run_initial_pipeline.main
    try:
        with parallel_export_preparation(run_initial_pipeline):
            raise RuntimeError("Intentional export adapter restoration check")
    except RuntimeError:
        pass
    checks["entry_restored_after_error"] = run_initial_pipeline.main is original
    trace = {}
    iterator = prepared_exports(iter(messages), decode_bayer, demosaic_rggb8_to_bgr,
        manifest["policy"]["jpeg_quality"], 4, 8, True, trace)
    next(iterator)
    iterator.close()
    checks["early_consumer_close_releases_queue"] = trace["closed"] and trace["consumed"] == 1 and trace["submitted"] <= 8
    trace = {}
    def failing_decoder(payload):
        raise ValueError("Intentional raw decoding error")
    try:
        list(prepared_exports(iter(messages), failing_decoder, demosaic_rggb8_to_bgr,
            manifest["policy"]["jpeg_quality"], 4, 8, True, trace))
        checks["worker_error_propagates_and_closes_queue"] = False
    except ValueError as error:
        checks["worker_error_propagates_and_closes_queue"] = str(error) == "Intentional raw decoding error" and trace["closed"]
    trace = {}
    empty = list(prepared_exports(iter(()), decode_bayer, demosaic_rggb8_to_bgr, 97, 4, 8, True, trace))
    checks["empty_reader_closes_without_fake_frames"] = not empty and trace["closed"] and trace["submitted"] == 0
    paths = [Path(__file__), Path(__file__).with_name("parallel_raw_export.py"),
        initial_selection.TOOLS/"camera_motion.py", initial_selection.TOOLS/"color_contract.py",
        initial_selection.TOOLS/"mcap_audit.py", Path(run_initial_pipeline.__file__)]
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "traces": traces, "sources": [data_identity(p) for p in paths],
        "raw_source": str(initial_selection.RAW), "source_manifest": data_identity(args.manifest),
        "counted": False, "three_full_raw_entries_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
    if not all(checks.values()):
        raise ValueError("The actual ordered raw export lifecycle did not pass")


if __name__ == "__main__":
    main()
