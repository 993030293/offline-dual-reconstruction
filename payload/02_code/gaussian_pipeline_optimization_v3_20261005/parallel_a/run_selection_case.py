"""Run the whole current raw entry with an immutable warm cache and one selector."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT/"02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V3)]
from runtime import atomic_json, data_identity
import quality_engine
import run_raw_calibration
import run_initial_pipeline
from spatial_novelty import spatial_novelty_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "hash"), required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.quality_cache.is_dir():
        raise ValueError("A new output and existing read-only warm quality cache are required")
    started = time.perf_counter()
    paths = [Path(__file__), Path(__file__).with_name("spatial_novelty.py"), V1/"selection_engine.py"]
    sources = [data_identity(path) for path in paths]
    original_write = quality_engine.ScoreCache.write
    original_open = Path.open
    original_atomic = run_initial_pipeline.atomic_json
    tail = {"started": None, "seconds": None, "starts": 0, "ends": 0}
    def timed_open(path, *positional, **keywords):
        if path == args.output/"initial/frame_decisions.jsonl":
            if tail["started"] is not None:
                raise ValueError("The complete original frame decision output opened more than once")
            tail.update(started=time.perf_counter(), starts=tail["starts"]+1)
        return original_open(path, *positional, **keywords)
    def timed_atomic(path, payload):
        result = original_atomic(path, payload)
        if Path(path) == args.output/"initial/STATE.json" and payload.get("status") == "completed_initial_selection":
            if tail["started"] is None:
                raise ValueError("The complete original selector output tail never started")
            tail.update(seconds=time.perf_counter()-tail["started"], ends=tail["ends"]+1)
        return result
    def reject_old_cache_write(store, key, value):
        raise ValueError("A frozen quality cache miss cannot write to the original cache")
    quality_engine.ScoreCache.write = reject_old_cache_write
    Path.open = timed_open
    run_initial_pipeline.atomic_json = timed_atomic
    preparation = time.perf_counter()-started
    previous = sys.argv
    try:
        with spatial_novelty_execution(args.mode == "hash") as trace:
            sys.argv = [str(Path(run_raw_calibration.__file__)), "--calibration-mode", "memo",
                "--output", str(args.output), "--quality-cache", str(args.quality_cache)]
            run_raw_calibration.main()
        before = time.perf_counter()
        quality_engine.ScoreCache.write = original_write
        Path.open = original_open
        run_initial_pipeline.atomic_json = original_atomic
        if sources != [data_identity(path) for path in paths] or not trace["aliases_restored"]:
            raise ValueError("A selector source or original function was not restored")
        if tail["starts"] != 1 or tail["ends"] != 1 or tail["seconds"] is None:
            raise ValueError("The actual frame decision and manifest output lifecycle was not completely timed")
        closing = time.perf_counter()-before
        output_started = time.perf_counter()
        atomic_json(args.output/"SELECTOR.json", {"mode": args.mode, "sources": sources, "trace": trace,
            "additional_complete_selector_lifecycle_seconds": preparation+trace["setup_seconds"]+trace["restore_seconds"]+closing,
            "complete_original_frame_decisions_manifests_and_final_state_output_seconds": tail["seconds"],
            "timing_bindings_restored": Path.open is original_open and run_initial_pipeline.atomic_json is original_atomic,
            "old_quality_cache_read_only_miss_rejected": True, "no_policy_or_matching_pair_change": True,
            "same_Q78_Q50_Q53_P18_R99_JPEG_cache_off": True, "counted": False})
        atomic_json(args.output/"COLLECTION_COST.json", {"selector_trace_output_seconds": time.perf_counter()-output_started,
            "measurement_receipt_write_is_separate_control_plane": True})
    finally:
        sys.argv = previous
        quality_engine.ScoreCache.write = original_write
        Path.open = original_open
        run_initial_pipeline.atomic_json = original_atomic


if __name__ == "__main__":
    main()
