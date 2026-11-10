"""Run the complete raw-only initial entry with an ordered CPU export queue."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_initial_pipeline
import run_initial_with_jpeg_cache
from parallel_raw_export import parallel_export_preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-mode", choices=["baseline", "queue"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("parallel_raw_export.py"),
        Path(run_initial_pipeline.__file__), Path(run_initial_with_jpeg_cache.__file__)]
    sources = [data_identity(p) for p in paths]
    cached = "--jpeg-cache" in remaining
    atomic_json(args.output/"EXPORT_QUEUE_CONTRACT.json", {"mode": args.export_mode, "sources": sources,
        "workers": 4, "capacity": 8, "OpenCV_threads_each": 1, "JPEG_cache_main_thread_only": cached,
        "minimum_complete_export_stage_reduction": .1, "all7338_decisions1002_image_bytes11583_pairs_exact": True,
        "maximum_seconds": 180, "maximum_RSS_GiB": 8, "counted": False})
    original_argv = sys.argv
    try:
        with parallel_export_preparation(run_initial_pipeline, args.export_mode == "queue", not cached) as records:
            sys.argv = [str(Path(run_initial_with_jpeg_cache.__file__)), "--output", str(args.output/"raw_entry"), *remaining]
            run_initial_with_jpeg_cache.main()
        if sources != [data_identity(p) for p in paths] or len(records) != 1:
            raise ValueError("The actual queued raw export changed its single immutable implementation")
        if not records[0]["closed"] or (records[0]["submitted"], records[0]["consumed"]) != (1002, 1002):
            raise ValueError("The bounded export queue did not preserve all selected frames")
        atomic_json(args.output/"EXPORT_QUEUE_TRACE.json", {"status": "actual_full_raw_entry_completed", "sources": sources,
            "records": records, "counted": False, "ordered_main_thread_file_writes": True})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
