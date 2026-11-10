"""Exercise immutable numeric scores and a read-only sensor map through the raw entry."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from quality_engine import FIELDS
import run_initial_pipeline
from columnar_quality_scores import columnar_score_cache
from encoded_jpeg_cache import cached_jpeg_encoding
from windowed_raw_reader import windowed_sensor_reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "quality-cache"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--score-table", type=Path)
    parser.add_argument("--reader-mode", choices=["indexed", "mapped"], default="indexed")
    parser.add_argument("--jpeg-cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("columnar_quality_scores.py"),
        Path(__file__).with_name("encoded_jpeg_cache.py"), V1/"run_initial_pipeline.py",
        V1/"quality_engine.py", V1/"mapped_raw_reader.py", Path(__file__).with_name("windowed_raw_reader.py")]
    identities = [data_identity(p) for p in paths]
    atomic_json(args.output/"RUNTIME_CONTRACT.json", {"sources": identities, "reader": args.reader_mode,
        "score_table": str(args.score_table) if args.score_table else None, "JPEG_cache": str(args.jpeg_cache) if args.jpeg_cache else None,
        "maximum_seconds": 180, "maximum_RSS_GiB": 8, "all7338_decisions1002_images11583_pairs_and_raw_odometry_exact": True,
        "initial_selection_uses_only_raw_coarse_pose_and_image_quality": True, "counted": False})
    original_argv = sys.argv
    try:
        with (columnar_score_cache(run_initial_pipeline, args.score_table, args.output/"score_misses", FIELDS)
                if args.score_table else nullcontext(None)) as scores, (
                cached_jpeg_encoding(args.jpeg_cache) if args.jpeg_cache else nullcontext(None)) as jpegs, windowed_sensor_reader(args.reader_mode == "mapped"):
            sys.argv = [str(V1/"run_initial_pipeline.py"), "--output", str(args.output/"initial"),
                "--cache", str(args.quality_cache), "--workers", "4", "--scorer", "compact",
                "--selector", "batched", "--pose-mode", "batched", "--reader", args.reader_mode]
            run_initial_pipeline.main()
        if identities != [data_identity(p) for p in paths] or (scores is not None and len(scores) != 1):
            raise ValueError("An immutable runtime choice changed during the complete raw entry")
        atomic_json(args.output/"RUNTIME_TRACE.json", {"status": "actual_full_raw_entry_completed", "sources": identities,
            "numeric_score_cache": scores[0].receipt() if scores is not None else None,
            "JPEG_cache": jpegs.receipt() if jpegs else None, "reader": args.reader_mode, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
