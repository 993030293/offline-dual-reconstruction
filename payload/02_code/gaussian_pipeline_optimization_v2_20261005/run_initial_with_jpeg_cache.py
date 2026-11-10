"""Invoke the accepted raw initial-selection entry with optional exact JPEG reuse."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from encoded_jpeg_cache import cached_jpeg_encoding
import run_initial_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpeg-cache", type=Path)
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("encoded_jpeg_cache.py"),
        V1/"run_initial_pipeline.py", V1/"runtime.py", V1/"quality_engine.py",
        V1/"selection_engine.py", V1/"pose_interpolation.py"]
    identities = [data_identity(path) for path in sources]
    atomic_json(args.output/"ENTRY_CONTRACT.json", {"mechanism": "Q33_EXACT_JPEG_ARTIFACT_REUSE",
        "sources": identities, "raw_entry": str(V1/"run_initial_pipeline.py"),
        "mode": "cache" if args.jpeg_cache else "reference", "quality_cache": str(args.quality_cache),
        "JPEG_cache": str(args.jpeg_cache) if args.jpeg_cache else None,
        "maximum_cache_bytes": 2*1024**3, "source_maps_and_decisions_reused": False,
        "selection_exclusions_unchanged": True, "counted": False})
    old_argv = sys.argv
    cache = None
    try:
        sys.argv = [str(V1/"run_initial_pipeline.py"), "--output", str(args.output/"initial"),
            "--cache", str(args.quality_cache), "--workers", "4", "--scorer", "compact",
            "--selector", "batched", "--pose-mode", "batched"]
        context = cached_jpeg_encoding(args.jpeg_cache) if args.jpeg_cache else nullcontext(None)
        with context as cache:
            run_initial_pipeline.main()
        if identities != [data_identity(path) for path in sources]:
            raise ValueError("The registered export sources changed during execution")
        atomic_json(args.output/"JPEG_CACHE_RESULT.json", {"status": "actual_raw_entry_completed",
            "sources_unchanged": True, "cache": cache.receipt() if cache else None,
            "actual_decode_and_export_remain_in_original_entry": True})
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
