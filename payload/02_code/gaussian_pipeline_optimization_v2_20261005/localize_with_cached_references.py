"""Exercise calibrated reference caching in the existing full24 localization route."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from calibrated_reference_cache import calibrated_reference_cache
import localize_with_cached_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mode", choices=["baseline", "cache"], required=True)
    parser.add_argument("--reference-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("calibrated_reference_cache.py"),
               Path(localize_with_cached_queries.__file__)]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"REFERENCE_CACHE_CONTRACT.json", {"sources": identities, "mode": args.reference_mode,
        "raw_pixels_camera_and_options_bound": True, "maximum_cache_bytes": 512*1024**2,
        "cold_and_repeat_costs_separate": True, "same24_development_denominator": True, "counted": False})
    original = sys.argv
    try:
        with calibrated_reference_cache(args.reference_cache_root, args.reference_mode == "cache") as receipt:
            sys.argv = [str(Path(localize_with_cached_queries.__file__)), "--output", str(args.output/"query_entry"), *remaining]
            localize_with_cached_queries.main()
            cache = receipt["cache"].receipt() if receipt["cache"] is not None else None
        if identities != [data_identity(p) for p in sources]:
            raise ValueError("Calibrated reference implementation changed during localization")
        atomic_json(args.output/"REFERENCE_CACHE_TRACE.json", {"status": "actual24_references_completed",
            "mode": args.reference_mode, "operations": receipt["operations"], "cache": cache,
            "reference_stage_seconds": sum(r["seconds"] for r in receipt["operations"]),
            "sources_unchanged": True, "counted": False})
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
