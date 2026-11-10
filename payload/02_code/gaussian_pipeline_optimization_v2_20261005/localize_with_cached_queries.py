"""Exercise map-independent query caching in the complete localization entry."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
sys.path.insert(0, str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"))
from runtime import atomic_json, data_identity
import feature_localization
import localize_with_neighbors
from query_feature_cache import QueryFeatureCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-mode", choices=["baseline", "cache"], required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = feature_localization.FeatureBank.extra
    original_argv = sys.argv
    sources = [Path(__file__), Path(__file__).with_name("query_feature_cache.py"),
               Path(localize_with_neighbors.__file__), Path(feature_localization.__file__)]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"FEATURE_CACHE_CONTRACT.json", {"mode": args.feature_mode,
        "sources": identities, "maximum_seconds": 900, "maximum_RSS_GiB": 8,
        "maximum_cache_bytes": 512*1024**2, "raw_image_and_extractor_key": True,
        "map_and_ROI_independent": True, "same24_development_queries": True, "counted": False})
    records = []
    cache = QueryFeatureCache(args.feature_cache_root, original) if args.feature_mode == "cache" else None

    def extra(bank, path):
        started = time.perf_counter()
        value = original(bank, path) if cache is None else cache(bank, path)
        records.append({"name": Path(path).name, "seconds": time.perf_counter()-started,
                        "feature_count": len(value[0])})
        return value

    try:
        feature_localization.FeatureBank.extra = extra
        sys.argv = [str(Path(localize_with_neighbors.__file__)), "--mode", "snapshot", "--output", str(args.output/"entry"), *remaining]
        localize_with_neighbors.main()
        if identities != [data_identity(p) for p in sources]:
            raise ValueError("The query cache implementation changed during the entry")
        atomic_json(args.output/"FEATURE_CACHE_TRACE.json", {"status": "all24_actual_queries_completed",
            "mode": args.feature_mode, "records": records, "query_feature_seconds": sum(r["seconds"] for r in records),
            "cache": cache.receipt() if cache else None, "sources_unchanged": True})
    finally:
        feature_localization.FeatureBank.extra = original
        sys.argv = original_argv


if __name__ == "__main__":
    main()
