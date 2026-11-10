"""Exercise exact byte descriptors through the complete cached localization entry."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
sys.path.insert(0, str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"))
from runtime import atomic_json, data_identity
import feature_localization
import localize_with_cached_references
from byte_sift_matching import byte_sift_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-mode", choices=["baseline", "bytes"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("byte_sift_matching.py"),
        Path(localize_with_cached_references.__file__), Path(feature_localization.__file__)]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"BYTE_SIFT_CONTRACT.json", {"mode": args.descriptor_mode, "sources": identities,
        "maximum_seconds": 900, "maximum_RSS_GiB": 8, "original_float32_normalized_arrays_recover_exactly": True,
        "matching_outputs_exact": True, "cache_capacity_unchanged": 32, "new_pose_accuracy_claim": False, "counted": False})
    original_argv = sys.argv
    timings = []
    try:
        with byte_sift_matching(args.descriptor_mode == "bytes") as trace:
            original_match = feature_localization.FeatureBank.match

            def timed(bank, *values, **kwargs):
                started = time.perf_counter()
                result = original_match(bank, *values, **kwargs)
                timings.append(time.perf_counter()-started)
                return result

            feature_localization.FeatureBank.match = timed
            try:
                sys.argv = [str(Path(localize_with_cached_references.__file__)), "--output", str(args.output/"reference_entry"), *remaining]
                localize_with_cached_references.main()
            finally:
                feature_localization.FeatureBank.match = original_match
        if identities != [data_identity(p) for p in sources]:
            raise ValueError("A byte descriptor adapter changed during the actual entry")
        atomic_json(args.output/"BYTE_SIFT_TRACE.json", {"status": "actual24_byte_matching_completed", "mode": args.descriptor_mode,
            "match_seconds": sum(timings), "all48_match_calls": len(timings), "mechanism_calls": trace,
            "sources_unchanged": True, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
