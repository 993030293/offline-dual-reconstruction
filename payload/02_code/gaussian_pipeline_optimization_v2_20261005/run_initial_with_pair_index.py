"""Exercise indexed candidate pairing through the complete raw-only initial entry."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_initial_with_jpeg_cache
import run_initial_pipeline
from indexed_candidate_pairs import indexed_pair_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-mode", choices=["baseline", "indexed"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("indexed_candidate_pairs.py"),
        Path(run_initial_pipeline.__file__), Path(run_initial_with_jpeg_cache.__file__)]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"PAIR_INDEX_CONTRACT.json", {"mode": args.pair_mode, "sources": identities,
        "maximum_seconds": 180, "maximum_RSS_GiB": 8, "minimum_pair_stage_reduction": .1,
        "all7338_quality_decisions1002_images11583_pairs_exact": True,
        "initial_selection_uses_only_raw_pose_and_quality": True,
        "old_camera_snapshot_mechanism_not_recounted": True, "counted": False})
    original_argv = sys.argv
    try:
        with indexed_pair_generation(run_initial_pipeline, args.pair_mode == "indexed") as records:
            sys.argv = [str(Path(run_initial_with_jpeg_cache.__file__)), "--output", str(args.output/"raw_entry"), *remaining]
            run_initial_with_jpeg_cache.main()
        if len(records) != 1 or identities != [data_identity(p) for p in sources]:
            raise ValueError("The full raw entry changed its single pairing call or sources")
        atomic_json(args.output/"PAIR_INDEX_TRACE.json", {"status": "full_raw_entry_completed", "records": records,
            "sources_unchanged": True, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
