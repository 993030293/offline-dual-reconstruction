"""Exercise the general admitted matcher in the unchanged current full pose entry."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import localize_integer_sift
from generic_sift_matching import generic_sift_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("generic_sift_worker.py"),
        Path(__file__).with_name("generic_sift_matching.py"), Path(localize_integer_sift.__file__)]
    identity = [data_identity(p) for p in sources]
    atomic_json(args.output/"GENERIC_PROTOCOL.json", {"sources": identity, "all_original24_48_protections": True,
        "rows_admitted_by_existing_matrix_and_transport_budgets": True, "new_optimization_count": 0,
        "original_R90_sources_unchanged": True, "different_worker_lifecycle_cost_measured_separately": True})
    previous_argv, previous = sys.argv, localize_integer_sift.integer_sift_matching
    try:
        localize_integer_sift.integer_sift_matching = generic_sift_matching
        sys.argv = [str(Path(localize_integer_sift.__file__)), "--matching-mode", "integer",
            "--output", str(args.output/"entry"), *remaining]
        localize_integer_sift.main()
        if identity != [data_identity(p) for p in sources]:
            raise ValueError("A general matcher source changed during the actual route")
        atomic_json(args.output/"RESULT.json", {"status": "actual_generic_pose_entry_completed_comparison_pending",
            "sources": identity, "new_optimization_count": 0})
    finally:
        sys.argv, localize_integer_sift.integer_sift_matching = previous_argv, previous


if __name__ == "__main__":
    main()
