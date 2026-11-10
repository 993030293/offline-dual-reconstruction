"""Run original localization with bounded parallel native descriptor distances."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from parallel_descriptor_distances import parallel_descriptor_distances
import localize_with_byte_sift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-mode", choices=["baseline", "parallel"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("parallel_descriptor_distances.py")]
    sources = [data_identity(p) for p in paths]
    original_argv = sys.argv
    try:
        with parallel_descriptor_distances(args.matching_mode == "parallel") as trace:
            sys.argv = [str(Path(localize_with_byte_sift.__file__)), "--descriptor-mode", "bytes",
                "--output", str(args.output/"distance_entry"), *remaining]
            localize_with_byte_sift.main()
        if sources != [data_identity(p) for p in paths]:
            raise ValueError("A bounded native distance implementation changed")
        atomic_json(args.output/"PARALLEL_MATCHING_TRACE.json", {"status": "actual24_localization_completed",
            "trace": trace, "sources": sources, "mode": args.matching_mode, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
