"""Run all current poses with original or bounded independent-neighbor F scheduling."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import localize_native_observations
from parallel_neighbor_geometry import parallel_neighbor_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-mode", choices=("incumbent", "serial", "parallel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.output.exists():
        raise FileExistsError("The actual neighbor geometry output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("parallel_neighbor_geometry.py"),
        Path(localize_native_observations.__file__), Path(__file__).with_name("native_observation_index.py"),
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py"]
    sources = [data_identity(path) for path in paths]
    old = sys.argv
    try:
        with parallel_neighbor_geometry(args.geometry_mode) as trace:
            sys.argv = [str(Path(localize_native_observations.__file__)), "--mode", "index", "--output", str(args.output), *remaining]
            localize_native_observations.main()
        index = json.loads((args.output/"INDEX_TRACE.json").read_text(encoding="utf-8"))
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("An actual parallel neighbor geometry source changed")
        atomic_json(args.output/"GEOMETRY.json", {"status": "actual_complete_neighbor_geometry_protection_pending",
            "sources": sources, "mode": args.geometry_mode, "trace": trace,
            "complete_CPU_geometry_seconds": index["trace"]["CPU_geometry_seconds_including_all_snapshot_and_verify"]
                +trace["setup_seconds"]+trace["shutdown_seconds"],
            "complete_original_index_trace": index["trace"], "counted": False})
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
