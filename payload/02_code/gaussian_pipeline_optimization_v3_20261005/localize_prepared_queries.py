"""Execute current full localization with bounded next-reference preparation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import localize_native_observations
from query_preparation_pipeline import query_preparation_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-mode", choices=("incumbent", "serial", "parallel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.output.exists():
        raise FileExistsError("The actual prepared-query output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("query_preparation_pipeline.py"),
        Path(localize_native_observations.__file__), Path(__file__).with_name("native_observation_index.py"),
        ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005/localize_new_map_evaluation.py"]
    sources = [data_identity(path) for path in paths]
    old = sys.argv
    try:
        with query_preparation_pipeline(args.preparation_mode) as trace:
            sys.argv = [str(Path(localize_native_observations.__file__)), "--mode", "index", "--output", str(args.output), *remaining]
            localize_native_observations.main()
        index = json.loads((args.output/"INDEX_TRACE.json").read_text(encoding="utf-8"))
        if sources != [data_identity(path) for path in paths] or trace["query_loop_seconds"] is None:
            raise ValueError("An actual prepared-query source or complete stage timer changed")
        atomic_json(args.output/"PREPARATION.json", {"status": "actual_complete_prepared_query_protection_pending",
            "sources": sources, "mode": args.preparation_mode, "trace": trace,
            "complete_query_loop_and_final_map_verification_seconds": trace["query_loop_seconds"]
                +trace["composition_seconds"]+index["trace"]["index"]["complete_verify_seconds"],
            "complete_original_index_trace": index["trace"], "common_incumbent_only_adds_stage_clock_and_trace": True,
            "producer_closed_inside_native_main_before_outer_reference_cache_exit": True, "counted": False})
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
