"""Run the unchanged complete coarse-pose selection through exact single-copy raw input."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import run_initial_pipeline
from single_copy_raw_reader import single_copy_sensor_reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "quality-cache"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--single-copy", choices=["true", "false"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("single_copy_raw_reader.py"), V1/"run_initial_pipeline.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"RUNTIME_CONTRACT.json", {"sources": sources, "single_copy": args.single_copy,
        "all7338_raw_only_decisions1002_JPEG11583_pairs_and_odometry_exact": True, "no_frame_reduction_or_SfM_prior_selection": True,
        "maximum_seconds": 180, "maximum_RSS_GiB": 8, "counted": False})
    original_argv = sys.argv
    try:
        with single_copy_sensor_reader(args.single_copy == "true") as trace:
            sys.argv = [str(V1/"run_initial_pipeline.py"), "--output", str(args.output/"initial"),
                "--cache", str(args.quality_cache), "--workers", "4", "--scorer", "compact",
                "--selector", "batched", "--pose-mode", "batched", "--reader", "indexed"]
            run_initial_pipeline.main()
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("A single-copy raw entry source changed during actual selection")
        atomic_json(args.output/"RUNTIME_TRACE.json", {"sources": sources, "single_copy_reader": trace,
            "numeric_score_cache": None, "JPEG_cache": None, "status": "actual_full_raw_entry_completed", "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
