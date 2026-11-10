"""Run complete protected localization with one isolated current-frame pipeline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import localize_integer_sift
import localize_with_locked_inputs
from paired_match_pipeline import pipeline_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["incumbent", "serial", "parallel"], required=True)
    parser.add_argument("--distance-worker", choices=["legacy", "generic"], default="generic")
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("paired_match_pipeline.py"), Path(localize_with_locked_inputs.__file__),
        Path(__file__).with_name("generic_sift_worker.py"), Path(__file__).with_name("generic_sift_matching.py")]
    identity = [data_identity(p) for p in paths]
    atomic_json(args.output/"PIPELINE_PROTOCOL.json", {"sources": identity, "mode": args.mode,
        "distance_worker": args.distance_worker, "one_current_frame_no_future_or_GT": True,
        "maximum_CPU_distance_tasks": 1, "maximum_GPU_distance_workers": 1, "all48_original_matches_and24_poses": True,
        "minimum_matching_phase_reduction_vs_same_serial_and_incumbent": .1,
        "whole_entry_must_not_regress": True, "source_binding_startup_prepare_and_queues_included": True, "counted": False})
    previous_argv, previous = sys.argv, localize_integer_sift.integer_sift_matching
    try:
        localize_integer_sift.integer_sift_matching = lambda log, enabled=True: pipeline_context(log, args.mode, args.distance_worker, enabled)
        sys.argv = [str(Path(localize_with_locked_inputs.__file__)), "--snapshot-mode", "locked",
            "--output", str(args.output/"entry"), *remaining]
        localize_with_locked_inputs.main()
        if identity != [data_identity(p) for p in paths]:
            raise ValueError("A measured pipeline source changed during localization")
        atomic_json(args.output/"RESULT.json", {"status": "actual_pair_pipeline_complete_acceptance_pending", "sources": identity, "counted": False})
    finally:
        sys.argv, localize_integer_sift.integer_sift_matching = previous_argv, previous


if __name__ == "__main__":
    main()
