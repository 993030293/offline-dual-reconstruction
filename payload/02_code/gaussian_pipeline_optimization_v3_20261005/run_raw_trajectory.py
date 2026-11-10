"""Run the original full initial route with one complete trajectory representation."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
import run_effective_initial
from typed_odometry_cdr import typed_odometry_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=("native", "typed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The full raw trajectory output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = args.output.with_name(args.output.name+"_trajectory_contract.json")
    if contract.exists():
        raise FileExistsError("The trajectory contract already exists")
    paths = [Path(__file__), Path(__file__).with_name("typed_odometry_cdr.py"), Path(run_effective_initial.__file__)]
    identities = [data_identity(path) for path in paths]
    atomic_json(contract, {"status": "frozen_before_actual_raw_entry", "sources": identities, "decoder": args.decoder,
        "same_unaccepted_R89_native_CDR_family_only_one_if_accepted": True,
        "all7338_raw_only_decisions1002_images11583_pairs_and_all_odometry_preserved": True,
        "all36_covariance_values_preserved_not_assumed_zero": True, "counted": False})
    previous = sys.argv
    try:
        with typed_odometry_execution(args.decoder == "typed"):
            sys.argv = [str(Path(run_effective_initial.__file__)), "--mode", "cold", "--quality-cache", str(args.quality_cache), "--output", str(args.output)]
            run_effective_initial.main()
        if identities != [data_identity(path) for path in paths]:
            raise ValueError("A complete trajectory source changed during the actual raw entry")
        atomic_json(args.output/"TRAJECTORY.json", {"status": "actual_raw_entry_completed_protection_pending", "sources": identities,
            "decoder": args.decoder, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
