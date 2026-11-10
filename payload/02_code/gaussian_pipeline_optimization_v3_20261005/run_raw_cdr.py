"""Run the accepted raw route with an explicit exact odometry decoder comparison."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import run_effective_initial
from bulk_odometry_cdr import bulk_odometry_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=["native", "bulk"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The native bulk decoder output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("bulk_odometry_cdr.py"), Path(run_effective_initial.__file__)]
    identities = [data_identity(path) for path in paths]
    contract = args.output.with_name(args.output.name+"_decoder_contract.json")
    if contract.exists():
        raise FileExistsError("The raw decoder contract already exists")
    atomic_json(contract, {"status": "frozen_before_actual_raw_entry", "decoder": args.decoder,
        "sources": identities, "current_accepted_Q78_Q50_Q53_common": True, "quality_cache_common_warm": True,
        "raw_selection_and_all_odometry_values_unchanged": True, "counted": False})
    original_argv = sys.argv
    try:
        with bulk_odometry_execution(args.decoder == "bulk"):
            sys.argv = [str(Path(run_effective_initial.__file__)), "--mode", "cold", "--output", str(args.output),
                "--quality-cache", str(args.quality_cache)]
            run_effective_initial.main()
        if identities != [data_identity(path) for path in paths]:
            raise ValueError("An exact raw CDR implementation changed during the entry")
        atomic_json(args.output/"DECODER.json", {"status": "actual_complete_raw_decoder_finished", "decoder": args.decoder,
            "sources": identities, "full_output_protection_pending": True, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
