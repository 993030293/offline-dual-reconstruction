"""Run complete raw selection with unchanged or exact memoized calibration parsing."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
import run_effective_initial
from calibration_body_cache import calibration_body_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-mode", choices=("incumbent", "native", "memo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The raw calibration output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("calibration_body_cache.py"), Path(run_effective_initial.__file__)]
    sources = [data_identity(p) for p in paths]
    contract = args.output.with_name(args.output.name+"_calibration_contract.json")
    if contract.exists():
        raise FileExistsError("The raw calibration source binding already exists")
    atomic_json(contract, {"status": "frozen_before_actual_entry", "mode": args.calibration_mode,
        "sources": sources, "all7338_headers_and_full_body_bytes_checked_on_memo_route": True,
        "original_invariant_intrinsics_check_and_selection_unchanged": True, "counted": False})
    previous = sys.argv
    try:
        context = nullcontext() if args.calibration_mode == "incumbent" else calibration_body_execution(args.calibration_mode == "memo")
        with context as setup:
            sys.argv = [str(Path(run_effective_initial.__file__)), "--mode", "cold", "--output", str(args.output), "--quality-cache", str(args.quality_cache)]
            run_effective_initial.main()
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("The raw calibration source changed during complete initial selection")
        atomic_json(args.output/"CALIBRATION.json", {"status": "actual_raw_calibration_entry_finished_protection_pending",
            "sources": sources, "mode": args.calibration_mode, "additional_source_composition_seconds": setup["composition_seconds"] if setup else 0., "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
