"""Run the unchanged complete24 geometry route with one owned distance worker."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import localize_with_byte_sift
from integer_sift_matching import integer_sift_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-mode", choices=["native", "integer"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("integer_sift_worker.py"),
        Path(__file__).with_name("integer_sift_matching.py"), Path(localize_with_byte_sift.__file__)]
    identity = [data_identity(p) for p in sources]
    atomic_json(args.output/"INTEGER_SIFT_CONTRACT.json", {"sources": identity,
        "mode": args.matching_mode, "exact_native_L2_and_mutual_ratio_and_F_and_PnP": True,
        "all24_original_queries": True, "maximum_seconds": 900, "maximum_RSS_GiB": 8,
        "maximum_CUDA_GiB": 4, "original_float32_distance_rounding": True,
        "worker_startup_and_IPC_in_whole_process_cost": True, "counted": False})
    original_argv = sys.argv
    try:
        with integer_sift_matching(args.output/"worker.log", args.matching_mode == "integer") as trace:
            sys.argv = [str(Path(localize_with_byte_sift.__file__)), "--descriptor-mode", "bytes",
                "--output", str(args.output/"distance_entry"), *remaining]
            localize_with_byte_sift.main()
        if identity != [data_identity(p) for p in sources]:
            raise ValueError("An exact integer SIFT implementation changed during localization")
        atomic_json(args.output/"INTEGER_SIFT_TRACE.json", {"status": "actual24_completed",
            "trace": trace, "sources": identity, "all48_geometry_calls_unchanged": True, "counted": False})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
