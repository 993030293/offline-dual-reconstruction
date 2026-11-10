"""Run every original development pose through an isolated native observation index."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
from native_observation_index import native_observation_index
import localize_with_locked_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "index"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("native_observation_index.py"),
        Path(__file__).with_name("binary_map_qa.py"), ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py",
        Path(localize_with_locked_inputs.__file__)]
    identity = [data_identity(p) for p in paths]
    atomic_json(args.output/"INDEX_PROTOCOL.json", {"mode": args.mode, "sources": identity,
        "same_current_R90_R92_and_original_24_48": True, "complete_native_snapshot_before_after": True,
        "minimum_complete_CPU_geometry_reduction": .1, "whole_entry_cannot_regress": True,
        "geometry_build_and_all_final_verification_cost_included": True, "counted": False})
    previous = sys.argv
    try:
        with native_observation_index(args.output, args.mode == "index") as trace:
            sys.argv = [str(Path(localize_with_locked_inputs.__file__)), "--snapshot-mode", "locked",
                "--output", str(args.output/"entry"), *remaining]
            localize_with_locked_inputs.main()
        if identity != [data_identity(p) for p in paths]:
            raise ValueError("A native observation index source changed during the actual entry")
        atomic_json(args.output/"INDEX_TRACE.json", {"status": "actual24_index_cost_complete_acceptance_pending",
            "sources": identity, "trace": trace, "counted": False})
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "counted": False})
        raise
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
