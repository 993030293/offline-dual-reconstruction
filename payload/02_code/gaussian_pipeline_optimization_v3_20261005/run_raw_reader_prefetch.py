"""Keep the complete accepted RAW pipeline and change only owned reader scheduling."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_raw_quality_workspace as original
from raw_reader_prefetch import raw_reader_prefetch_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "prefetch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWN)
    args.quality_cache.resolve().relative_to(OWN)
    paths = [Path(__file__), Path(__file__).with_name("raw_reader_prefetch.py"), Path(original.__file__)]
    sources = [data_identity(path) for path in paths]
    previous = sys.argv
    try:
        with raw_reader_prefetch_execution(args.mode == "prefetch") as trace:
            sys.argv = [str(Path(original.__file__)), "--mode", "workspace", "--output", str(args.output),
                        "--quality-cache", str(args.quality_cache)]
            original.main()
        if not trace["binding_restored"] or sources != [data_identity(path) for path in paths]:
            raise ValueError("The owned RAW reader source or original scoring binding changed")
        if args.mode == "prefetch" and (len(trace["calls"]) != 1 or trace["calls"][0]["delivered"] != 7338
                or not trace["calls"][0]["closed"] or trace["calls"][0]["reader_alive_after_join"]):
            raise ValueError("The complete original RAW denominator or producer closure changed")
        atomic_json(args.output / "READER.json", {"sources": sources, "mode": args.mode, "trace": trace,
            "additional_reader_lifecycle_seconds": trace["setup_seconds"] + trace["restore_seconds"],
            "common_original_R141_R104_R99_Q78_Q53_P18_and4worker16queue": True, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
