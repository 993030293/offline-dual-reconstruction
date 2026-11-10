"""Run the original complete cold initial pipeline with owned native planes."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from native_quality_workspace import native_quality_workspace_execution
import run_raw_histogram_quality as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "workspace"), required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), Path(__file__).with_name("native_quality_workspace.py"), Path(original.__file__)]
    sources = [data_identity(path) for path in paths]
    previous = sys.argv
    try:
        with native_quality_workspace_execution(args.mode == "workspace") as trace:
            sys.argv = [str(Path(original.__file__)), "--quality-mode", "histogram", "--quality-cache", str(args.quality_cache),
                        "--output", str(args.output)]
            original.main()
        if not trace["original_builder_restored"] or sources != [data_identity(path) for path in paths]:
            raise ValueError("The owned native workspace source or builder did not restore")
        atomic_json(args.output / "WORKSPACE.json", {"sources": sources, "mode": args.mode, "trace": trace,
            "additional_workspace_lifecycle_seconds": trace["setup_seconds"] + trace["restore_seconds"],
            "same_original_R104_R99_Q78_Q53_P18_input_quality_and_export": True, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
