"""Run every raw frame with native or equivalent histogram quality and empty caches."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from quality_histogram import histogram_quality_execution
import run_raw_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-mode", choices=("native", "histogram"), required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.quality_cache.exists():
        raise FileExistsError("Both the complete raw output and cold quality cache must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("quality_histogram.py"), Path(run_raw_calibration.__file__),
        Path(__file__).with_name("calibration_body_cache.py"), ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005/quality_engine.py"]
    sources = [data_identity(path) for path in paths]
    previous = sys.argv
    try:
        with histogram_quality_execution(args.quality_mode == "histogram") as trace:
            sys.argv = [str(Path(run_raw_calibration.__file__)), "--calibration-mode", "memo",
                "--output", str(args.output), "--quality-cache", str(args.quality_cache)]
            run_raw_calibration.main()
        if sources != [data_identity(path) for path in paths] or not trace["aliases_restored"]:
            raise ValueError("An actual cold histogram source or quality alias changed")
        atomic_json(args.output/"HISTOGRAM.json", {"sources": sources, "mode": args.quality_mode, "trace": trace,
            "additional_quality_lifecycle_seconds": trace["setup_seconds"]+trace["binding_setup_seconds"]+trace["restore_seconds"],
            "cold_quality_cache": str(args.quality_cache), "cache_did_not_exist_before_run": True,
            "same_current_Q78_Q50_Q53_P18_R99_JPEG_cache_off": True,
            "original_score_cache_schema_private_and_exact_values_checked_after_run": True, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
