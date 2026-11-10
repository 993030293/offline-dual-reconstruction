"""Compare original durable writes and a drained private write pipeline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from quality_histogram import histogram_quality_execution
from quality_persistence_pipeline import quality_persistence_execution
import run_raw_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-mode", choices=("native", "pipeline"), required=True)
    parser.add_argument("--cache-temperature", choices=("cold", "warm"), default="cold")
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The complete raw output must be new")
    if (args.cache_temperature == "cold" and args.quality_cache.exists()
            or args.cache_temperature == "warm" and not args.quality_cache.is_dir()):
        raise ValueError("The quality cache does not have its registered temperature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(__file__).with_name("quality_persistence_pipeline.py"),
        Path(__file__).with_name("quality_histogram.py"), Path(run_raw_calibration.__file__),
        Path(__file__).with_name("calibration_body_cache.py"),
        ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005/quality_engine.py"]
    sources = [data_identity(path) for path in paths]
    previous = sys.argv
    try:
        with histogram_quality_execution(True) as histogram:
            with quality_persistence_execution(args.write_mode == "pipeline", args.quality_cache) as writer:
                sys.argv = [str(Path(run_raw_calibration.__file__)), "--calibration-mode", "memo",
                    "--output", str(args.output), "--quality-cache", str(args.quality_cache)]
                run_raw_calibration.main()
        if (sources != [data_identity(path) for path in paths]
                or not histogram["aliases_restored"] or not writer["original_functions_restored"]):
            raise ValueError("An original raw scoring source or binding was not preserved")
        additional = (histogram["setup_seconds"]+histogram["binding_setup_seconds"]
            +histogram["restore_seconds"]+writer["setup_seconds"]+writer["restore_seconds"])
        atomic_json(args.output/"PERSISTENCE.json", {"sources": sources, "mode": args.write_mode,
            "histogram_trace": histogram, "writer_trace": writer,
            "additional_quality_lifecycle_seconds": additional,
            "quality_cache": str(args.quality_cache), "cache_temperature": args.cache_temperature,
            "same_current_Q78_Q50_Q53_P18_R99_R104_JPEG_cache_off": True,
            "all_actual_write_waits_shutdown_and_original_fsync_in_scoring_wall": True, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
