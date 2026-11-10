"""Apply metadata omission only around a full native Scene input comparison."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
import validate_camera_prefetch
from training_camera_metadata import training_metadata_reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-mode", choices=["baseline", "metadata"], required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("training_camera_metadata.py"), Path(validate_camera_prefetch.__file__)]
    sources = [data_identity(p) for p in paths]
    atomic_json(args.output/"CONTRACT.json", {"mode": args.metadata_mode, "sources": sources,
        "common_prefetch_mode": "prefetch", "observation_source_and_full_training_input_unchanged": True,
        "minimum_reader_stage_speed_or_Scene_speed_or_peak_RAM_reduction": .1,
        "absolute_reader_and_complete_Scene_costs_disclosed": True, "complete_training_quality_pending": True,
        "counted": False})
    native_activate = validate_camera_prefetch.activate_reference
    runtime = activate_reference(args.runtime)
    original_argv = sys.argv
    try:
        validate_camera_prefetch.activate_reference = lambda _: runtime
        with training_metadata_reader(args.metadata_mode == "metadata") as records:
            sys.argv = [str(Path(validate_camera_prefetch.__file__)), "--mode", "prefetch", "--runtime", str(args.runtime),
                "--output", str(args.output/"scene_entry"), *remaining]
            validate_camera_prefetch.main()
        if sources != [data_identity(p) for p in paths] or len(records) != 1:
            raise ValueError("The full Scene metadata entry or sources changed")
        result = json.loads((args.output/"scene_entry/RESULT.json").read_text())
        atomic_json(args.output/"METADATA_TRACE.json", {"status": "actual_Scene_completed", "sources": sources,
            "mode": args.metadata_mode, "records": records, "Scene_load_seconds": result["Scene_load_seconds"], "counted": False})
    finally:
        validate_camera_prefetch.activate_reference = native_activate
        sys.argv = original_argv


if __name__ == "__main__":
    main()
