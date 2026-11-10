"""Validate combined camera execution using the original all-camera pixel and state oracle."""

import argparse
import json
from pathlib import Path
import sys

from camera_execution import camera_execution
import load_with_camera_specialization as checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    config = json.loads(args.camera_config.read_text())
    flags = ("opaque_alpha", "clampfree", "calibrated_projection")
    if set(config) != {"schema_version", *flags} or config["schema_version"] != 1 or any(type(config[key]) is not bool for key in flags):
        raise ValueError("The combined camera execution schema changed")
    original_context, original_json, original_argv = checker.specialized_training_camera, checker.validator.atomic_json, sys.argv

    def context(mode):
        return camera_execution(**{key: config[key] for key in flags})

    def publish(path, value):
        if Path(path).name == "RESULT.json":
            value = {**value, "actual_combination_config": config,
                "actual_combination_sources": [checker.data_identity(Path(__file__)), checker.data_identity(args.camera_config),
                    checker.data_identity(Path(__file__).with_name("camera_execution.py"))]}
        original_json(path, value)

    try:
        checker.specialized_training_camera, checker.validator.atomic_json = context, publish
        sys.argv = [str(Path(checker.__file__)), "--camera-mode", "opaque" if config["opaque_alpha"] else "baseline", *remaining]
        checker.main()
    finally:
        checker.specialized_training_camera, checker.validator.atomic_json, sys.argv = original_context, original_json, original_argv


if __name__ == "__main__":
    main()
