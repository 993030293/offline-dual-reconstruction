"""Use the accepted training adapters with a private resolution-one configuration."""

import importlib.util
import json
from pathlib import Path
import sys


def main():
    arguments = sys.argv[1:]
    installed = Path(arguments[arguments.index("--runtime") + 1])
    dataset = Path(arguments[arguments.index("--dataset") + 1])
    private = dataset.parent / "gaussian_runtime"
    private.mkdir(exist_ok=False)
    for name in ("source", "runtime"):
        (private / name).symlink_to(installed / name, target_is_directory=True)
    configuration = json.loads((installed / "MAIN_CONFIG.json").read_text())
    previous = configuration["parameters"]["resolution"]
    configuration["parameters"]["resolution"] = 1
    (private / "MAIN_CONFIG.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (private / "RESOLUTION_CONTRACT.json").write_text(json.dumps({
        "installed_configuration": str(installed / "MAIN_CONFIG.json"),
        "previous_resolution_divisor": previous, "actual_resolution_divisor": 1,
        "reason": "Shared inputs already have their admitted training resolution",
        "historical_trained_model_read": False, "installed_source_and_extensions_reused": True,
        "other_native_training_parameters_changed": False}, indent=2) + "\n")
    arguments[arguments.index("--runtime") + 1] = str(private)
    old = Path(__file__).resolve().parents[1] / "gaussian_full_raw_ab_20261008"
    sys.path.insert(0, str(old))
    spec = importlib.util.spec_from_file_location("dual_native_gaussian_execution", old / "train_variant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(old / "train_variant.py"), *arguments]
    module.main()


if __name__ == "__main__":
    main()
