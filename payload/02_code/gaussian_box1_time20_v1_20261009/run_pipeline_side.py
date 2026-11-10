"""Run the preserved complete side, optionally binding sparse augmentation only."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("reference", "sparse"), required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("preserved_full_side", OLD / "run_side.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_run = subprocess.run
    old_argv = sys.argv

    def dispatch(command, *parameters, **keywords):
        values = list(map(str, command))
        original_stage = str(OLD / "run_native_stage.py")
        if args.route == "sparse" and original_stage in values and "--kind" in values:
            if values[values.index("--kind")+1] == "temporal":
                values[values.index(original_stage)] = str(Path(__file__).with_name("run_native_sparse.py"))
        return original_run(values, *parameters, **keywords)

    subprocess.run = dispatch
    sys.argv = [str(OLD / "run_side.py"), *remaining]
    try:
        native.main()
    finally:
        subprocess.run = original_run
        sys.argv = old_argv


if __name__ == "__main__":
    main()
