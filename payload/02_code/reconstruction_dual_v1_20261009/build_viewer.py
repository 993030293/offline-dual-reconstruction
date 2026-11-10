"""Bind the preserved generic renderer to an explicitly named recording case."""

import argparse
import importlib.util
import sys

from common import OLD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args, _ = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("dual_native_viewer_builder", OLD / "build_raw_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.case not in module.CASES:
        module.CASES[args.case] = args.case
    module.main()


if __name__ == "__main__":
    main()
