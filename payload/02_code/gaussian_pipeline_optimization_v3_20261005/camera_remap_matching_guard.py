"""Bound the unchanged native matcher equally on both remapping experiment sides."""

import argparse
import json
from pathlib import Path
import runpy
import sys

from mapped_image_linux_guard import LinuxRSSGuard


def publish(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.native.resolve() != Path(__file__).with_name("integer_sift_worker.py").resolve():
        raise ValueError("Only the unchanged owned native matcher can run")
    args.resource.resolve().relative_to(root / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001")
    sys.argv = [str(args.native)]
    with LinuxRSSGuard(args.resource.parent, publish, maximum_bytes=8 * 1024**3):
        runpy.run_path(str(args.native), run_name="__main__")


if __name__ == "__main__":
    main()
