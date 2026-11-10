"""Bind the unchanged full original resource controller to the diagnostic tree."""

from pathlib import Path

import sift_process_node_003 as original


def main():
    previous = original.OUT
    try:
        original.OUT = Path(__file__).resolve().parents[2] / "04_outputs/gaussian_pipeline_optimization_v3_20261005/olp001"
        original.main()
    finally:
        original.OUT = previous


if __name__ == "__main__":
    main()
