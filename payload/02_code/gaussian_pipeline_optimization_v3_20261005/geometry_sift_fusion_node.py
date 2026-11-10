"""Scope the unchanged protected SIFT node to one private fusion output tree."""

from pathlib import Path

import sift_process_node_003 as node

OWNED = Path(__file__).resolve().parents[2]/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"


def main():
    original = node.OUT
    try:
        node.OUT = OWNED
        node.main()
    finally:
        node.OUT = original


if __name__ == "__main__":
    main()
