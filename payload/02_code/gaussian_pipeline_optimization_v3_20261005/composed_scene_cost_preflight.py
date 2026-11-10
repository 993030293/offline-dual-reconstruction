"""Reuse original full-input environment checks for the fresh diagnostic."""

import scene_fingerprint_preflight as parent


if __name__ == "__main__":
    parent.OUT = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"
    parent.main()
