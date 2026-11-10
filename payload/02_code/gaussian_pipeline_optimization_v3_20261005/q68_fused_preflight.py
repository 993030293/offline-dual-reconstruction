"""Reuse the original environment preflight in a fresh composition scope."""

from pathlib import Path
import scene_fingerprint_preflight as parent


if __name__ == "__main__":
    parent.OUT = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
    parent.main()
