"""Reuse current full-input resource checks before the fresh phase campaign."""

import scene_fingerprint_preflight as parent


if __name__ == "__main__":
    parent.OUT = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sip001"
    parent.main()
