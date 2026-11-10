"""Run current complete input and resource checks before a private path trial."""

import scene_fingerprint_preflight as parent


if __name__ == "__main__":
    parent.OUT = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/spr004"
    parent.main()
