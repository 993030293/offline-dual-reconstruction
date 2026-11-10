"""Run the established bounded controller in the strict-encoding private root."""

from pathlib import Path

import codec_localization_node

OUT = Path(__file__).resolve().parents[2] / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/R171"


def main():
    function = codec_localization_node.compose()
    function.__globals__.update(OUT=OUT, __file__=str(Path(__file__).resolve()))
    function()


if __name__ == "__main__":
    main()
