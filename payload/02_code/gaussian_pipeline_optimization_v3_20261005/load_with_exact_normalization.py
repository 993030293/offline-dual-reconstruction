"""Run the original complete Scene oracle with one image conversion mechanism."""

import argparse
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import data_identity
import validate_camera_prefetch as validator
from camera_execution import camera_execution
from exact_image_normalization import exact_image_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization-mode", choices=("native", "lookup"), required=True)
    args, remaining = parser.parse_known_args()
    original_loader, original_json, original_argv = validator.camera_loader_adapter, validator.atomic_json, sys.argv
    traces = []

    def alpha(camera):
        if camera.alpha_mask is not None:
            raise ValueError("The shared accepted RGB camera route unexpectedly allocated alpha")
        return {"shape": [1, camera.image_height, camera.image_width], "device": "cpu", "effective_opaque_RGB": True}

    source = inspect.getsource(validator.main)
    anchor = '"alpha": tensor(camera.alpha_mask)'
    if source.count(anchor) != 1:
        raise ValueError("The complete Scene oracle changed its alpha field")
    source = source.replace(anchor, '"alpha": effective_alpha(camera)')
    namespace = dict(validator.main.__globals__)
    namespace["effective_alpha"] = alpha
    exec(compile(textwrap.dedent(source), "<native_full_Scene_normalization_oracle>", "exec"), namespace)
    compiled = namespace["main"]

    @contextmanager
    def loader(dataset, enabled=True):
        with (original_loader(dataset, enabled=enabled) as rows,
              exact_image_normalization(args.normalization_mode == "lookup") as normalization,
              camera_execution(opaque_alpha=True, clampfree=True, calibrated_projection=True) as camera):
            yield rows
        traces.append({"normalization": normalization, "camera": camera})

    def publish(path, value):
        if Path(path).name == "RESULT.json":
            value = {**value, "normalization_mode": args.normalization_mode, "normalization_traces": traces,
                "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("exact_image_normalization.py"))]}
        original_json(path, value)

    try:
        compiled.__globals__["camera_loader_adapter"] = loader
        compiled.__globals__["atomic_json"] = publish
        sys.argv = [str(Path(validator.__file__)), "--mode", "prefetch", *remaining]
        compiled()
    finally:
        validator.camera_loader_adapter, validator.atomic_json, sys.argv = original_loader, original_json, original_argv


if __name__ == "__main__":
    main()
