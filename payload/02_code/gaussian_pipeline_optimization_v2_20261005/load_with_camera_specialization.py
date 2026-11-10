"""Fingerprint every real Scene camera with an explicit opaque-alpha equivalence oracle."""

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import textwrap
import inspect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import validate_camera_prefetch as validator
from training_camera_metadata import training_metadata_reader
from specialized_training_camera import specialized_training_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-mode", choices=["baseline", "opaque", "clampfree", "projection"], required=True)
    args, remaining = parser.parse_known_args()
    original_main, original_loader, original_json, original_argv = validator.main, validator.camera_loader_adapter, validator.atomic_json, sys.argv
    traces = []
    alpha_checks = {"all_nonempty_alpha_tensors_exactly_ones": True, "opaque_cameras": 0, "native_alpha_cameras": 0}

    def opaque(camera):
        if camera.alpha_mask is None:
            if args.camera_mode != "opaque":
                raise ValueError("A non-opaque camera mode lost its native alpha mask")
            alpha_checks["opaque_cameras"] += 1
        else:
            import torch
            if not torch.equal(camera.alpha_mask, torch.ones_like(camera.alpha_mask)):
                raise ValueError("An original camera has transparency or a nontrivial image mask")
            alpha_checks["native_alpha_cameras"] += 1
        return {"device": "cpu", "effective_opaque_RGB": True, "shape": [1, camera.image_height, camera.image_width]}

    source = inspect.getsource(original_main)
    before = '"alpha": tensor(camera.alpha_mask)'
    if source.count(before) != 1:
        raise ValueError("The native field validator alpha oracle changed")
    source = source.replace(before, '"alpha": opaque_alpha(camera)')
    namespace = dict(original_main.__globals__)
    namespace["opaque_alpha"] = opaque
    exec(compile(textwrap.dedent(source), "<full_Scene_camera_specialization_oracle>", "exec"), namespace)
    compiled = namespace["main"]

    @contextmanager
    def loader(dataset, enabled=True):
        if not enabled:
            raise ValueError("The camera specialization requires full calibrated projection")
        with original_loader(dataset, enabled=enabled) as rows, training_metadata_reader(False) as metadata, specialized_training_camera(args.camera_mode) as trace:
            yield rows
        traces.append({"specialization": trace, "metadata": metadata})

    def publish(path, value):
        if Path(path).name == "RESULT.json":
            value = {**value, "camera_specialization": traces, "opaque_alpha_equivalence": alpha_checks,
                "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("specialized_training_camera.py"))]}
        original_json(path, value)

    try:
        validator.camera_loader_adapter, validator.atomic_json = loader, publish
        compiled.__globals__["camera_loader_adapter"] = loader
        compiled.__globals__["atomic_json"] = publish
        sys.argv = [str(Path(validator.__file__)), "--mode", "prefetch", *remaining]
        compiled()
    finally:
        validator.main, validator.camera_loader_adapter, validator.atomic_json, sys.argv = original_main, original_loader, original_json, original_argv


if __name__ == "__main__":
    main()
