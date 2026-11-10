"""Bind exact camera execution to the native complete training and recovery identity."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import train_case
import checkpoint_adapter
from prefetch_camera_images import prefetch_camera_images
from camera_execution import camera_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    config = json.loads(args.camera_config.read_text())
    flags = ("opaque_alpha", "clampfree", "calibrated_projection")
    if set(config) != {"schema_version", *flags} or config["schema_version"] != 1 or any(type(config[key]) is not bool for key in flags):
        raise ValueError("The native camera execution configuration changed its explicit schema")

    def argument(name):
        if remaining.count(name) != 1:
            raise ValueError("The native camera entry requires explicit "+name)
        return remaining[remaining.index(name)+1]

    if argument("--projection") != "full" or int(argument("--iterations")) not in (1100, 12000) or argument("--expected-image-count") != "1016":
        raise ValueError("The native exact-camera campaign requires its frozen full-K1016 input")
    if "--gsplat-build" not in remaining or any(name in remaining for name in
            ("--gpu-resident-inputs", "--compact-host-images", "--compact-image-cache", "--pyramid-until", "--target-density-roi", "--resident-fused-adam")):
        raise ValueError("Unverified training formula or input representation combinations are excluded")
    output = Path(argument("--output"))
    contract = output.parent/(output.name+"_camera_contract.json")
    snapshot = output.parent/(output.name+"_camera_sources")
    if output.exists() or contract.exists() or snapshot.exists():
        raise FileExistsError("An exact-camera training output or frozen identity already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir()
    paths = [Path(__file__), Path(__file__).with_name("camera_execution.py"), Path(__file__).with_name("prefetch_camera_images.py")]
    identities = [data_identity(path) for path in paths]
    for path in paths:
        shutil.copyfile(path, snapshot/path.name)
    external = {"schema_version": 1, "camera_execution": config, "source_sha256": [row["sha256"] for row in identities],
        "common_bounded_prefetch": {"workers": 4, "capacity": 8}, "native_binary_metadata_reader": True}
    atomic_json(contract, {"status": "frozen_before_actual_training", "external_adapters": external, "sources": identities,
        "camera_config_source": str(args.camera_config), "arguments": remaining, "counted": False})
    original_loader, original_recovery = train_case.camera_loader_adapter, checkpoint_adapter.RecoveryState
    original_json, original_argv = train_case.atomic_json, sys.argv
    receipts = []

    class BoundRecoveryState(original_recovery):
        def __init__(self, output, identity, interval, resume=None, crash_after=None):
            if "external_adapters" in identity:
                raise ValueError("A native recovery identity already has external adapter bindings")
            super().__init__(output, {**identity, "external_adapters": external}, interval, resume, crash_after)

    @contextmanager
    def adapted(dataset, enabled=True):
        with (original_loader(dataset, enabled=enabled) as rows,
                camera_execution(**{key: config[key] for key in flags}) as execution,
                prefetch_camera_images(4, 8) as prefetch):
            yield rows
        receipts.append({"camera_execution": execution, "prefetch": prefetch})

    def publish(path, value):
        if Path(path) == output/"CONFIG.json":
            value = {**value, "external_adapters": external, "external_adapter_sources": identities}
        original_json(path, value)

    try:
        train_case.camera_loader_adapter, checkpoint_adapter.RecoveryState = adapted, BoundRecoveryState
        train_case.atomic_json = publish
        sys.argv = [str(Path(train_case.__file__)), *remaining]
        train_case.main()
        state = json.loads((output/"STATE.json").read_text())
        if identities != [data_identity(path) for path in paths] or state["status"] != "completed" or len(receipts) != 1:
            raise ValueError("The native exact-camera entry lost its source identity or full Scene")
        receipt = receipts[0]
        if (receipt["prefetch"]["prepared"] != 1016 or receipt["prefetch"]["consumed"] != 1016 or
                receipt["prefetch"]["maximum_pending"] > 8 or receipt["camera_execution"]["native_camera_calls"] != 1016):
            raise ValueError("A native exact-camera input was missing or exceeded its declared queue")
        atomic_json(output/"CAMERA_EXECUTION_RESULT.json", {"status": "actual_training_completed_acceptance_pending",
            "sources": identities, "external_adapters": external, "receipts": receipts, "counted": False})
    finally:
        train_case.camera_loader_adapter, checkpoint_adapter.RecoveryState = original_loader, original_recovery
        train_case.atomic_json, sys.argv = original_json, original_argv


if __name__ == "__main__":
    main()
