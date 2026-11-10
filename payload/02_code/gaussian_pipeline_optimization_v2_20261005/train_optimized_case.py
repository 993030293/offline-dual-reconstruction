"""Bind isolated input adapters to the complete native training and recovery identity."""

import argparse
from contextlib import ExitStack, contextmanager
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
from training_camera_metadata import training_metadata_reader


def configuration(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"schema_version", "camera_prefetch", "training_metadata"} or config["schema_version"] != 1:
        raise ValueError("The explicit training adapter schema changed")
    prefetch = config["camera_prefetch"]
    if set(prefetch) != {"enabled", "workers", "capacity"} or type(prefetch["enabled"]) is not bool:
        raise ValueError("The prefetch adapter configuration is ambiguous")
    if prefetch["workers"] != 4 or prefetch["capacity"] != 8:
        raise ValueError("Only the resource-validated camera prefetch bounds are supported")
    metadata = config["training_metadata"]
    if set(metadata) != {"enabled"} or type(metadata["enabled"]) is not bool:
        raise ValueError("The training metadata adapter configuration is ambiguous")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-config", type=Path, required=True)
    args, training_args = parser.parse_known_args()
    config = configuration(args.adapter_config)
    for incompatible in ("--gpu-resident-inputs", "--compact-host-images", "--compact-image-cache",
                         "--pyramid-until", "--target-density-roi", "--resident-fused-adam"):
        if incompatible in training_args:
            raise ValueError("The native input adapters exclude unverified storage or quality combinations")

    def argument(name):
        if training_args.count(name) != 1:
            raise ValueError("The native training entry must explicitly specify "+name)
        return training_args[training_args.index(name)+1]

    output = Path(argument("--output"))
    if argument("--projection") != "full" or int(argument("--iterations")) not in (1100, 12000):
        raise ValueError("The isolated adapter validation requires full K and its declared training budget")
    if "--gsplat-build" not in training_args or int(argument("--expected-image-count")) <= 0:
        raise ValueError("The explicit compiled backend and admitted image count are required")
    contract_path = output.parent/(output.name+"_adapter_contract.json")
    snapshot = output.parent/(output.name+"_adapter_sources")
    if output.exists() or contract_path.exists() or snapshot.exists():
        raise FileExistsError("An isolated training output or adapter identity already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir()
    sources = [Path(__file__), Path(__file__).with_name("prefetch_camera_images.py"),
        Path(__file__).with_name("training_camera_metadata.py")]
    identities = [data_identity(path) for path in sources]
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
    external_identity = {"schema_version": 1, "configuration": config,
        "source_sha256": [r["sha256"] for r in identities],
        "camera_prefetch_state": "all initialization jobs consumed before the first optimization step",
        "training_metadata_state": "immutable binary camera view only; no observations deleted"}
    atomic_json(contract_path, {"status": "frozen_before_actual_training", "external_adapters": external_identity,
        "sources": identities, "configuration_source": str(args.adapter_config), "arguments": training_args,
        "native_complete_recovery_is_required": True, "counted": False})
    original_loader = train_case.camera_loader_adapter
    original_recovery = checkpoint_adapter.RecoveryState
    original_atomic_json = train_case.atomic_json
    original_argv = sys.argv
    receipts = []

    class BoundRecoveryState(original_recovery):
        def __init__(self, output, identity, interval, resume=None, crash_after=None):
            if "external_adapters" in identity:
                raise ValueError("The native recovery identity already has an external adapter binding")
            super().__init__(output, {**identity, "external_adapters": external_identity}, interval, resume, crash_after)

    @contextmanager
    def adapted(dataset, enabled=True):
        with ExitStack() as stack:
            cameras = stack.enter_context(original_loader(dataset, enabled=enabled))
            metadata = stack.enter_context(training_metadata_reader(config["training_metadata"]["enabled"]))
            prefetch = None
            if config["camera_prefetch"]["enabled"]:
                prefetch = stack.enter_context(prefetch_camera_images(config["camera_prefetch"]["workers"],
                    config["camera_prefetch"]["capacity"]))
            yield cameras
        receipts.append({"metadata": metadata, "camera_prefetch": prefetch})

    def configured_json(path, value):
        if Path(path) == output/"CONFIG.json":
            value = {**value, "external_adapters": external_identity, "external_adapter_sources": identities}
        original_atomic_json(path, value)

    try:
        train_case.camera_loader_adapter = adapted
        checkpoint_adapter.RecoveryState = BoundRecoveryState
        train_case.atomic_json = configured_json
        sys.argv = [str(Path(train_case.__file__)), *training_args]
        train_case.main()
        if identities != [data_identity(path) for path in sources]:
            raise ValueError("An identity-bound input adapter changed during actual training")
        state = json.loads((output/"STATE.json").read_text())
        if state["status"] != "completed" or len(receipts) != 1:
            raise ValueError("The native optimized entry did not complete its single full Scene")
        receipt = receipts[0]
        if config["camera_prefetch"]["enabled"] and (receipt["camera_prefetch"]["consumed"] != state["image_count"] or
                receipt["camera_prefetch"]["maximum_pending"] > 8):
            raise ValueError("An optimized Scene did not consume its complete bounded input queue")
        if config["training_metadata"]["enabled"] and (len(receipt["metadata"]) != 1 or
                receipt["metadata"][0]["image_count"] != state["image_count"]):
            raise ValueError("The training metadata view lost a declared image")
        atomic_json(output/"EXTERNAL_ADAPTER_RESULT.json", {"status": "native_training_completed_combination_acceptance_pending",
            "external_adapters": external_identity, "sources": identities, "actual_receipts": receipts,
            "image_count": state["image_count"], "resumed_iteration": state["resumed_iteration"], "counted": False})
    finally:
        train_case.camera_loader_adapter = original_loader
        checkpoint_adapter.RecoveryState = original_recovery
        train_case.atomic_json = original_atomic_json
        sys.argv = original_argv


if __name__ == "__main__":
    main()
