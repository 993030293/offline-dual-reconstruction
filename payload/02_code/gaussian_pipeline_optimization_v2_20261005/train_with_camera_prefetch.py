"""Install bounded image prefetch around the existing frozen training entry."""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
import train_case
from prefetch_camera_images import prefetch_camera_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch-mode", choices=["baseline", "prefetch"], required=True)
    parser.add_argument("--resource-validation", type=Path, required=True)
    args, training_args = parser.parse_known_args()
    resource = json.loads(args.resource_validation.read_text())
    if resource["status"] != "input_resource_guards_passed_training_pending" or not all(resource["checks"].values()):
        raise ValueError("Three full-input camera-loading pairs must pass before training")
    adapter_path = Path(__file__).with_name("prefetch_camera_images.py")
    if resource["sources"][0] != data_identity(adapter_path):
        raise ValueError("The resource-validated prefetch adapter changed")
    for forbidden in ("--checkpoint-every", "--resume-file", "--simulate-crash-after",
                      "--gpu-resident-inputs", "--compact-host-images", "--compact-image-cache"):
        if forbidden in training_args:
            raise ValueError("First prefetch training checks exclude unverified recovery and storage combinations")
    def argument(name):
        if training_args.count(name) != 1:
            raise ValueError("Training must explicitly specify "+name)
        return training_args[training_args.index(name)+1]
    output = Path(argument("--output"))
    if argument("--projection") != "full" or argument("--input-version") != "new_map_admission_v1" or argument("--expected-image-count") != "1016":
        raise ValueError("Prefetch training requires the registered new1016 full-K input")
    if argument("--iterations") not in ("1100", "12000"):
        raise ValueError("Prefetch validation budgets are1100 or12000 steps")
    contract_path = output.parent/(output.name+"_adapter_contract.json")
    if output.exists() or contract_path.exists():
        raise FileExistsError("The isolated training output or adapter contract already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), adapter_path, Path(train_case.__file__)]
    identities = [data_identity(path) for path in source_paths]
    snapshot = output.parent/(output.name+"_adapter_sources")
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot/path.name)
    atomic_json(contract_path, {"status": "frozen_before_training", "mode": args.prefetch_mode,
        "resource_validation": data_identity(args.resource_validation), "sources": identities,
        "arguments": training_args, "workers": 4, "capacity": 8, "source_snapshot": str(snapshot),
        "training_algorithm_changed": False, "checkpoint_combination_validated": False})
    original_adapter = train_case.camera_loader_adapter
    original_argv = sys.argv
    receipts = []
    @contextmanager
    def adapted(dataset, enabled=True):
        with original_adapter(dataset, enabled=enabled) as cameras, (prefetch_camera_images() if args.prefetch_mode == "prefetch" else nullcontext()) as receipt:
            yield cameras
        if receipt is not None:
            receipts.append(receipt)
    try:
        train_case.camera_loader_adapter = adapted
        sys.argv = [str(Path(train_case.__file__)), *training_args]
        train_case.main()
        if identities != [data_identity(path) for path in source_paths]:
            raise ValueError("The registered prefetch training sources changed")
        state = json.loads((output/"STATE.json").read_text())
        if state["status"] != "completed" or state["image_count"] != 1016:
            raise ValueError("The complete native training entry did not finish")
        if args.prefetch_mode == "prefetch" and (len(receipts) != 1 or receipts[0]["consumed"] != 1016 or receipts[0]["maximum_pending"] > 8):
            raise ValueError("The actual training did not use the bounded full-input prefetch")
        atomic_json(output/"PREFETCH_ADAPTER.json", {"status": "actual_training_completed", "mode": args.prefetch_mode,
            "sources": identities, "prefetch": receipts, "resource_validation": data_identity(args.resource_validation),
            "image_count": state["image_count"], "iteration": state["iteration"], "counted": False})
    finally:
        train_case.camera_loader_adapter = original_adapter
        sys.argv = original_argv


if __name__ == "__main__":
    main()
