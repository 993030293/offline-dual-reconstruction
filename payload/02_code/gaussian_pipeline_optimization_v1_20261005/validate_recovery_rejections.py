"""Reject stale, truncated and corrupted complete-state checkpoints."""

import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from checkpoint_adapter import RecoveryState
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.checkpoint.with_suffix(".json").read_text())
    identity = receipt["identity"]
    checks = {}

    def reject(label, path, contract, expected):
        state = RecoveryState(args.output, contract, 900, path)
        try:
            state.prepare(None, SimpleNamespace(iterations=identity["parameters"]["iterations"]), None)
        except expected as error:
            checks[label] = {"rejected": True, "error": str(error)}
        else:
            raise AssertionError("Invalid checkpoint was accepted")

    for field in ["input_sha256", "parameters", "source_sha256"]:
        changed = copy.deepcopy(identity)
        if field == "parameters":
            changed[field]["iterations"] += 1
        else:
            changed[field][0] = "0"*64
        reject(field+"_mismatch", args.checkpoint, changed, ValueError)
    if identity.get("gsplat_identity"):
        changed = copy.deepcopy(identity)
        changed["gsplat_identity"]["module"]["sha256"] = "0"*64
        reject("gsplat_module_mismatch", args.checkpoint, changed, ValueError)
        changed = copy.deepcopy(identity)
        changed["gsplat_identity"]["adapter"]["sha256"] = "0"*64
        reject("gsplat_adapter_mismatch", args.checkpoint, changed, ValueError)
    for field in ("gpu_resident_inputs", "trim_host_after_load"):
        if field in identity:
            changed = copy.deepcopy(identity)
            changed[field] = not changed[field]
            reject(field+"_mismatch", args.checkpoint, changed, ValueError)
    if "ply_serialization" in identity:
        changed = copy.deepcopy(identity)
        changed["ply_serialization"]["vectorized"] = not changed["ply_serialization"]["vectorized"]
        reject("ply_serialization_mismatch", args.checkpoint, changed, ValueError)
    incomplete = args.output/"incomplete.pth.tmp"
    incomplete.write_bytes(b"interrupted write without a completion receipt")
    reject("incomplete_file", incomplete, identity, FileNotFoundError)
    corrupt = args.output/"corrupt.pth"
    shutil.copyfile(args.checkpoint, corrupt)
    shutil.copyfile(args.checkpoint.with_suffix(".json"), corrupt.with_suffix(".json"))
    with corrupt.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    reject("payload_corruption", corrupt, identity, ValueError)
    atomic_json(args.output/"RESULT.json", {"status": "all_rejections_passed", "checks": checks,
        "valid_restore_verified_by": "separate actual continuation and RESTORE_CHECK.json"})
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
