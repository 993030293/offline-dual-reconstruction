"""Invoke unchanged complete recovery with a separately owned native mapped input."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from private_checkpoint_load import private_checkpoint_load
import train_camera_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--transport-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--resume-file" in remaining or remaining.count("--output") != 1:
        raise ValueError("The loader must own exactly one complete native resume input")
    args.transport_output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("private_checkpoint_load.py")]
    sources = [data_identity(p) for p in paths]
    native = data_identity(args.native_checkpoint)
    receipt = json.loads(args.native_checkpoint.with_suffix(".json").read_text())
    if receipt.get("status") != "complete" or receipt["checkpoint"]["sha256"] != native["sha256"]:
        raise ValueError("The known complete original native checkpoint receipt is missing")
    owned = args.transport_output/"native.pth"
    before = time.perf_counter()
    shutil.copyfile(args.native_checkpoint, owned)
    identity = data_identity(owned)
    if identity["sha256"] != native["sha256"] or identity["bytes"] != native["bytes"]:
        raise ValueError("The owned complete recovery input differs from the immutable original")
    atomic_json(owned.with_suffix(".json"), {**receipt, "checkpoint": identity})
    record = {"status": "verified_native_copy_restore_pending", "original": native, "owned": identity, "sources": sources,
        "copy_and_complete_verify_seconds": time.perf_counter()-before, "owned_copy_preserved": True, "counted": False}
    atomic_json(args.transport_output/"LOADER.json", record)
    previous = sys.argv
    try:
        with private_checkpoint_load(owned) as trace:
            sys.argv = [str(Path(train_camera_execution.__file__)), *remaining, "--resume-file", str(owned)]
            train_camera_execution.main()
        output = Path(remaining[remaining.index("--output")+1])
        state = json.loads((output/"STATE.json").read_text())
        restore = json.loads((output/"RESTORE_CHECK.json").read_text())
        if state["status"] != "completed" or not all(restore["checks"].values()) or len(trace["calls"]) != 2:
            raise ValueError("The mapped actual continuation or two complete native state loads failed")
        if data_identity(args.native_checkpoint) != native or data_identity(owned) != identity or sources != [data_identity(p) for p in paths]:
            raise ValueError("A mapped source, immutable original or private native copy changed")
        record.update(status="actual_native_mapped_recovery_complete_quality_pending", restored_state=restore,
            trace=trace, output=str(output), owned_source_and_original_unchanged=True)
    except BaseException as error:
        record.update(status="failed_original_and_owned_copy_preserved", error=repr(error))
        raise
    finally:
        sys.argv = previous
        atomic_json(args.transport_output/"LOADER.json", record)


if __name__ == "__main__":
    main()
