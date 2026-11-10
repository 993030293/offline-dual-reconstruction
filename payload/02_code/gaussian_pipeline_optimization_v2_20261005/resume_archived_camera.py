"""Expand a verified lossless transport artifact before native full-state recovery."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from checkpoint_archive import verified_checkpoint_stream
import train_camera_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--transport-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--resume-file" in remaining or remaining.count("--output") != 1:
        raise ValueError("The archive transport owns the single native resume argument")
    args.transport_output.mkdir(parents=True, exist_ok=False)
    sources = [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("checkpoint_archive.py"))]
    manifest = json.loads(args.checkpoint_archive.with_suffix(".ptgz.json").read_text())
    native_receipt = manifest.get("native_receipt")
    if not native_receipt or native_receipt.get("status") != "complete":
        raise ValueError("A native complete-state receipt is required for archive recovery")
    expanded = args.transport_output/"expanded.pth"
    before = time.perf_counter()
    stream, trace = verified_checkpoint_stream(args.checkpoint_archive)
    try:
        with expanded.open("xb") as target:
            shutil.copyfileobj(stream, target, length=4*1024**2)
    finally:
        stream.close()
    decoded = data_identity(expanded)
    if any(decoded[key] != manifest["native_payload"][key] for key in ("bytes", "sha256")):
        raise ValueError("The expanded native checkpoint changed its exact bytes")
    # Transport identity is separate from the unchanged algorithm and data identity.
    atomic_json(expanded.with_suffix(".json"), {**native_receipt, "checkpoint": decoded})
    receipt = {"status": "verified_native_payload_expanded_training_pending", "archive": data_identity(args.checkpoint_archive),
        "native_payload": manifest["native_payload"], "expanded": decoded, "sources": sources,
        "trace": trace, "verification_and_expansion_seconds": time.perf_counter()-before,
        "transient_uncompressed_disk_bytes": expanded.stat().st_size, "native_algorithm_identity_unchanged": True,
        "expanded_copy_removed": False, "counted": False}
    atomic_json(args.transport_output/"TRANSPORT.json", receipt)
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(train_camera_execution.__file__)), *remaining, "--resume-file", str(expanded)]
        train_camera_execution.main()
        output = Path(remaining[remaining.index("--output")+1])
        state = json.loads((output/"STATE.json").read_text())
        restored = json.loads((output/"RESTORE_CHECK.json").read_text())
        if state["status"] != "completed" or not all(restored["checks"].values()):
            raise ValueError("The archived native recovery did not preserve complete state")
        if sources != [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("checkpoint_archive.py"))]:
            raise ValueError("The archive transport changed during actual recovery")
        if expanded.parent.resolve() != args.transport_output.resolve() or data_identity(expanded) != decoded:
            raise ValueError("The owned recoverable expansion changed before removal")
        expanded.unlink()
        expanded.with_suffix(".json").unlink()
        receipt.update(status="native_complete_recovery_passed_quality_pending", expanded_copy_removed=True,
            restored_state=restored, actual_training_output=str(output))
    except BaseException as error:
        receipt.update(status="failed_expanded_copy_preserved", error=repr(error))
        raise
    finally:
        sys.argv = original_argv
        atomic_json(args.transport_output/"TRANSPORT.json", receipt)


if __name__ == "__main__":
    main()
